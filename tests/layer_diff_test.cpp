#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"
#include "graph/op_type.hpp"
#include "core/tensor.hpp"
#include "executor/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

using namespace powerserve;

static const char *MODEL_DIR = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT    = "In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of large language models.";

static int    N_THREADS    = 8;
static size_t BATCH_SIZE   = 4;
static int    DECODE_STEPS = 24;
static int    TOPK_PRINT   = 8;

// Layer-diff gating. Keep these looser than strict allclose, good for locating drift.
static float FAIL_MAX_ABS   = 1e-2f;
static float FAIL_MEAN_ABS  = 1e-3f;
static float FAIL_COSINE_LT = 0.9990f;

static const char *op_type_to_string(OpType t) {
    switch (t) {
    case OpType::GET_EMBEDDING: return "GET_EMBEDDING";
    case OpType::ADD: return "ADD";
    case OpType::MAT_MUL: return "MAT_MUL";
    case OpType::RMS_NORM: return "RMS_NORM";
    case OpType::SILU_HADAMARD: return "SILU_HADAMARD";
    case OpType::ROPE: return "ROPE";
    case OpType::SOFTMAX: return "SOFTMAX";
    case OpType::COPY: return "COPY";
    case OpType::ADD_CACHE: return "ADD_CACHE";
    case OpType::PERMUTE: return "PERMUTE";
    case OpType::CONT: return "CONT";
    case OpType::VIEW: return "VIEW";
    case OpType::SOFTMAX_EXT: return "SOFTMAX_EXT";
    case OpType::GET_MASK: return "GET_MASK";
    case OpType::TRANSPOSE: return "TRANSPOSE";
    case OpType::PRINT: return "PRINT";
    default: return "UNKNOWN";
    }
}

static inline uint64_t op_out_key(int op_idx, int out_idx) {
    return (uint64_t(uint32_t(op_idx)) << 32) | uint32_t(out_idx);
}

static int argmax_span(std::span<const float> v) {
    if (v.empty()) return 0;
    int bi = 0;
    float bv = v[0];
    for (int i = 1; i < (int)v.size(); ++i) {
        if (v[i] > bv) {
            bv = v[i];
            bi = i;
        }
    }
    return bi;
}

static void dump_topk(std::span<const float> logits, int k, const char *tag) {
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, (int)idx.size()), idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("[%s top%d] ", tag, k);
    for (int i = 0; i < k && i < (int)idx.size(); ++i) {
        int id = idx[i];
        std::printf("(%d %.6f) ", id, logits[id]);
    }
    std::printf("\n");
}

static std::vector<float> tensor_to_f32_vec_cpu(const Tensor *t) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32 || !t->m_data) return out;
    auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get());
    if (!cb) return out;

    const auto shape = t->m_shape;
    const auto stride = cb->m_stride;
    out.resize(t->n_elements());

    size_t p = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    float *ptr = (float *)((char *)cb->m_data +
                                           i3 * stride[3] + i2 * stride[2] +
                                           i1 * stride[1] + i0 * stride[0]);
                    out[p++] = *ptr;
                }
            }
        }
    }
    return out;
}

static std::vector<float> tensor_to_f32_vec_opencl(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32 || !t->m_data) return out;
    POWERSERVE_ASSERT(cl_backend);
    Tensor tmp_cpu(DataType::FP32, t->m_shape);
    tmp_cpu.m_data = CPUBuffer::create_buffer<float>(t->m_shape);
    cl_backend->copy(&tmp_cpu, t);
    return tensor_to_f32_vec_cpu(&tmp_cpu);
}

static std::vector<float> tensor_to_f32_vec_any(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    if (!t || t->m_dtype != DataType::FP32 || !t->m_data) return {};
    if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) return tensor_to_f32_vec_opencl(t, cl_backend);
    if (dynamic_cast<CPUBuffer*>(t->m_data.get())) return tensor_to_f32_vec_cpu(t);
    return {};
}

struct DiffRec {
    int op_idx = -1;
    int out_idx = -1;
    int layer = -1;
    OpType op = OpType::PRINT;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float cosine = 1.0f;
    size_t worst_i = 0;
    float ref_v = 0.0f;
    float ocl_v = 0.0f;
};

struct LayerAgg {
    int count = 0;
    float max_abs = 0.0f;
    float mean_abs_acc = 0.0f;
    float min_cosine = 1.0f;
};

static DiffRec diff_vecs(const std::vector<float> &ref, const std::vector<float> &ocl) {
    DiffRec r;
    if (ref.size() != ocl.size() || ref.empty()) {
        r.max_abs = std::numeric_limits<float>::infinity();
        r.mean_abs = std::numeric_limits<float>::infinity();
        r.cosine = -1.0f;
        return r;
    }
    double sum = 0.0;
    double dot = 0.0, nr = 0.0, no = 0.0;
    float mx = 0.0f;
    size_t mi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float d = std::fabs(ref[i] - ocl[i]);
        sum += d;
        if (d > mx) {
            mx = d;
            mi = i;
        }
        dot += (double)ref[i] * (double)ocl[i];
        nr += (double)ref[i] * (double)ref[i];
        no += (double)ocl[i] * (double)ocl[i];
    }
    r.max_abs = mx;
    r.mean_abs = (float)(sum / (double)ref.size());
    r.cosine = (nr > 0.0 && no > 0.0) ? (float)(dot / (std::sqrt(nr) * std::sqrt(no))) : 0.0f;
    r.worst_i = mi;
    r.ref_v = ref[mi];
    r.ocl_v = ocl[mi];
    return r;
}

int main() {
    POWERSERVE_LOG_INFO("==== layer_diff_test (teacher-forcing, ggml vs opencl) ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("THREADS={}, BATCH_SIZE={}, DECODE_STEPS={}", N_THREADS, BATCH_SIZE, DECODE_STEPS);

    HyperParams hparams;
    hparams.n_threads = N_THREADS;
    hparams.batch_size = BATCH_SIZE;

    auto model_ggml = load_model(MODEL_DIR);
    auto model_ocl  = load_model(MODEL_DIR);
    model_ggml->m_attn = std::make_shared<powerserve::NormAttention>(model_ggml->m_config->llm, model_ggml->m_weights);
    model_ocl->m_attn  = std::make_shared<powerserve::NormAttention>(model_ocl->m_config->llm,  model_ocl->m_weights);
    model_ggml->m_config->model_id = "ggml_ref";
    model_ocl->m_config->model_id  = "opencl_test";

    auto platform = std::make_shared<Platform>();
    model_ggml->m_platform = platform;
    model_ocl->m_platform  = platform;

    platform->init_ggml_backend(model_ggml->m_config, hparams);
    platform->init_ggml_backend(model_ocl->m_config, hparams);
    platform->init_opencl_backend(model_ocl->m_config, hparams);
    platform->ggml_backends[model_ggml->m_config->model_id]->setup_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->setup_threadpool();

    std::string vocab_path = std::string(MODEL_DIR) + "/" + MODEL_VOCAB_FILENAME;
    Tokenizer tokenizer(vocab_path);
    std::vector<Token> tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (tokens.empty()) {
        POWERSERVE_LOG_ERROR("Prompt tokenization returned empty tokens");
        return 1;
    }

    auto run_and_compare = [&](const std::vector<Token> &in_tokens,
                               const std::vector<int> &in_pos,
                               const CausalAttentionMask &mask,
                               bool lm_head,
                               const char *phase,
                               int step) -> std::pair<decltype(model_ggml->forward(in_tokens, in_pos, mask, lm_head)),
                                                      decltype(model_ocl->forward(in_tokens, in_pos, mask, lm_head))> {
        std::unordered_map<uint64_t, std::vector<float>> gg_outs;
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out || out->m_dtype != DataType::FP32 || !out->m_data) continue;
                gg_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
            }
        });
        auto ret_g = model_ggml->forward(in_tokens, in_pos, mask, lm_head);
        set_op_after_exec_hook(nullptr);

        auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            platform->get_backend(model_ocl->m_config->model_id));
        POWERSERVE_ASSERT(cl_backend);

        std::vector<DiffRec> diffs;
        std::unordered_map<int, LayerAgg> by_layer;
        int layer_cursor = 0;

        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            if (op->op == OpType::ADD_CACHE) {
                ++layer_cursor;
            }
            Tensor *out = (op->next.empty() ? nullptr : op->next[0]->tensor());
            if (!out || out->m_dtype != DataType::FP32 || !out->m_data) return;

            auto it = gg_outs.find(op_out_key(op_idx, 0));
            if (it == gg_outs.end()) return;
            auto ocl = tensor_to_f32_vec_any(out, cl_backend);
            if (ocl.empty()) return;
            DiffRec d = diff_vecs(it->second, ocl);
            d.op_idx = op_idx;
            d.out_idx = 0;
            d.op = op->op;
            d.layer = layer_cursor;
            diffs.push_back(d);

            auto &agg = by_layer[d.layer];
            agg.count++;
            agg.max_abs = std::max(agg.max_abs, d.max_abs);
            agg.mean_abs_acc += d.mean_abs;
            agg.min_cosine = std::min(agg.min_cosine, d.cosine);
        });
        auto ret_o = model_ocl->forward(in_tokens, in_pos, mask, lm_head);
        set_op_after_exec_hook(nullptr);

        if (!diffs.empty()) {
            std::sort(diffs.begin(), diffs.end(), [](const DiffRec &a, const DiffRec &b) {
                return a.max_abs > b.max_abs;
            });
            std::printf("\n[%s step=%d] top-%d worst ops\n", phase, step, TOPK_PRINT);
            for (int i = 0; i < TOPK_PRINT && i < (int)diffs.size(); ++i) {
                const auto &d = diffs[i];
                std::printf("  #%d op#%d layer=%d type=%s max_abs=%.6g mean_abs=%.6g cosine=%.8f worst_i=%zu ref=%.6g ocl=%.6g\n",
                            i, d.op_idx, d.layer, op_type_to_string(d.op),
                            d.max_abs, d.mean_abs, d.cosine, d.worst_i, d.ref_v, d.ocl_v);
            }
            std::printf("[%s step=%d] per-layer summary\n", phase, step);
            std::vector<int> lids;
            lids.reserve(by_layer.size());
            for (const auto &kv : by_layer) lids.push_back(kv.first);
            std::sort(lids.begin(), lids.end());
            for (int lid : lids) {
                const auto &a = by_layer[lid];
                const float mean_abs = a.count > 0 ? (a.mean_abs_acc / (float)a.count) : 0.0f;
                std::printf("  layer=%d count=%d max_abs=%.6g mean_abs=%.6g min_cosine=%.8f\n",
                            lid, a.count, a.max_abs, mean_abs, a.min_cosine);
            }

            const auto &w = diffs.front();
            if (w.max_abs > FAIL_MAX_ABS || w.mean_abs > FAIL_MEAN_ABS || w.cosine < FAIL_COSINE_LT) {
                std::printf("[FAIL-TRIGGER] phase=%s step=%d worst: op#%d layer=%d type=%s max_abs=%.6g mean_abs=%.6g cosine=%.8f\n",
                            phase, step, w.op_idx, w.layer, op_type_to_string(w.op), w.max_abs, w.mean_abs, w.cosine);
                auto lg = ret_g.logits_vector.back();
                auto lo = ret_o.logits_vector.back();
                dump_topk(lg, TOPK_PRINT, "ggml");
                dump_topk(lo, TOPK_PRINT, "opencl");
            }
        }

        return {ret_g, ret_o};
    };

    std::vector<int> pos(tokens.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int)i;

    // Prefill
    {
        auto mask = CausalAttentionMask(tokens.size());
        auto [ret_g, ret_o] = run_and_compare(tokens, pos, mask, /*lm_head=*/true, "PREFILL", 0);
        auto lg = ret_g.logits_vector.back();
        auto lo = ret_o.logits_vector.back();
        std::printf("[PREFILL] argmax ggml=%d opencl=%d\n", argmax_span(lg), argmax_span(lo));
    }

    // Decode teacher-forcing: feed ggml argmax as next token.
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;
        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        if (tokens.size() > 1) {
            std::vector<Token> prefill_tokens(tokens.begin(), tokens.end() - 1);
            std::vector<int> prefill_pos(prefill_tokens.size());
            for (size_t i = 0; i < prefill_pos.size(); ++i) prefill_pos[i] = (int)i;
            auto prefill_mask = CausalAttentionMask(prefill_tokens.size());
            (void)run_and_compare(prefill_tokens, prefill_pos, prefill_mask, /*lm_head=*/false, "DECODE_PREFILL", -1);
        }

        int token_in = tokens.back();
        for (int step = 0; step < DECODE_STEPS; ++step) {
            size_t kv_pos_g = platform->get_kv_position(id_g);
            size_t kv_pos_o = platform->get_kv_position(id_o);
            POWERSERVE_ASSERT(kv_pos_g == kv_pos_o);

            std::vector<Token> in_tok(1, token_in);
            std::vector<int> in_pos(1, (int)kv_pos_g);
            auto mask = CausalAttentionMask(1);
            auto [ret_g, ret_o] = run_and_compare(in_tok, in_pos, mask, /*lm_head=*/true, "DECODE", step);
            auto lg = ret_g.logits_vector.back();
            auto lo = ret_o.logits_vector.back();
            int next_token = argmax_span(lg);
            std::printf("[DECODE step=%d] kv_pos=%zu token_in=%d next(ggml)=%d next(ocl)=%d\n",
                        step, kv_pos_g, token_in, next_token, argmax_span(lo));
            token_in = next_token;
        }
    }

    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    POWERSERVE_LOG_INFO("layer_diff_test finished");
    return 0;
}

