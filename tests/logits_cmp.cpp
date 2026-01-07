// tests/logits_cmp.cpp
#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/model_loader.hpp"
#include "tokenizer/tokenizer.hpp"
#include "core/config.hpp"
#include "model/module/norm_attention.hpp"
#include "executor/executor.hpp"
#include "graph/op_type.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <typeinfo>

using namespace powerserve;

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
#if defined(POWERSERVE_WITH_QNN)
    case OpType::QNN_FORWARD: return "QNN_FORWARD";
    case OpType::QNN_FORWARD_VL: return "QNN_FORWARD_VL";
#endif
    default: return "UNKNOWN";
    }
}

// ======================
// 写死参数
// ======================
static const char *MODEL_DIR   = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT      = "你好，请介绍你自己";

static int    MAX_STEPS   = 64;
static int    N_THREADS   = 8;
static size_t BATCH_SIZE  = 1;

static float ATOL = 1e-3f;
static float RTOL = 1e-3f;
static int   TOPK = 10;

static inline bool allclose_span(std::span<const float> a, std::span<const float> b,
                                 float atol, float rtol, size_t *bad_i=nullptr, float *diff=nullptr) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::fabs(b[i]);
        if (!(d <= tol)) {
            if (bad_i) *bad_i = i;
            if (diff) *diff = d;
            return false;
        }
    }
    return true;
}

static void dump_topk(std::span<const float> logits, int k, const char *tag) {
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    fmt::print("[{} top{}] ", tag, k);
    for (int i = 0; i < k; ++i) {
        int id = idx[i];
        fmt::print("({} {:.6f}) ", id, logits[id]);
    }
    fmt::print("\n");
}

static int argmax_token(std::span<const float> logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static std::vector<float> tensor_to_f32_vec_cpu(const Tensor *t) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32) return out;
    if (!t->m_data) return out;

    auto shape = t->m_shape;
    size_t n = t->n_elements();
    out.resize(n);

    auto *base = t->m_data.get();
    auto *cb = dynamic_cast<CPUBuffer*>(base);
    if (!cb) return {};

    auto stride = cb->m_stride;

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    float *ptr = (float *)((char *)cb->m_data +
                                           i3 * stride[3] + i2 * stride[2] +
                                           i1 * stride[1] + i0 * stride[0]);
                    out[idx++] = *ptr;
                }
            }
        }
    }
    return out;
}

static std::vector<float> tensor_to_f32_vec_opencl(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32) return out;
    if (!t->m_data) return out;
    POWERSERVE_ASSERT(cl_backend);

    Tensor tmp_cpu(DataType::FP32, t->m_shape);
    tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(t->m_shape);

    cl_backend->copy(&tmp_cpu, t);
    return tensor_to_f32_vec_cpu(&tmp_cpu);
}

// ==== NEW ====：对“CPU / OpenCL”统一 flatten
static std::vector<float> tensor_to_f32_vec_any(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    if (!t || t->m_dtype != DataType::FP32) return {};
    if (!t->m_data) return {};
#if defined(POWERSERVE_WITH_OPENCL)
    if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
        return tensor_to_f32_vec_opencl(t, cl_backend);
    }
#endif
    if (dynamic_cast<CPUBuffer*>(t->m_data.get())) {
        return tensor_to_f32_vec_cpu(t);
    }
    return {};
}

// ==== NEW ====：避免 get<T>() bad cast abort，改用 dynamic_cast
static void print_tensor_meta(const Tensor *t, const char *tag) {
    if (!t) {
        fmt::print("  {}: <null>\n", tag);
        return;
    }

    auto shape = t->m_shape;

    if (!t->m_data) {
        fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] strideB=<view(null)>\n",
                   tag, (int)t->m_dtype,
                   shape[0], shape[1], shape[2], shape[3]);
        return;
    }

    std::array<size_t,4> stride = {0,0,0,0};
    const char *buf_kind = "unknown";

#if defined(POWERSERVE_WITH_OPENCL)
    if (auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
        stride   = {clb->m_stride[0], clb->m_stride[1], clb->m_stride[2], clb->m_stride[3]};
        buf_kind = "opencl";
    } else
#endif
    if (auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get())) {
        stride   = {cb->m_stride[0], cb->m_stride[1], cb->m_stride[2], cb->m_stride[3]};
        buf_kind = "cpu";
    } else {
        buf_kind = typeid(*t->m_data).name();
    }

    fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] strideB=[{}, {}, {}, {}] buf={}\n",
               tag, (int)t->m_dtype,
               shape[0], shape[1], shape[2], shape[3],
               stride[0], stride[1], stride[2], stride[3],
               buf_kind);
}

static inline uint64_t op_out_key(int op_idx, int out_idx) {
    return (uint64_t(uint32_t(op_idx)) << 32) | uint32_t(out_idx);
}

int main() {
    POWERSERVE_LOG_INFO("==== Qwen2 logits compare test (ggml vs opencl) ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("MAX_STEPS={}, THREADS={}, BATCH_SIZE={}", MAX_STEPS, N_THREADS, BATCH_SIZE);

#if !defined(POWERSERVE_WITH_OPENCL)
    POWERSERVE_LOG_ERROR("POWERSERVE_WITH_OPENCL not enabled in compile.");
    return 1;
#endif

    HyperParams hparams;
    hparams.n_threads  = N_THREADS;
    hparams.batch_size = BATCH_SIZE;

    // ----------------------------
    // 1) load 两份模型（同一 GGUF）
    // ----------------------------
    auto model_ggml = load_model(MODEL_DIR);
    auto model_ocl  = load_model(MODEL_DIR);

    model_ggml->m_attn = std::make_shared<powerserve::NormAttention>(model_ggml->m_config->llm, model_ggml->m_weights);
    model_ocl->m_attn  = std::make_shared<powerserve::NormAttention>(model_ocl->m_config->llm,  model_ocl->m_weights);

    model_ggml->m_config->model_id = "ggml_ref";
    model_ocl->m_config->model_id  = "opencl_test";

    // ==== NEW ====：双 Platform，彻底隔离 platform 内部状态
    auto platform_g = std::make_shared<Platform>();
    auto platform_o = std::make_shared<Platform>();
    model_ggml->m_platform = platform_g;
    model_ocl->m_platform  = platform_o;

    // ----------------------------
    // 2) init backend
    // ----------------------------
    platform_g->init_ggml_backend(model_ggml->m_config, hparams);

    // opencl platform: 需要 ggml backend（kv 管理/以及你未来可能用到的平台接口），也需要 opencl backend
    platform_o->init_ggml_backend(model_ocl->m_config, hparams);
    platform_o->init_opencl_backend(model_ocl->m_config, hparams);

    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform_g->reset_kv_position(id_g);
        platform_o->reset_kv_position(id_o);

        platform_g->ggml_backends[id_g]->setup_threadpool();
        platform_o->ggml_backends[id_o]->setup_threadpool();
    }

    // ----------------------------
    // 3) tokenizer + tokenize
    // ----------------------------
    std::string vocab_path = std::string(MODEL_DIR) + "/" + MODEL_VOCAB_FILENAME;
    Tokenizer tokenizer(vocab_path);

    std::vector<Token> tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (tokens.empty()) {
        POWERSERVE_LOG_ERROR("Prompt tokenization returned empty tokens.");
        return 1;
    }

    std::vector<int> pos(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) pos[i] = (int)i;

    POWERSERVE_LOG_INFO("Prompt token count = {}", tokens.size());

    // ----------------------------
    // 4) PREFILL compare
    // ----------------------------
    {
        auto mask = CausalAttentionMask(tokens.size());

        // OpenCL backend handle
        auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            platform_o->get_backend(model_ocl->m_config->model_id));
        POWERSERVE_ASSERT(cl_backend && "OpenCLBackend is null");

        // Pass 1: GGML run, record per-op outputs + (op#2 MATMUL) input B
        std::unordered_map<uint64_t, std::vector<float>> ggml_op_outs;

        // ==== NEW ====：记录 op#2 MAT_MUL 的输入 B（prev[1]）
        std::vector<float> ggml_op2_B;

        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            // record outputs
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out) continue;
                if (!out->m_data) continue; // skip view
                if (out->m_dtype != DataType::FP32) continue;

                ggml_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
            }

            // record op#2 MAT_MUL input B
            if (op_idx == 2 && op->op == OpType::MAT_MUL) {
                Tensor *B = (op->prev.size() > 1) ? op->prev[1]->tensor() : nullptr;
                if (B && B->m_dtype == DataType::FP32 && B->m_data) {
                    ggml_op2_B = tensor_to_f32_vec_cpu(B);
                }
            }
        });

        platform_g->reset_kv_position(model_ggml->m_config->model_id);
        auto ret_g = model_ggml->forward(tokens, pos, mask, true);
        set_op_after_exec_hook(nullptr);

        // Pass 2: OpenCL run, compare per-op outputs; additionally compare op#2 input B
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {

            // ==== NEW ====：在“第一个可疑点”直接比 input B
            if (op_idx == 2 && op->op == OpType::MAT_MUL) {
                Tensor *B = (op->prev.size() > 1) ? op->prev[1]->tensor() : nullptr;

                auto ocl_B = tensor_to_f32_vec_any(B, cl_backend);
                if (!ggml_op2_B.empty() && !ocl_B.empty()) {
                    size_t bad_i = 0;
                    float diff = 0.f;
                    if (!allclose_span(ocl_B, ggml_op2_B, ATOL, RTOL, &bad_i, &diff)) {
                        fmt::print("\n[FIRST INPUT MISMATCH]\n");
                        fmt::print("  op#{} type={} INPUT=B(prev[1])\n", op_idx, op_type_to_string(op->op));
                        print_tensor_meta(B, "in1(B)");
                        fmt::print("  bad_i={} ggml_B={} ocl_B={} diff={}\n",
                                   bad_i, ggml_op2_B[bad_i], ocl_B[bad_i], diff);
                        std::exit(1);
                    }
                } else {
                    fmt::print("\n[WARN] op#2 MAT_MUL input B not captured (ggml_B={} ocl_B={})\n",
                               ggml_op2_B.size(), ocl_B.size());
                }
            }

            // regular per-op output compare (same as before)
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out) continue;
                if (!out->m_data) continue;
                if (out->m_dtype != DataType::FP32) continue;

                auto it = ggml_op_outs.find(op_out_key(op_idx, oi));
                if (it == ggml_op_outs.end()) continue;

                auto ocl_vec = tensor_to_f32_vec_opencl(out, cl_backend);
                auto &gg_vec = it->second;

                if (ocl_vec.size() != gg_vec.size()) {
                    fmt::print("\n[FIRST MISMATCH]\n");
                    fmt::print("  op#{} type={} out#{} (size mismatch ocl={} ggml={})\n",
                               op_idx, op_type_to_string(op->op), oi, ocl_vec.size(), gg_vec.size());
                    print_tensor_meta(out, "out");
                    std::exit(1);
                }

                size_t bad_i = 0;
                float diff = 0.f;
                if (!allclose_span(ocl_vec, gg_vec, ATOL, RTOL, &bad_i, &diff)) {
                    fmt::print("\n[FIRST MISMATCH]\n");
                    fmt::print("  op#{} type={} out#{}\n", op_idx, op_type_to_string(op->op), oi);
                    print_tensor_meta(out, "out");

                    if (op->op == OpType::MAT_MUL) {
                        Tensor *A = op->prev.size() > 0 ? op->prev[0]->tensor() : nullptr;
                        Tensor *B = op->prev.size() > 1 ? op->prev[1]->tensor() : nullptr;
                        print_tensor_meta(A, "in0(A)");
                        print_tensor_meta(B, "in1(B)");
                    }

                    fmt::print("  bad_i={} ggml={} ocl={} diff={}\n",
                               bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff);
                    std::exit(1);
                }
            }
        });

        platform_o->reset_kv_position(model_ocl->m_config->model_id);
        auto ret_o = model_ocl->forward(tokens, pos, mask, true);
        set_op_after_exec_hook(nullptr);

        // optional end-to-end logits compare
        auto lg = ret_g.logits_vector.back();
        auto lo = ret_o.logits_vector.back();
        size_t bad_i = 0;
        float diff = 0;
        if (!allclose_span(lo, lg, ATOL, RTOL, &bad_i, &diff)) {
            fmt::print("PREFILL logits mismatch (end-to-end): bad_i={}, ocl={}, ggml={}, diff={}\n",
                       bad_i, lo[bad_i], lg[bad_i], diff);
            dump_topk(lg, TOPK, "ggml");
            dump_topk(lo, TOPK, "opencl");
            return 1;
        }

        POWERSERVE_LOG_INFO("Prefill per-op compare PASS");
    }

    // ----------------------------
    // 5) DECODE loop compare
    // ----------------------------
    for (int step = 0; step < MAX_STEPS; ++step) {
        std::vector<Token> one_tok = { tokens.back() };
        std::vector<int>   one_pos = { (int)(tokens.size() - 1) };
        auto mask = CausalAttentionMask(tokens.size());

        auto ret_g = model_ggml->forward(one_tok, one_pos, mask, true);
        auto ret_o = model_ocl->forward(one_tok, one_pos, mask, true);

        auto lg = ret_g.logits_vector.back();
        auto lo = ret_o.logits_vector.back();

        size_t bad_i = 0;
        float diff = 0;
        if (!allclose_span(lo, lg, ATOL, RTOL, &bad_i, &diff)) {
            fmt::print("STEP {} logits mismatch: bad_i={}, ocl={}, ggml={}, diff={}\n",
                       step, bad_i, lo[bad_i], lg[bad_i], diff);
            dump_topk(lg, TOPK, "ggml");
            dump_topk(lo, TOPK, "opencl");
            return 1;
        }

        int next = argmax_token(lg);
        tokens.push_back(next);

        if (next == tokenizer.bos_token() || tokenizer.should_stop(next)) {
            POWERSERVE_LOG_INFO("Stop at step {} (token={})", step, next);
            break;
        }
    }

    POWERSERVE_LOG_INFO("All steps PASS: logits match within atol={} rtol={}", ATOL, RTOL);

    platform_g->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform_o->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
