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
#include <unordered_set>

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
// 你只需要改这里：写死参数
// ======================
static const char *MODEL_DIR   = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT      = "你好，请介绍你自己";

// decode 比较多少步
static int MAX_STEPS  = 64;
static int N_THREADS  = 8;
static size_t BATCH_SIZE = 1;

// compare 阈值（端到端误差会比单 op 大，先宽松一点）
static float ATOL = 1e-3f;
static float RTOL = 1e-3f;
static int TOPK   = 10;

// ======================
// helper: allclose + topk
// ======================
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

    auto shape = t->m_shape;
    size_t n = t->n_elements();
    out.resize(n);

    auto &cb = const_cast<Tensor*>(t)->get<CPUBuffer>();
    auto stride = cb.m_stride;

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    float *ptr = (float *)((char *)cb.m_data +
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
    POWERSERVE_ASSERT(cl_backend);

    // create tmp cpu tensor with same shape
    Tensor tmp_cpu(DataType::FP32, t->m_shape);
    tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(t->m_shape);

    // D2H copy
    cl_backend->copy(&tmp_cpu, t);

    // flatten
    return tensor_to_f32_vec_cpu(&tmp_cpu);
}

static std::vector<float> tensor_to_f32_vec_any(const Tensor *t,
                                                powerserve::opencl::OpenCLBackend *cl_backend) {
    if (!t || t->m_dtype != DataType::FP32) return {};
    if (!t->m_data) return {}; // view 没法直接 dump

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


static bool is_opencl_tensor(const Tensor *t) {
#if defined(POWERSERVE_WITH_OPENCL)
    return t && t->m_data && dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get());
#else
    return false;
#endif
}



static void print_tensor_meta(const Tensor *t, const char *tag) {
    if (!t) {
        fmt::print("  {}: <null>\n", tag);
        return;
    }

    auto shape = t->m_shape;

#if defined(POWERSERVE_WITH_OPENCL)
    if (t->m_data) {
        if (auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
            fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] strideB=[{}, {}, {}, {}] [OpenCLBuffer]\n",
                       tag, (int)t->m_dtype,
                       shape[0], shape[1], shape[2], shape[3],
                       clb->m_stride[0], clb->m_stride[1], clb->m_stride[2], clb->m_stride[3]);
            return;
        }
    }
#endif

    if (t->m_data) {
        if (auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get())) {
            fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] strideB=[{}, {}, {}, {}] [CPUBuffer]\n",
                       tag, (int)t->m_dtype,
                       shape[0], shape[1], shape[2], shape[3],
                       cb->m_stride[0], cb->m_stride[1], cb->m_stride[2], cb->m_stride[3]);
            return;
        }
    }

    // view / unknown
    fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] [NO_DATA/UNKNOWN]\n",
               tag, (int)t->m_dtype, shape[0], shape[1], shape[2], shape[3]);
}


static void dump_f32_sample(const std::vector<float> &v, const char *tag, int n = 8) {
    if (v.empty()) return;
    fmt::print("  {} sample: ", tag);
    for (int i = 0; i < n && i < (int)v.size(); ++i) {
        fmt::print("{:.6f} ", v[i]);
    }
    fmt::print("\n");
}

static inline uint64_t op_out_key(int op_idx, int out_idx) {
    return (uint64_t(uint32_t(op_idx)) << 32) | uint32_t(out_idx);
}

// producer_map 打印版本：额外输出 ptr / is_view / has_data / producer_op
static void print_tensor_meta2(const Tensor *t, const char *tag,
                               const std::unordered_map<const Tensor*, int> &producer) {
    if (!t) {
        fmt::print("  {}: <null>\n", tag);
        return;
    }

    print_tensor_meta(t, tag);

    auto it = producer.find(t);
    int p = (it == producer.end()) ? -1 : it->second;

    bool is_view = (t->m_data == nullptr);
    bool is_cl = is_opencl_tensor(t);

    fmt::print("    ptr={} producer_op={} is_view={} has_data={} {}\n",
               (const void*)t, p, is_view, (t->m_data != nullptr),
               is_cl ? "[CL]" : "[CPU]");
}

int main() {
    POWERSERVE_LOG_INFO("==== Qwen2 logits compare test (ggml vs opencl) ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("MAX_STEPS={}, THREADS={}, BATCH_SIZE={}", MAX_STEPS, N_THREADS, BATCH_SIZE);

#if !defined(POWERSERVE_WITH_OPENCL)
    POWERSERVE_LOG_ERROR("POWERSERVE_WITH_OPENCL not enabled in compile.");
    return 1;
#endif

    // ----------------------------
    // 1) 构造 hyper params
    // ----------------------------
    HyperParams hparams;
    hparams.n_threads = N_THREADS;
    hparams.batch_size = BATCH_SIZE;
    // sampler_config 在此测试中不用，因为我们直接 argmax

    // ----------------------------
    // 2) load 两份模型（同一 GGUF）
    // ----------------------------
    auto model_ggml = load_model(MODEL_DIR);
    auto model_ocl  = load_model(MODEL_DIR);

    model_ggml->m_attn = std::make_shared<powerserve::NormAttention>(model_ggml->m_config->llm, model_ggml->m_weights);
    model_ocl->m_attn  = std::make_shared<powerserve::NormAttention>(model_ocl->m_config->llm,  model_ocl->m_weights);

    // 重要：给它们不同 model_id，否则 platform 里的 backend/cache 会覆盖
    model_ggml->m_config->model_id = "ggml_ref";
    model_ocl->m_config->model_id  = "opencl_test";

    // 共享 platform
    auto platform = std::make_shared<Platform>();
    model_ggml->m_platform = platform;
    model_ocl->m_platform  = platform;

    // ----------------------------
    // 3) init backend
    // ----------------------------
    platform->init_ggml_backend(model_ggml->m_config, hparams);
    platform->init_ggml_backend(model_ocl->m_config,  hparams); // 给 opencl model 也 init ggml（fallback 会用）
    platform->init_opencl_backend(model_ocl->m_config, hparams);
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        platform->ggml_backends[id_g]->setup_threadpool();
        platform->ggml_backends[id_o]->setup_threadpool();
    }

    // ----------------------------
    // 4) tokenizer
    // ----------------------------
    std::string vocab_path = std::string(MODEL_DIR) + "/" + MODEL_VOCAB_FILENAME;
    Tokenizer tokenizer(vocab_path);

    // ----------------------------
    // 5) tokenize prompt
    // ----------------------------
    std::vector<Token> tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (tokens.empty()) {
        POWERSERVE_LOG_ERROR("Prompt tokenization returned empty tokens.");
        return 1;
    }

    std::vector<int> pos(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) pos[i] = (int)i;

    POWERSERVE_LOG_INFO("Prompt token count = {}", tokens.size());

    // ----------------------------
    // 6) PREFILL compare (full prompt forward)
    // ----------------------------
    {
        auto mask = CausalAttentionMask(tokens.size());

        // ----------------------------------
        // Pass 1: run GGML, record each op output
        // ----------------------------------
        std::unordered_map<uint64_t, std::vector<float>> ggml_op_outs;

        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out) continue;
                if (!out->m_data) continue; // 跳过 view 输出（保持你原逻辑）
                if (out->m_dtype != DataType::FP32) continue;

                ggml_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
            }
        });

        auto ret_g = model_ggml->forward(tokens, pos, mask, true);

        // disable hook between passes
        set_op_after_exec_hook(nullptr);

        // ----------------------------------
        // Pass 2: run OpenCL, compare each op output with GGML
        // ----------------------------------
        auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            platform->get_backend(model_ocl->m_config->model_id));
        POWERSERVE_ASSERT(cl_backend && "OpenCLBackend is null");

        // ✅ 构建 producer map：tensor* -> producer op_idx
        std::unordered_map<const Tensor*, int> producer;
        {
            // 用 GGML pass 的 hook 无法拿到 graph，但 op 本身带 next tensor
            // 所以这里在第一次 compare hook 时 lazily build
            // 简单做法：在第一次进入 hook 时 build 一次
        }
        bool producer_built = false;

        // --------- Backtrace state (OpenCL pass) ----------
        struct MismatchRecord {
            int op_idx;
            int out_idx;
            size_t bad_i;
            float ggml_v;
            float ocl_v;
            float diff;
        };

        std::vector<MismatchRecord> mismatch_chain;
        std::unordered_set<int> want_check_ops;     // ops we want to check ASAP
        std::unordered_set<int> visited_backtrace;  // prevent infinite loop
        bool first_mismatch_seen = false;

        // helper: dump mismatch + inputs meta
        auto dump_mismatch_detail = [&](int op_idx, const OpNode *op, int out_idx,
                                        const Tensor *out,
                                        const std::vector<float> &gg_vec,
                                        const std::vector<float> &ocl_vec,
                                        size_t bad_i, float diff) {
            fmt::print("\n[MISMATCH] op#{} type={} out#{}\n", op_idx, op_type_to_string(op->op), out_idx);
            fmt::print("  bad_i={} ggml={} ocl={} diff={}\n", bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff);

            print_tensor_meta2(out, "out", producer);

            // print inputs
            for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                Tensor *in = op->prev[pi]->tensor();
                char name[64];
                std::snprintf(name, sizeof(name), "in[%d]", pi);
                print_tensor_meta2(in, name, producer);
            }

            // show activation producer (if exists)
            if ((int)op->prev.size() > 1) {
                Tensor *act = op->prev[1]->tensor();
                if (act) {
                    auto itp = producer.find(act);
                    int p = (itp == producer.end()) ? -1 : itp->second;
                    fmt::print("  ==> activation producer_op = {}\n", p);
                }
            }
        };

        // helper: schedule producer(op->prev[1]) for backtrace
        auto schedule_activation_producer = [&](int op_idx, const OpNode *op) {
            if ((int)op->prev.size() <= 1) return;
            Tensor *act = op->prev[1]->tensor();
            if (!act) return;

            auto itp = producer.find(act);
            int p = (itp == producer.end()) ? -1 : itp->second;
            if (p < 0) return;
            if (p == op_idx) return; // self-loop protection

            // prevent loops
            if (visited_backtrace.count(p)) return;
            visited_backtrace.insert(p);

            want_check_ops.insert(p);
        };

        // helper: compare out#0 for a given op_idx
        auto compare_out0 = [&](int op_idx, const OpNode *op) -> bool {
            if ((int)op->next.size() <= 0) return true;
            Tensor *out = op->next[0]->tensor();
            if (!out) return true;
            if (!out->m_data) return true; // skip view output compare
            if (out->m_dtype != DataType::FP32) return true;

            auto it = ggml_op_outs.find(op_out_key(op_idx, 0));
            if (it == ggml_op_outs.end()) return true; // no ref

            auto ocl_vec = tensor_to_f32_vec_any(out, cl_backend);
            auto &gg_vec = it->second;
            if (ocl_vec.size() != gg_vec.size()) {
                fmt::print("\n[MISMATCH] op#{} type={} out#0 size mismatch ocl={} ggml={}\n",
                        op_idx, op_type_to_string(op->op), ocl_vec.size(), gg_vec.size());
                print_tensor_meta2(out, "out", producer);
                return false;
            }

            size_t bad_i = 0;
            float diff = 0.f;
            if (!allclose_span(ocl_vec, gg_vec, ATOL, RTOL, &bad_i, &diff)) {
                // record + dump
                mismatch_chain.push_back({op_idx, 0, bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff});
                dump_mismatch_detail(op_idx, op, 0, out, gg_vec, ocl_vec, bad_i, diff);

                // schedule backtrace for activation producer
                schedule_activation_producer(op_idx, op);

                return false;
            }
            return true;
        };

        // --------- OpenCL pass hook ----------
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {

            // build producer mapping (outputs -> op_idx)
            for (int oi2 = 0; oi2 < (int)op->next.size(); ++oi2) {
                Tensor *o2 = op->next[oi2]->tensor();
                if (o2 && producer.find(o2) == producer.end()) {
                    producer[o2] = op_idx;
                }
            }

            // --- Normal per-op compare: out#0 only (can extend to all outs) ---
            // If we haven't seen mismatch yet, compare every op out#0.
            // After mismatch, we still compare current op out#0, but we ALSO prioritize checking want_check_ops.
            if (!first_mismatch_seen) {
                bool ok = compare_out0(op_idx, op);
                if (!ok) {
                    first_mismatch_seen = true;
                    // ensure current op is marked visited to avoid loops
                    visited_backtrace.insert(op_idx);
                }
            } else {
                // After first mismatch: only compare ops that are in want_check_ops,
                // plus keep comparing sequentially if you want more context.
                if (want_check_ops.count(op_idx)) {
                    want_check_ops.erase(op_idx);
                    compare_out0(op_idx, op);
                }
            }

            // If backtrace chain is finished (no more want_check_ops), exit after printing summary
            if (first_mismatch_seen && want_check_ops.empty()) {
                fmt::print("\n========== Divergence Backtrace Summary ==========\n");
                for (size_t i = 0; i < mismatch_chain.size(); ++i) {
                    auto &r = mismatch_chain[i];
                    fmt::print("  [{}] op#{} out#{} bad_i={} ggml={} ocl={} diff={}\n",
                            i, r.op_idx, r.out_idx, r.bad_i, r.ggml_v, r.ocl_v, r.diff);
                }
                fmt::print("=================================================\n");
                std::exit(1);
            }
        });


        auto ret_o = model_ocl->forward(tokens, pos, mask, true);

        // disable hook for later code
        set_op_after_exec_hook(nullptr);

        // still keep logits compare (optional)
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
    // 7) DECODE loop compare
    //    每步只 forward 最后一个 token
    //    mask 仍然用全长 tokens.size()
    // ----------------------------
    for (int step = 0; step < MAX_STEPS; ++step) {
        std::vector<Token> one_tok = { tokens.back() };
        std::vector<int> one_pos   = { (int)(tokens.size() - 1) };
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

        // 使用 GGML logits 做 argmax，保证 tokens 序列完全一致
        int next = argmax_token(lg);
        tokens.push_back(next);

        // stop conditions
        if (next == tokenizer.bos_token() || tokenizer.should_stop(next)) {
            POWERSERVE_LOG_INFO("Stop at step {} (token={})", step, next);
            break;
        }
    }

    POWERSERVE_LOG_INFO("All steps PASS: logits match within atol={} rtol={}", ATOL, RTOL);
    // ✅ 对齐仓库 generate() 的做法：用完就 reset，避免析构顺序/资源泄漏问题
    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
