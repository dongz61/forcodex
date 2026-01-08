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
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <span>

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
// CONFIG (hard-coded)
// ======================
static const char *MODEL_DIR   = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT      = "你好，请介绍你自己";

static int MAX_STEPS  = 64;
static int N_THREADS  = 8;
static size_t BATCH_SIZE = 1;

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

// ======================
// f32 tensor dump helpers
// ======================
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

static std::vector<float> tensor_to_f32_vec_any(const Tensor *t,
                                                powerserve::opencl::OpenCLBackend *cl_backend) {
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

// ======================
// NEW: raw bytes signature (for quant weights etc.)
// ======================
static inline uint64_t fnv1a64(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Flatten CPU tensor into contiguous bytes following tensor stride.
// (We cannot assume contiguous layout.)
static bool tensor_to_bytes_cpu(const Tensor *t, std::vector<uint8_t> &out_bytes) {
    out_bytes.clear();
    if (!t || !t->m_data) return false;
    if (!dynamic_cast<CPUBuffer*>(t->m_data.get())) return false;

    size_t elem_size = powerserve::get_type_size(t->m_dtype);
    if (elem_size == 0) return false;

    auto shape = t->m_shape;
    size_t n = t->n_elements();
    size_t nbytes = n * elem_size;
    if (nbytes == 0) return false;

    out_bytes.resize(nbytes);

    auto &cb = const_cast<Tensor*>(t)->get<CPUBuffer>();
    auto stride = cb.m_stride;

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    uint8_t *ptr = (uint8_t *)((char *)cb.m_data +
                                               i3 * stride[3] + i2 * stride[2] +
                                               i1 * stride[1] + i0 * stride[0]);
                    std::memcpy(out_bytes.data() + idx, ptr, elem_size);
                    idx += elem_size;
                }
            }
        }
    }
    POWERSERVE_ASSERT(idx == nbytes);
    return true;
}

// Compute signature for ANY tensor (CPU or OpenCL) without modifying backend.
// For OpenCL tensor: D2H copy into a CPU tensor (same dtype/shape), then flatten bytes by strides.
static bool tensor_bytes_sig_any(const Tensor *t,
                                powerserve::opencl::OpenCLBackend *cl_backend,
                                uint64_t *out_hash4096,
                                size_t *out_nbytes,
                                uint64_t *out_hash_full=nullptr) {
    if (!t || !t->m_data) return false;

    size_t elem_size = powerserve::get_type_size(t->m_dtype);
    if (elem_size == 0) return false;

    size_t nbytes = t->n_elements() * elem_size;
    if (nbytes == 0) return false;

    std::vector<uint8_t> bytes;

#if defined(POWERSERVE_WITH_OPENCL)
    if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
        POWERSERVE_ASSERT(cl_backend);

        Tensor tmp_cpu(t->m_dtype, t->m_shape);

        if (elem_size == 4) {
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<uint32_t>(t->m_shape);
        } else if (elem_size == 2) {
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<uint16_t>(t->m_shape);
        } else if (elem_size == 1) {
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<uint8_t>(t->m_shape);
        } else if (elem_size == 8) {
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<uint64_t>(t->m_shape);
        } else if (elem_size == 16) {
            struct u128 { uint64_t a,b; };
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<u128>(t->m_shape);
        } else {
            return false;
        }

        cl_backend->copy(&tmp_cpu, t);
        if (!tensor_to_bytes_cpu(&tmp_cpu, bytes)) return false;
    } else
#endif
    if (dynamic_cast<CPUBuffer*>(t->m_data.get())) {
        if (!tensor_to_bytes_cpu(t, bytes)) return false;
    } else {
        return false;
    }

    size_t hsz = std::min(bytes.size(), (size_t)4096);
    uint64_t h4096 = fnv1a64(bytes.data(), hsz);
    *out_hash4096 = h4096;
    *out_nbytes = bytes.size();
    if (out_hash_full) *out_hash_full = fnv1a64(bytes.data(), bytes.size());
    return true;
}

// ======================
// meta printing
// ======================
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

    fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] [NO_DATA/UNKNOWN]\n",
               tag, (int)t->m_dtype, shape[0], shape[1], shape[2], shape[3]);
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

static void dump_f32_sample(const std::vector<float> &v, const char *tag, int n = 8) {
    if (v.empty()) return;
    fmt::print("  {} sample: ", tag);
    for (int i = 0; i < n && i < (int)v.size(); ++i) {
        fmt::print("{:.6f} ", v[i]);
    }
    fmt::print("\n");
}

int main() {
    POWERSERVE_LOG_INFO("==== Qwen2 logits compare test (ggml vs opencl) [CMP2 + A-sig + B-compare] ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("MAX_STEPS={}, THREADS={}, BATCH_SIZE={}", MAX_STEPS, N_THREADS, BATCH_SIZE);

#if !defined(POWERSERVE_WITH_OPENCL)
    POWERSERVE_LOG_ERROR("POWERSERVE_WITH_OPENCL not enabled in compile.");
    return 1;
#endif

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
    platform->init_ggml_backend(model_ocl->m_config,  hparams);
    platform->init_opencl_backend(model_ocl->m_config, hparams);

    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        platform->ggml_backends[id_g]->setup_threadpool();
        platform->ggml_backends[id_o]->setup_threadpool();
    }

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

    // =========================
    // PREFILL compare
    // =========================
    {
        auto mask = CausalAttentionMask(tokens.size());

        // GGML caches:
        std::unordered_map<uint64_t, std::vector<float>> ggml_op_outs;

        // For B compare: cache MAT_MUL input[1] activation (fp32 vector)
        std::unordered_map<int, std::vector<float>> ggml_matmul_B_vec;

        // For A compare: cache MAT_MUL input[0] signature (raw bytes)
        struct SigRec { uint64_t h4096=0, hfull=0; size_t nbytes=0; bool ok=false; };
        std::unordered_map<int, SigRec> ggml_matmul_A_sig;

        // ---- GGML pass hook ----
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            // cache all FP32 outs
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out) continue;
                if (!out->m_data) continue;
                if (out->m_dtype != DataType::FP32) continue;
                ggml_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
            }

            // cache MAT_MUL inputs
            if (op->op == OpType::MAT_MUL) {
                // A signature
                if ((int)op->prev.size() > 0) {
                    Tensor *A = op->prev[0]->tensor();
                    SigRec r;
                    if (tensor_bytes_sig_any(A, /*cl_backend*/nullptr, &r.h4096, &r.nbytes, &r.hfull)) {
                        r.ok = true;
                        ggml_matmul_A_sig[op_idx] = r;
                    }
                }
                // B vector
                if ((int)op->prev.size() > 1) {
                    Tensor *B = op->prev[1]->tensor();
                    if (B && B->m_data && B->m_dtype == DataType::FP32) {
                        ggml_matmul_B_vec[op_idx] = tensor_to_f32_vec_cpu(B);
                    }
                }
            }
        });

        auto ret_g = model_ggml->forward(tokens, pos, mask, true);
        set_op_after_exec_hook(nullptr);

        // ---- OpenCL pass ----
        auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            platform->get_backend(model_ocl->m_config->model_id));
        POWERSERVE_ASSERT(cl_backend && "OpenCLBackend is null");

        std::unordered_map<const Tensor*, int> producer;

        struct MismatchRecord {
            int op_idx;
            int out_idx;
            size_t bad_i;
            float ggml_v;
            float ocl_v;
            float diff;
        };

        std::vector<MismatchRecord> mismatch_chain;
        std::unordered_set<int> want_check_ops;
        std::unordered_set<int> visited_backtrace;
        bool first_mismatch_seen = false;

        auto dump_mismatch_detail = [&](int op_idx, const OpNode *op, int out_idx,
                                        const Tensor *out,
                                        const std::vector<float> &gg_vec,
                                        const std::vector<float> &ocl_vec,
                                        size_t bad_i, float diff) {
            fmt::print("\n[MISMATCH] op#{} type={} out#{}\n", op_idx, op_type_to_string(op->op), out_idx);
            fmt::print("  bad_i={} ggml={} ocl={} diff={}\n", bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff);

            print_tensor_meta2(out, "out", producer);

            // print inputs meta
            for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                Tensor *in = op->prev[pi]->tensor();
                char name[64];
                std::snprintf(name, sizeof(name), "in[%d]", pi);
                print_tensor_meta2(in, name, producer);

                if (pi == 1 && in && in->m_dtype == DataType::FP32 && in->m_data) {
                    auto v_ocl = tensor_to_f32_vec_any(in, cl_backend);
                    dump_f32_sample(v_ocl, "in[1](activation)", 8);

                    // ---- B compare (STRICT) ----
                    auto itB = ggml_matmul_B_vec.find(op_idx);
                    if (itB != ggml_matmul_B_vec.end()) {
                        auto &v_gg = itB->second;
                        size_t badB = 0;
                        float diffB = 0.f;
                        bool okB = (v_ocl.size() == v_gg.size()) &&
                                   allclose_span(v_ocl, v_gg, /*atol*/0.f, /*rtol*/0.f, &badB, &diffB);
                        fmt::print("  [B-compare] size_ocl={} size_ggml={} match={}\n",
                                   v_ocl.size(), v_gg.size(), okB);
                        if (!okB && !v_ocl.empty() && badB < v_ocl.size()) {
                            fmt::print("    bad_i={} ggml={} ocl={} diff={}\n",
                                       badB, v_gg[badB], v_ocl[badB], std::fabs(v_gg[badB]-v_ocl[badB]));
                        }
                    } else {
                        fmt::print("  [B-compare] <missing ggml cached B>\n");
                    }
                }
            }

            // activation producer
            if ((int)op->prev.size() > 1) {
                Tensor *act = op->prev[1]->tensor();
                if (act) {
                    auto itp = producer.find(act);
                    int p = (itp == producer.end()) ? -1 : itp->second;
                    fmt::print("  ==> activation producer_op = {}\n", p);
                }
            }

            // ---- A signature compare (quant weight) ----
            if (op->op == OpType::MAT_MUL && (int)op->prev.size() > 0) {
                Tensor *A_ocl = op->prev[0]->tensor();
                uint64_t h4096_ocl = 0, hfull_ocl = 0;
                size_t nbytes_ocl = 0;
                bool ok_ocl = tensor_bytes_sig_any(A_ocl, cl_backend, &h4096_ocl, &nbytes_ocl, &hfull_ocl);

                fmt::print("  [A-sig][OCL ] ok={} nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                           ok_ocl, nbytes_ocl, h4096_ocl, hfull_ocl);

                auto itg = ggml_matmul_A_sig.find(op_idx);
                if (itg != ggml_matmul_A_sig.end() && itg->second.ok) {
                    auto &r = itg->second;
                    fmt::print("  [A-sig][GGML] nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                               r.nbytes, r.h4096, r.hfull);
                    bool match = ok_ocl && (r.nbytes == nbytes_ocl) && (r.h4096 == h4096_ocl) && (r.hfull == hfull_ocl);
                    fmt::print("  [A-sig] match={}\n", match);
                } else {
                    fmt::print("  [A-sig][GGML] <missing>\n");
                }
            }
        };

        auto schedule_activation_producer = [&](int op_idx, const OpNode *op) {
            if ((int)op->prev.size() <= 1) return;
            Tensor *act = op->prev[1]->tensor();
            if (!act) return;

            auto itp = producer.find(act);
            int p = (itp == producer.end()) ? -1 : itp->second;
            if (p < 0) return;
            if (p == op_idx) return;

            if (visited_backtrace.count(p)) return;
            visited_backtrace.insert(p);
            want_check_ops.insert(p);
        };

        auto compare_out0 = [&](int op_idx, const OpNode *op) -> bool {
            if ((int)op->next.size() <= 0) return true;
            Tensor *out = op->next[0]->tensor();
            if (!out) return true;
            if (!out->m_data) return true;
            if (out->m_dtype != DataType::FP32) return true;

            auto it = ggml_op_outs.find(op_out_key(op_idx, 0));
            if (it == ggml_op_outs.end()) return true;

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
                mismatch_chain.push_back({op_idx, 0, bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff});
                dump_mismatch_detail(op_idx, op, 0, out, gg_vec, ocl_vec, bad_i, diff);
                schedule_activation_producer(op_idx, op);
                return false;
            }
            return true;
        };

        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            // build producer mapping
            for (int oi2 = 0; oi2 < (int)op->next.size(); ++oi2) {
                Tensor *o2 = op->next[oi2]->tensor();
                if (o2 && producer.find(o2) == producer.end()) producer[o2] = op_idx;
            }

            if (!first_mismatch_seen) {
                bool ok = compare_out0(op_idx, op);
                if (!ok) {
                    first_mismatch_seen = true;
                    visited_backtrace.insert(op_idx);
                }
            } else {
                if (want_check_ops.count(op_idx)) {
                    want_check_ops.erase(op_idx);
                    compare_out0(op_idx, op);
                }
            }

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
        set_op_after_exec_hook(nullptr);

        // end-to-end logits compare (optional)
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

    // =========================
    // DECODE loop compare
    // =========================
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

        int next = argmax_token(lg);
        tokens.push_back(next);

        if (next == tokenizer.bos_token() || tokenizer.should_stop(next)) {
            POWERSERVE_LOG_INFO("Stop at step {} (token={})", step, next);
            break;
        }
    }

    POWERSERVE_LOG_INFO("All steps PASS: logits match within atol={} rtol={}", ATOL, RTOL);

    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
