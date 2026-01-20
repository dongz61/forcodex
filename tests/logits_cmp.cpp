// tests/logits_cmp.cpp
#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"
#include "graph/op_type.hpp"
#include "core/tensor.hpp"
#include "executor/executor.hpp"
#include "core/timer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>

using namespace powerserve;

// ======================
// CONFIG (adjust if needed)
// ======================
static const char *MODEL_DIR   = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT      = "你好，请介绍你自己";

static int N_THREADS     = 8;
static size_t BATCH_SIZE = 1;

static float ATOL = 1e-5f;
static float RTOL = 1e-5f;
static int TOPK   = 10;

// ======================
// Small helpers
// ======================
static const char *op_type_to_string(OpType t) {
    switch (t) {
    case OpType::GET_EMBEDDING:   return "GET_EMBEDDING";
    case OpType::ADD:             return "ADD";
    case OpType::MAT_MUL:         return "MAT_MUL";
    case OpType::RMS_NORM:        return "RMS_NORM";
    case OpType::SILU_HADAMARD:   return "SILU_HADAMARD";
    case OpType::ROPE:            return "ROPE";
    case OpType::SOFTMAX:         return "SOFTMAX";
    case OpType::COPY:            return "COPY";
    case OpType::ADD_CACHE:       return "ADD_CACHE";
    case OpType::PERMUTE:         return "PERMUTE";
    case OpType::CONT:            return "CONT";
    case OpType::VIEW:            return "VIEW";
    case OpType::SOFTMAX_EXT:     return "SOFTMAX_EXT";
    case OpType::GET_MASK:        return "GET_MASK";
    case OpType::TRANSPOSE:       return "TRANSPOSE";
    case OpType::PRINT:           return "PRINT";
#if defined(POWERSERVE_WITH_QNN)
    case OpType::QNN_FORWARD:     return "QNN_FORWARD";
    case OpType::QNN_FORWARD_VL:  return "QNN_FORWARD_VL";
#endif
    default:                      return "UNKNOWN";
    }
}

static inline bool allclose_span(std::span<const float> a, std::span<const float> b,
                                 float atol, float rtol,
                                 size_t *bad_i=nullptr, float *diff=nullptr) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        float ai = a[i];
        float bi = b[i];

        // 1) 先处理精确相等：包含 +inf==+inf、-inf==-inf、+0==-0，直接认为相等
        if (ai == bi) {
            continue;
        }

        // 2) 只要出现 NaN：直接判 mismatch（并避免 NaN 污染比较逻辑）
        if (std::isnan(ai) || std::isnan(bi)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = std::numeric_limits<float>::quiet_NaN();
            return false;
        }

        // 3) 只要出现 Inf（但又不相等）：一定 mismatch
        if (std::isinf(ai) || std::isinf(bi)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = std::numeric_limits<float>::infinity();
            return false;
        }

        // 4) 正常有限数走原来的 atol/rtol 比较
        float d   = std::fabs(ai - bi);
        float tol = atol + rtol * std::fabs(bi);
        if (!(d <= tol)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = d;
            return false;
        }
    }

    return true;
}


static void dump_topk(std::span<const float> logits, int k, const char *tag) {
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, (int)idx.size()), idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    fmt::print("[{} top{}] ", tag, k);
    for (int i = 0; i < k && i < (int)idx.size(); ++i) {
        int id = idx[i];
        fmt::print("({} {:.6f}) ", id, logits[id]);
    }
    fmt::print("\n");
}

// ======================
// Tensor -> FP32 vector
// ======================
static std::vector<float> tensor_to_f32_vec_cpu(const Tensor *t) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32) return out;
    if (!t->m_data) return out;

    auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get());
    if (!cb) return {};

    auto shape  = t->m_shape;
    auto stride = cb->m_stride;

    size_t n = t->n_elements();
    out.resize(n);

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

#if defined(POWERSERVE_WITH_OPENCL)
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
#endif

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
// Raw bytes signature (for A/B bytes exact compare)
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

static bool tensor_to_bytes_cpu(const Tensor *t, std::vector<uint8_t> &out_bytes) {
    out_bytes.clear();
    if (!t || !t->m_data) return false;
    auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get());
    if (!cb) return false;

    size_t elem_size = powerserve::get_type_size(t->m_dtype);
    if (elem_size == 0) return false;

    size_t nbytes = t->n_elements() * elem_size;
    if (nbytes == 0) return false;
    out_bytes.resize(nbytes);

    auto shape  = t->m_shape;
    auto stride = cb->m_stride;

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    uint8_t *ptr = (uint8_t *)((char *)cb->m_data +
                                               i3 * stride[3] + i2 * stride[2] +
                                               i1 * stride[1] + i0 * stride[0]);
                    std::memcpy(out_bytes.data() + idx, ptr, elem_size);
                    idx += elem_size;
                }
            }
        }
    }
    return idx == nbytes;
}

static bool tensor_bytes_sig_any(const Tensor *t,
                                powerserve::opencl::OpenCLBackend *cl_backend,
                                uint64_t *out_hash4096,
                                size_t *out_nbytes,
                                uint64_t *out_hash_full=nullptr) {
    if (!t || !t->m_data) return false;

    size_t elem_size = powerserve::get_type_size(t->m_dtype);
    if (elem_size == 0) return false;

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
    *out_nbytes   = bytes.size();
    if (out_hash_full) *out_hash_full = fnv1a64(bytes.data(), bytes.size());
    return true;
}

// ======================
// Meta printing
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

static void print_tensor_meta2(const Tensor *t, const char *tag,
                               const std::unordered_map<const Tensor*, int> &producer) {
    if (!t) {
        fmt::print("  {}: <null>\n", tag);
        return;
    }

    print_tensor_meta(t, tag);

    auto it = producer.find(t);
    int p = (it == producer.end()) ? -1 : it->second;

    fmt::print("    ptr={} producer_op={} is_view={} has_data={} {}\n",
               (const void*)t, p,
               (t->m_data == nullptr),
               (t->m_data != nullptr),
               is_opencl_tensor(t) ? "[CL]" : "[CPU]");
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

// ======================
// ULP helpers (strict compare)
// ======================
static inline uint32_t f32_to_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline int32_t f32_to_ordered_int(float x) {
    uint32_t u = f32_to_bits(x);
    if (u & 0x80000000u) u = 0xFFFFFFFFu - u;
    else u = u + 0x80000000u;
    return (int32_t)u;
}

static inline int32_t ulp_dist(float a, float b) {
    int32_t ia = f32_to_ordered_int(a);
    int32_t ib = f32_to_ordered_int(b);
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

static inline bool allclose_ulp_span(std::span<const float> a, std::span<const float> b,
                                     int32_t max_ulp,
                                     size_t *bad_i=nullptr,
                                     float *bad_abs=nullptr,
                                     int32_t *bad_ulp=nullptr) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t u = ulp_dist(a[i], b[i]);
        if (u > max_ulp) {
            if (bad_i) *bad_i = i;

            if (bad_abs) {
                float ai = a[i], bi = b[i];
                *bad_abs = (std::isfinite(ai) && std::isfinite(bi))
                        ? std::fabs(ai - bi)
                        : std::numeric_limits<float>::infinity();
            }

            if (bad_ulp) *bad_ulp = u;
            return false;
        }
    }
    return true;
}

// ======================
// Diff stats (for B)
// ======================
static void dump_vec_diff_stats(const std::vector<float>& gg, const std::vector<float>& oc) {
    size_t n = std::min(gg.size(), oc.size());
    size_t first_bad = (size_t)-1;
    size_t max_i = 0;
    float max_abs = 0.f;
    int32_t max_ulp = 0;
    size_t cnt_nonzero = 0;
    size_t cnt_gt_1e7 = 0; // >1e-7
    size_t cnt_gt_1e6 = 0; // >1e-6
    size_t cnt_gt_1e5 = 0; // >1e-5

    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(gg[i] - oc[i]);
        if (d != 0.f) {
            if (first_bad == (size_t)-1) first_bad = i;
            cnt_nonzero++;
            if (d > 1e-7f) cnt_gt_1e7++;
            if (d > 1e-6f) cnt_gt_1e6++;
            if (d > 1e-5f) cnt_gt_1e5++;
        }
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        int32_t u = ulp_dist(gg[i], oc[i]);
        if (u > max_ulp) max_ulp = u;
    }

    fmt::print("  [B-diff] n={} first_bad={} cnt_nonzero={} >1e-7:{} >1e-6:{} >1e-5:{}\n",
               n, first_bad == (size_t)-1 ? 0 : first_bad, cnt_nonzero,
               cnt_gt_1e7, cnt_gt_1e6, cnt_gt_1e5);
    fmt::print("  [B-diff] max_abs={} at i={} (ggml={} ocl={}) max_ulp={}\n",
               max_abs, max_i, gg[max_i], oc[max_i], max_ulp);

    const int K = 8;
    std::array<size_t, K> top_i{};
    std::array<float, K> top_d{};
    top_d.fill(-1.f);
    top_i.fill(0);

    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(gg[i] - oc[i]);
        for (int k = 0; k < K; k++) {
            if (d > top_d[k]) {
                for (int t = K - 1; t > k; t--) {
                    top_d[t] = top_d[t - 1];
                    top_i[t] = top_i[t - 1];
                }
                top_d[k] = d;
                top_i[k] = i;
                break;
            }
        }
    }
    fmt::print("  [B-diff] top{}:\n", K);
    for (int k = 0; k < K; k++) {
        size_t i = top_i[k];
        fmt::print("    i={} diff={} ggml={} ocl={} ulp={}\n",
                   i, top_d[k], gg[i], oc[i], ulp_dist(gg[i], oc[i]));
    }
}

int main() {
    POWERSERVE_LOG_INFO("==== Qwen2 logits compare test (ggml vs opencl) [offline backtrace ULP=0] ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("THREADS={}, BATCH_SIZE={}", N_THREADS, BATCH_SIZE);

#if !defined(POWERSERVE_WITH_OPENCL)
    POWERSERVE_LOG_ERROR("POWERSERVE_WITH_OPENCL not enabled in compile.");
    return 1;
#endif

    HyperParams hparams;
    hparams.n_threads  = N_THREADS;
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

        // GGML cached outs (FP32 only)
        std::unordered_map<uint64_t, std::vector<float>> ggml_op_outs;

        // Cache MAT_MUL input[1] activation in GGML (fp32 vector) + signatures
        struct SigRec { uint64_t h4096=0, hfull=0; size_t nbytes=0; bool ok=false; };
        std::unordered_map<int, std::vector<float>> ggml_matmul_B_vec;
        std::unordered_map<int, SigRec> ggml_matmul_A_sig;
        std::unordered_map<int, SigRec> ggml_matmul_B_sig;
        std::unordered_map<int, std::vector<float>> ggml_silu_in0_vec; // hb
        std::unordered_map<int, std::vector<float>> ggml_silu_in1_vec; // hb2


        // ---- GGML pass hook ----
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            // cache all FP32 outs (useful for backtrace)
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out || !out->m_data) continue;
                if (out->m_dtype != DataType::FP32) continue;
                ggml_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
            }

            // cache MAT_MUL inputs: A/B sig + B vec
            if (op->op == OpType::MAT_MUL) {
                if ((int)op->prev.size() > 0) {
                    Tensor *A = op->prev[0]->tensor();
                    SigRec r;
                    if (tensor_bytes_sig_any(A, /*cl_backend*/nullptr, &r.h4096, &r.nbytes, &r.hfull)) {
                        r.ok = true;
                        ggml_matmul_A_sig[op_idx] = r;
                    }
                }
                if ((int)op->prev.size() > 1) {
                    Tensor *B = op->prev[1]->tensor();
                    SigRec r;
                    if (tensor_bytes_sig_any(B, /*cl_backend*/nullptr, &r.h4096, &r.nbytes, &r.hfull)) {
                        r.ok = true;
                        ggml_matmul_B_sig[op_idx] = r;
                    }
                    if (B && B->m_data && B->m_dtype == DataType::FP32) {
                        ggml_matmul_B_vec[op_idx] = tensor_to_f32_vec_cpu(B);
                    }
                }
            }

            // cache SILU_HADAMARD inputs: hb/hb2 vec
            if (op->op == OpType::SILU_HADAMARD) {
                if ((int)op->prev.size() > 0) {
                    Tensor *hb = op->prev[0]->tensor();
                    if (hb && hb->m_data && hb->m_dtype == DataType::FP32) {
                        ggml_silu_in0_vec[op_idx] = tensor_to_f32_vec_cpu(hb);
                    }
                }
                if ((int)op->prev.size() > 1) {
                    Tensor *hb2 = op->prev[1]->tensor();
                    if (hb2 && hb2->m_data && hb2->m_dtype == DataType::FP32) {
                        ggml_silu_in1_vec[op_idx] = tensor_to_f32_vec_cpu(hb2);
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

        auto &id_g = model_ggml->m_config->model_id;
        auto *gg_backend = platform->ggml_backends[id_g].get();
        POWERSERVE_ASSERT(gg_backend && "GGML backend null");

        // Cache OpenCL FP32 outputs for offline backtrace
        std::unordered_map<uint64_t, std::vector<float>> ocl_op_outs;
        std::unordered_map<int, const OpNode*> ocl_op_nodes;

        // producer map: Tensor* -> op_idx
        std::unordered_map<const Tensor*, int> producer;

        // Pick a backtrace input tensor: choose first prev tensor whose producer_op >= 0
        auto pick_backtrace_input =
            [&](const OpNode* op, int* out_prod_op_idx, int* out_prev_i) -> const Tensor* {
                if (out_prod_op_idx) *out_prod_op_idx = -1;
                if (out_prev_i) *out_prev_i = -1;
                if (!op) return nullptr;

                for (int i = 0; i < (int)op->prev.size(); ++i) {
                    const Tensor* t = op->prev[i] ? op->prev[i]->tensor() : nullptr;
                    if (!t) continue;

                    auto it = producer.find(t);
                    int p = (it == producer.end()) ? -1 : it->second;
                    if (p >= 0) {
                        if (out_prod_op_idx) *out_prod_op_idx = p;
                        if (out_prev_i) *out_prev_i = i;
                        return t;
                    }
                }
                return nullptr;
            };

        // Detailed mismatch dumper
        auto dump_mismatch_detail =
            [&](int op_idx, const OpNode *op, int out_idx,
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

                // print chosen trace input (for understanding backtrace path)
                {
                    int p = -1, prev_i = -1;
                    pick_backtrace_input(op, &p, &prev_i);
                    fmt::print("  ==> trace_input = in[{}], producer_op = {}\n", prev_i, p);
                }

                // B/A diagnostics for MAT_MUL 
                if (op->op == OpType::MAT_MUL && (int)op->prev.size() > 1) {
                    Tensor *B_in = op->prev[1]->tensor();
                    if (B_in && B_in->m_dtype == DataType::FP32 && B_in->m_data) {
                        // B signature compare (bytes)
                        uint64_t h4096_ocl=0, hfull_ocl=0;
                        size_t nbytes_ocl=0;
                        bool ok_ocl_sig = tensor_bytes_sig_any(B_in, cl_backend, &h4096_ocl, &nbytes_ocl, &hfull_ocl);

                        auto itBsig = ggml_matmul_B_sig.find(op_idx);
                        bool ok_gg_sig = (itBsig != ggml_matmul_B_sig.end()) && itBsig->second.ok;

                        fmt::print("  [B-sig][OCL ] ok={} nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                                   ok_ocl_sig, nbytes_ocl, h4096_ocl, hfull_ocl);
                        if (ok_gg_sig) {
                            auto &r = itBsig->second;
                            fmt::print("  [B-sig][GGML] ok={} nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                                       r.ok, r.nbytes, r.h4096, r.hfull);
                            fmt::print("  [B-sig] match={}\n",
                                       (ok_ocl_sig && r.nbytes==nbytes_ocl && r.h4096==h4096_ocl && r.hfull==hfull_ocl));
                        } else {
                            fmt::print("  [B-sig][GGML] <missing>\n");
                        }

                        // B value compare (vector)
                        auto v_ocl = tensor_to_f32_vec_any(B_in, cl_backend);
                        dump_f32_sample(v_ocl, "in[1](activation)", 8);

                        auto itB = ggml_matmul_B_vec.find(op_idx);
                        if (itB != ggml_matmul_B_vec.end()) {
                            auto &v_gg = itB->second;
                            size_t badB = 0;
                            float diffB = 0.f;
                            bool okB = (v_ocl.size() == v_gg.size()) &&
                                       allclose_span(v_ocl, v_gg, /*atol*/0.f, /*rtol*/0.f, &badB, &diffB);

                            fmt::print("  [B-compare] size_ocl={} size_ggml={} match={}\n",
                                       v_ocl.size(), v_gg.size(), okB);
                            dump_vec_diff_stats(v_gg, v_ocl);

                            // column-specific stats for the output's bad column (out is [K,N])
                            if (out && out->m_shape[0] > 0 && out->m_shape[1] > 0) {
                                size_t K = out->m_shape[0];
                                size_t N = out->m_shape[1];
                                size_t col = bad_i / K;
                                if (col < N && v_gg.size() == v_ocl.size() && v_gg.size() == K * N) {
                                    float max_abs_col = 0.f;
                                    size_t max_k = 0;
                                    size_t cnt_nz_col = 0;
                                    for (size_t k = 0; k < K; k++) {
                                        size_t idx = k + col * K;
                                        float dcol = std::fabs(v_gg[idx] - v_ocl[idx]);
                                        if (dcol != 0.f) cnt_nz_col++;
                                        if (dcol > max_abs_col) { max_abs_col = dcol; max_k = k; }
                                    }
                                    fmt::print("  [B-col] col={} cnt_nonzero={} max_abs={} at k={} (ggml={} ocl={})\n",
                                               col, cnt_nz_col, max_abs_col, max_k,
                                               v_gg[max_k + col * K], v_ocl[max_k + col * K]);
                                }
                            }

                            if (!okB && !v_ocl.empty() && badB < v_ocl.size()) {
                                fmt::print("    bad_i={} ggml={} ocl={} diff={}\n",
                                           badB, v_gg[badB], v_ocl[badB], std::fabs(v_gg[badB]-v_ocl[badB]));
                            }
                        } else {
                            fmt::print("  [B-compare] <missing ggml cached B>\n");
                        }
                    }

                    // A signature compare
                    Tensor *A_in = op->prev[0]->tensor();
                    uint64_t h4096_A=0, hfull_A=0;
                    size_t nbytes_A=0;
                    bool okA = tensor_bytes_sig_any(A_in, cl_backend, &h4096_A, &nbytes_A, &hfull_A);
                    fmt::print("  [A-sig][OCL ] ok={} nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                               okA, nbytes_A, h4096_A, hfull_A);

                    auto itAsig = ggml_matmul_A_sig.find(op_idx);
                    if (itAsig != ggml_matmul_A_sig.end() && itAsig->second.ok) {
                        auto &r = itAsig->second;
                        fmt::print("  [A-sig][GGML] nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                                   r.nbytes, r.h4096, r.hfull);
                        bool matchA = okA && (r.nbytes == nbytes_A) && (r.h4096 == h4096_A) && (r.hfull == hfull_A);
                        fmt::print("  [A-sig] match={}\n", matchA);
                    } else {
                        fmt::print("  [A-sig][GGML] <missing>\n");
                    }

                    // EXP1: recompute GGML matmul using B read back from OpenCL
                    // Tensor *A_in2 = op->prev[0]->tensor();
                    // Tensor *B_in2 = op->prev[1]->tensor();
                    // if (out && out->m_dtype == DataType::FP32 && A_in2 && B_in2 &&
                    //     A_in2->m_data && B_in2->m_data && B_in2->m_dtype == DataType::FP32) {

                    //     Tensor B_cpu(DataType::FP32, B_in2->m_shape);
                    //     B_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(B_in2->m_shape);
                    //     cl_backend->copy(&B_cpu, B_in2);

                    //     Tensor C_cpu(DataType::FP32, out->m_shape);
                    //     C_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

                    //     gg_backend->matmul(&C_cpu, A_in2, &B_cpu);

                    //     auto gg_re = tensor_to_f32_vec_cpu(&C_cpu);
                    //     if (!gg_re.empty() && gg_re.size() == ocl_vec.size()) {
                    //         size_t bad2 = 0;
                    //         float diff2 = 0.f;
                    //         bool ok2 = allclose_span(gg_re, ocl_vec, /*atol*/0.f, /*rtol*/0.f, &bad2, &diff2);
                    //         fmt::print("  [MATMUL-RE] ggml(A, B_ocl) vs ocl_out match={} bad_i={} ggml_re={} ocl={} diff={}\n",
                    //                    ok2, bad2, gg_re[bad2], ocl_vec[bad2], diff2);
                    //     } else {
                    //         fmt::print("  [MATMUL-RE] skip (size gg_re={} ocl_vec={})\n", gg_re.size(), ocl_vec.size());
                    //     }
                    // }
                }
                
                // SILU_HADAMARD diagnostics
                if (op->op == OpType::SILU_HADAMARD && (int)op->prev.size() >= 2) {
                    Tensor *hb  = op->prev[0]->tensor();
                    Tensor *hb2 = op->prev[1]->tensor();

                    // 读 OpenCL 侧输入（用你现成的 tensor_to_f32_vec_any，它会对 CL tensor 做 D2H copy）:contentReference[oaicite:5]{index=5}
                    auto hb_ocl  = tensor_to_f32_vec_any(hb,  cl_backend);
                    auto hb2_ocl = tensor_to_f32_vec_any(hb2, cl_backend);

                    // 找 GGML 侧输入缓存
                    auto it_hb  = ggml_silu_in0_vec.find(op_idx);
                    auto it_hb2 = ggml_silu_in1_vec.find(op_idx);

                    if (it_hb != ggml_silu_in0_vec.end() && it_hb2 != ggml_silu_in1_vec.end() &&
                        hb_ocl.size() == it_hb->second.size() && hb2_ocl.size() == it_hb2->second.size()) {

                        fmt::print("  [SILU-IN0] compare hb (ggml vs ocl)\n");
                        dump_vec_diff_stats(it_hb->second, hb_ocl);

                        fmt::print("  [SILU-IN1] compare hb2 (ggml vs ocl)\n");
                        dump_vec_diff_stats(it_hb2->second, hb2_ocl);
                    } else {
                        fmt::print("  [SILU-IN] ggml input cache missing or size mismatch: hb_ocl={} hb2_ocl={}\n",
                                hb_ocl.size(), hb2_ocl.size());
                    }

                    // ---- 小测试：用“同一份 OCL 输入”在 CPU 上按 GGML 公式重算，然后对比 OCL out ----
                    // out 的 OCL vec 你已经传进来了：ocl_vec（是 op out 的 f32 vector）:contentReference[oaicite:6]{index=6}
                    if (!hb_ocl.empty() && hb_ocl.size() == hb2_ocl.size() && hb_ocl.size() == ocl_vec.size()) {
                        std::vector<float> cpu_ref(ocl_vec.size());
                        for (size_t i = 0; i < cpu_ref.size(); ++i) {
                            float x = hb_ocl[i];
                            float s = x * (1.0f / (1.0f + expf(-x)));
                            cpu_ref[i] = s * hb2_ocl[i];
                        }

                        fmt::print("  [SILU-CPU-REF] cpu_ref(ocl_in) vs ocl_out\n");
                        dump_vec_diff_stats(cpu_ref, ocl_vec);

                        // 也可以顺便看 cpu_ref(ocl_in) vs ggml_out（gg_vec 是 ggml out0 vec）:contentReference[oaicite:7]{index=7}
                        fmt::print("  [SILU-CPU-REF] cpu_ref(ocl_in) vs ggml_out\n");
                        dump_vec_diff_stats(cpu_ref, gg_vec);
                    }
                }

            };

        // Offline strict backtrace along chosen producer chain using cached outputs (ULP=0)
        auto backtrace_chain = [&](int start_op_idx) {
            fmt::print("\n========== Strict Divergence Backtrace (ULP=0) ==========\n");

            int cur = start_op_idx;
            std::unordered_set<int> seen;

            while (true) {
                if (seen.count(cur)) break;
                seen.insert(cur);

                auto itop = ocl_op_nodes.find(cur);
                const OpNode *cur_op = (itop == ocl_op_nodes.end()) ? nullptr : itop->second;
                if (!cur_op) {
                    fmt::print("  [BT] op#{}: missing op node, stop.\n", cur);
                    break;
                }

                int p = -1, prev_i = -1;
                const Tensor* t_trace = pick_backtrace_input(cur_op, &p, &prev_i);
                if (!t_trace || p < 0) {
                    fmt::print("  [BT] op#{} type={} no traceable input (all producers <0), stop.\n",
                               cur, op_type_to_string(cur_op->op));
                    break;
                }

                auto itg = ggml_op_outs.find(op_out_key(p, 0));
                auto ito = ocl_op_outs.find(op_out_key(p, 0));
                auto itpnode = ocl_op_nodes.find(p);
                const OpNode *pnode = (itpnode == ocl_op_nodes.end()) ? nullptr : itpnode->second;

                if (itg == ggml_op_outs.end() || ito == ocl_op_outs.end() || !pnode) {
                    fmt::print("  [BT] producer op#{} missing cached out0/opnode (ggml:{} ocl:{} node:{}), stop.\n",
                               p,
                               itg != ggml_op_outs.end(),
                               ito != ocl_op_outs.end(),
                               pnode != nullptr);
                    break;
                }

                size_t bad_i = 0;
                float bad_abs = 0.f;
                int32_t bad_ulp = 0;

                bool ok = allclose_ulp_span(ito->second, itg->second, /*max_ulp*/0, &bad_i, &bad_abs, &bad_ulp);
                if (ok) {
                    fmt::print("  [BT] producer op#{} type={} via prev[{}] out0 matches exactly (ULP=0). stop.\n",
                               p, op_type_to_string(pnode->op), prev_i);
                    break;
                }

                fmt::print("  [BT] MISMATCH at producer op#{} type={} via prev[{}] bad_i={} ocl={} ggml={} abs={} ulp={}\n",
                           p, op_type_to_string(pnode->op), prev_i,
                           bad_i, ito->second[bad_i], itg->second[bad_i], bad_abs, bad_ulp);

                Tensor *p_out = nullptr;
                if ((int)pnode->next.size() > 0) p_out = pnode->next[0]->tensor();
                if (p_out) {
                    dump_mismatch_detail(p, pnode, 0, p_out, itg->second, ito->second, bad_i, bad_abs);
                }

                cur = p;
            }

            fmt::print("========================================================\n");
        };

        // compare current op out0 against GGML cached out0 using ATOL/RTOL (fast trigger)
        auto compare_out0 = [&](int op_idx, const OpNode *op) -> bool {
            if ((int)op->next.size() <= 0) return true;
            Tensor *out = op->next[0]->tensor();
            if (!out || !out->m_data) return true;
            if (out->m_dtype != DataType::FP32) return true;

            auto itg = ggml_op_outs.find(op_out_key(op_idx, 0));
            if (itg == ggml_op_outs.end()) return true;

            auto ocl_vec = tensor_to_f32_vec_any(out, cl_backend);
            auto &gg_vec = itg->second;

            if (ocl_vec.size() != gg_vec.size()) {
                fmt::print("\n[MISMATCH] op#{} type={} out#0 size mismatch ocl={} ggml={}\n",
                           op_idx, op_type_to_string(op->op), ocl_vec.size(), gg_vec.size());
                print_tensor_meta2(out, "out", producer);
                backtrace_chain(op_idx);
                std::exit(1);
            }

            size_t bad_i = 0;
            float diff = 0.f;
            if (!allclose_span(ocl_vec, gg_vec, ATOL, RTOL, &bad_i, &diff)) {
                dump_mismatch_detail(op_idx, op, 0, out, gg_vec, ocl_vec, bad_i, diff);
                backtrace_chain(op_idx);
                std::exit(1);
            }
            return true;
        };

        // ---- OpenCL pass hook ----
        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            ocl_op_nodes[op_idx] = op;

            // build producer mapping
            for (int oi2 = 0; oi2 < (int)op->next.size(); ++oi2) {
                Tensor *o2 = op->next[oi2]->tensor();
                if (o2 && producer.find(o2) == producer.end()) producer[o2] = op_idx;
            }

            // cache all FP32 outputs for offline backtrace
            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out || !out->m_data) continue;
                if (out->m_dtype != DataType::FP32) continue;
                ocl_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_any(out, cl_backend);
            }

            // live compare (fast trigger)
            compare_out0(op_idx, op);
        });

        auto ret_o = model_ocl->forward(tokens, pos, mask, true);
        set_op_after_exec_hook(nullptr);

        // end-to-end logits compare
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

    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
