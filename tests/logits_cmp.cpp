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

static float ATOL = 1e-6f;
static float RTOL = 1e-6f;
static float ATOL_MATMUL_B_FROM_SILU = 1e-4f;
static float RTOL_MATMUL_B_FROM_SILU = 1e-4f;
static int TOPK   = 10;
static int DECODE_STEPS = 16; 


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

static inline bool matmul_B_is_from_silu(
    const OpNode *op,
    const std::unordered_map<const Tensor*, int> &producer,
    const std::unordered_map<int, const OpNode*> &ocl_op_nodes) {

    if (!op || op->op != OpType::MAT_MUL) return false;
    if ((int)op->prev.size() <= 1) return false;

    const Tensor *t = op->prev[1] ? op->prev[1]->tensor() : nullptr; // B
    if (!t) return false;

    auto it = producer.find(t);
    if (it == producer.end()) return false;

    int p = it->second;

    // Follow a short chain: B may be wrapped by VIEW/CONT/COPY/TRANSPOSE/PERMUTE
    // before feeding MATMUL. Bound the walk to avoid cycles.
    for (int depth = 0; depth < 8; ++depth) {
        auto itn = ocl_op_nodes.find(p);
        if (itn == ocl_op_nodes.end() || !itn->second) return false;

        const OpNode *node = itn->second;
        if (node->op == OpType::SILU_HADAMARD) return true;

        bool pass_through =
            (node->op == OpType::VIEW) ||
            (node->op == OpType::CONT) ||
            (node->op == OpType::COPY) ||
            (node->op == OpType::TRANSPOSE) ||
            (node->op == OpType::PERMUTE);

        if (!pass_through) return false;
        if ((int)node->prev.size() <= 0) return false;

        const Tensor *up = node->prev[0] ? node->prev[0]->tensor() : nullptr;
        if (!up) return false;

        auto itp = producer.find(up);
        if (itp == producer.end()) return false;

        p = itp->second;
    }
    return false;
}

static int argmax_span(std::span<const float> v) {
    if (v.empty()) return 0;
    int best_i = 0;
    float best_v = v[0];
    for (int i = 1; i < (int)v.size(); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best_i = i;
        }
    }
    return best_i;
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
            BaseBuffer &base = const_cast<Tensor*>(t)->get<BaseBuffer>();
        auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&base);
        if (clb) {
            POWERSERVE_ASSERT(cl_backend && "tensor_bytes_sig_any: OpenCL tensor requires cl_backend");
        }

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

    fmt::print("    ptr={} producer_op={} no_data={} has_data={} {}\n",
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
    // Run one forward with per-op compare (GGML vs OpenCL)
    // =========================
    auto run_forward_per_op_cmp =
        [&](const std::vector<Token> &in_tokens,
            const std::vector<int>   &in_pos,
            const CausalAttentionMask &in_mask,
            bool lm_head,
            const char *phase_tag) -> std::pair<LogitsVector, LogitsVector> {

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

            auto ret_g = model_ggml->forward(in_tokens, in_pos, in_mask, lm_head);
            set_op_after_exec_hook(nullptr);

            // ---- OpenCL pass ----
            auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
                platform->get_backend(model_ocl->m_config->model_id));
            POWERSERVE_ASSERT(cl_backend && "OpenCLBackend is null");

            auto &id_g = model_ggml->m_config->model_id;
            auto *gg_backend = platform->ggml_backends[id_g].get();
            POWERSERVE_ASSERT(gg_backend && "GGML backend null");
            POWERSERVE_UNUSED(gg_backend);

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

                    fmt::print("\n[MISMATCH][{}] op#{} type={} out#{}\n", phase_tag, op_idx, op_type_to_string(op->op), out_idx);
                    fmt::print("  bad_i={} ggml={} ocl={} diff={}\n", bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff);
                    print_tensor_meta2(out, "out", producer);

                    // print inputs
                    for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                        Tensor *in = op->prev[pi]->tensor();
                        char name[64];
                        std::snprintf(name, sizeof(name), "in[%d]", pi);
                        print_tensor_meta2(in, name, producer);
                    }

                    // print chosen trace input
                    {
                        int p = -1, prev_i = -1;
                        pick_backtrace_input(op, &p, &prev_i);
                        fmt::print("  ==> trace_input = in[{}], producer_op = {}\n", prev_i, p);
                    }

                    // (保留你原来的 MATMUL/SILU 诊断逻辑)
                    if (op->op == OpType::MAT_MUL && (int)op->prev.size() > 1) {
                        Tensor *B_in = op->prev[1]->tensor();
                        if (B_in && B_in->m_dtype == DataType::FP32 && B_in->m_data) {
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

                            auto v_ocl = tensor_to_f32_vec_any(B_in, cl_backend);
                            dump_f32_sample(v_ocl, "in[1](activation)", 8);

                            auto itB = ggml_matmul_B_vec.find(op_idx);
                            if (itB != ggml_matmul_B_vec.end()) {
                                auto &v_gg = itB->second;
                                fmt::print("  [B-compare] size_ocl={} size_ggml={}\n", v_ocl.size(), v_gg.size());
                                if (v_gg.size() == v_ocl.size()) dump_vec_diff_stats(v_gg, v_ocl);
                            } else {
                                fmt::print("  [B-compare] <missing ggml cached B>\n");
                            }
                        }
                    }

                    if (op->op == OpType::SILU_HADAMARD && (int)op->prev.size() >= 2) {
                        Tensor *hb  = op->prev[0]->tensor();
                        Tensor *hb2 = op->prev[1]->tensor();
                        auto hb_ocl  = tensor_to_f32_vec_any(hb,  cl_backend);
                        auto hb2_ocl = tensor_to_f32_vec_any(hb2, cl_backend);

                        auto it_hb  = ggml_silu_in0_vec.find(op_idx);
                        auto it_hb2 = ggml_silu_in1_vec.find(op_idx);

                        if (it_hb != ggml_silu_in0_vec.end() && it_hb2 != ggml_silu_in1_vec.end() &&
                            hb_ocl.size() == it_hb->second.size() && hb2_ocl.size() == it_hb2->second.size()) {

                            fmt::print("  [SILU-IN0] compare hb (ggml vs ocl)\n");
                            dump_vec_diff_stats(it_hb->second, hb_ocl);

                            fmt::print("  [SILU-IN1] compare hb2 (ggml vs ocl)\n");
                            dump_vec_diff_stats(it_hb2->second, hb2_ocl);
                        }
                    }

                    if (op->op == OpType::VIEW) {
                        Tensor *out = op->next[0]->tensor();
                        Tensor *in0 = op->prev.size() > 0 ? op->prev[0]->tensor() : nullptr;

                        auto dump_cl_buf = [&](const char *tag, Tensor *t) {
                            if (!t || !t->m_data) {
                                fprintf(stderr, "  [VIEW_DBG] %s: null tensor/data\n", tag);
                                return;
                            }
                            auto *cl = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&t->get<BaseBuffer>());
                            if (!cl) {
                                fprintf(stderr, "  [VIEW_DBG] %s: not OpenCLBuffer\n", tag);
                                return;
                            }
                            fprintf(stderr,
                                "  [VIEW_DBG] %s: dev=%p base_off=%zu size=%zu dtype=%d shape=[%d,%d,%d,%d] strideB=[%zu,%zu,%zu,%zu]\n",
                                tag,
                                (void*)cl->get_device_buffer(),
                                (size_t)cl->get_base_offset(),
                                (size_t)cl->m_size,   // 用字段，避免 get_size() 不存在导致编译失败
                                (int)t->m_dtype,
                                (int)t->m_shape[0], (int)t->m_shape[1], (int)t->m_shape[2], (int)t->m_shape[3],
                                (size_t)cl->m_stride[0], (size_t)cl->m_stride[1], (size_t)cl->m_stride[2], (size_t)cl->m_stride[3]
                            );

                        };

                        fprintf(stderr, "  [VIEW_DBG] ---- VIEW mismatch extra dump ----\n");
                        dump_cl_buf("in0(parent)", in0);
                        dump_cl_buf("out(view)", out);
                        fprintf(stderr, "  [VIEW_DBG] --------------------------------\n");
                    }
                    // ---- Extra: VIEW parent leaf sanity (only on mismatch) ----
                    if (op->op == OpType::VIEW && (int)op->prev.size() > 0) {
                        Tensor *parent = op->prev[0]->tensor();
                        if (parent) {
                           uint64_t h4096_o=0, hfull_o=0;
                            size_t nb_o=0;
                            bool ok_o = tensor_bytes_sig_any(parent, cl_backend, &h4096_o, &nb_o, &hfull_o);
                            fmt::print("  [VIEW-parent-sig][OCL ] ok={} nbytes={} hash4096=0x{:016x} hash_full=0x{:016x}\n",
                                    ok_o, nb_o, h4096_o, hfull_o);

                            auto v_o = tensor_to_f32_vec_any(parent, cl_backend);
                            dump_f32_sample(v_o, "VIEW-parent ocl", 8);
                        }
                    }
                };

            // Offline strict backtrace along chosen producer chain using cached outputs (ULP=0)
            auto backtrace_chain = [&](int start_op_idx) {
                fmt::print("\n========== Strict Divergence Backtrace (ULP=0) [{}] ==========\n", phase_tag);

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
                    POWERSERVE_UNUSED(t_trace);
                    if (!t_trace || p < 0) {
                        fmt::print("  [BT] op#{} type={} no traceable input, stop.\n", cur, op_type_to_string(cur_op->op));
                        break;
                    }

                    auto itg = ggml_op_outs.find(op_out_key(p, 0));
                    auto ito = ocl_op_outs.find(op_out_key(p, 0));
                    auto itpnode = ocl_op_nodes.find(p);
                    const OpNode *pnode = (itpnode == ocl_op_nodes.end()) ? nullptr : itpnode->second;

                    if (itg == ggml_op_outs.end() || ito == ocl_op_outs.end() || !pnode) {
                        fmt::print("  [BT] producer op#{} missing cached out0/opnode, stop.\n", p);
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

                fmt::print("==============================================================\n");
            };

            // compare current op out0 against GGML cached out0 using ATOL/RTOL
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
                    fmt::print("\n[MISMATCH][{}] op#{} type={} out#0 size mismatch ocl={} ggml={}\n",
                               phase_tag, op_idx, op_type_to_string(op->op), ocl_vec.size(), gg_vec.size());
                    print_tensor_meta2(out, "out", producer);
                    backtrace_chain(op_idx);
                    std::exit(1);
                }

                float atol = ATOL;
                float rtol = RTOL;
                if (matmul_B_is_from_silu(op, producer, ocl_op_nodes)) {
                    atol = ATOL_MATMUL_B_FROM_SILU;
                    rtol = RTOL_MATMUL_B_FROM_SILU;
                }

                size_t bad_i = 0;
                float diff = 0.f;
                if (!allclose_span(ocl_vec, gg_vec, atol, rtol, &bad_i, &diff)) {
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

                // live compare
                compare_out0(op_idx, op);
            });

            auto ret_o = model_ocl->forward(in_tokens, in_pos, in_mask, lm_head);
            set_op_after_exec_hook(nullptr);

            return {ret_g, ret_o};
        };

    // =========================
    // PREFILL compare (original behavior)
    // =========================
    {
        auto mask = CausalAttentionMask(tokens.size());

        auto [ret_g, ret_o] = run_forward_per_op_cmp(tokens, pos, mask, /*lm_head*/true, "PREFILL");

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
    // DECODE compare (N steps)
    // Mimic ModelTokenIterator: prefill(prompt[:-1], lm_head=false), then decode 1 token each step
    // =========================
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        // prefill: feed all prompt tokens except the last one (build KV only)
        if (tokens.size() > 1) {
            std::vector<Token> prefill_tokens(tokens.begin(), tokens.end() - 1);
            std::vector<int> prefill_pos(prefill_tokens.size());
            for (size_t i = 0; i < prefill_pos.size(); ++i) prefill_pos[i] = (int)i;
            auto prefill_mask = CausalAttentionMask(prefill_tokens.size());

            (void)run_forward_per_op_cmp(prefill_tokens, prefill_pos, prefill_mask, /*lm_head*/false, "DECODE_PREFILL");
        }

        int token_in = tokens.back(); // first decode token = prompt last token
        for (int step = 0; step < DECODE_STEPS; ++step) {
            size_t pos_g = platform->get_kv_position(id_g);
            size_t pos_o = platform->get_kv_position(id_o);
            POWERSERVE_ASSERT(pos_g == pos_o);

            std::vector<Token> step_tokens(1, token_in);
            std::vector<int> step_pos(1, (int)pos_g);
            auto step_mask = CausalAttentionMask(1);

            auto [ret_g, ret_o] = run_forward_per_op_cmp(step_tokens, step_pos, step_mask, /*lm_head*/true, "DECODE");

            auto lg = ret_g.logits_vector.back();
            auto lo = ret_o.logits_vector.back();

            // end-to-end logits compare for this step
            size_t bad_i = 0;
            float diff = 0.f;
            if (!allclose_span(lo, lg, ATOL, RTOL, &bad_i, &diff)) {
                fmt::print("\n[DECODE logits mismatch] step={} pos={} token_in={}\n", step, pos_g, token_in);
                fmt::print("  bad_i={}, ocl={}, ggml={}, diff={}\n", bad_i, lo[bad_i], lg[bad_i], diff);
                dump_topk(lg, TOPK, "ggml");
                dump_topk(lo, TOPK, "opencl");
                return 1;
            }

            int next_token = argmax_span(lg);
            fmt::print("[decode step {}] pos={} token_in={} -> next_token={}\n", step, pos_g, token_in, next_token);
            token_in = next_token;
        }

        POWERSERVE_LOG_INFO("Decode per-op compare PASS ({} steps)", DECODE_STEPS);
    }


    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
