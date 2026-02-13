#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"
#include "graph/op_type.hpp"
#include "graph/op_params.hpp"
#include "core/tensor.hpp"
#include "executor/executor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

using namespace powerserve;

static const char *MODEL_DIR = "/data/local/tmp/ziqian/models/qwen2-0.5b-work/qwen2-0.5b";
static const char *PROMPT    = "In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of large language models (LLMs). Traditionally, these models have been deployed in cloud environments due to their computational demands. However, the emergence of on-edge LLMs is reshaping how AI can be utilized at the edge of networks, offering numerous advantages in terms of latency, privacy, and accessibility.";

static int    N_THREADS    = 8;
static size_t BATCH_SIZE   = 4;
static int    DECODE_STEPS = 0;
static int    TOPK_PRINT   = 8;
static int    RECHECK_SOFTMAX_UP_OP_IDX = 143;

// Layer-diff gating. Keep these looser than strict allclose, good for locating drift.
static float FAIL_MAX_ABS   = 1e-2f;
static float FAIL_MEAN_ABS  = 1e-3f;
static float FAIL_COSINE_LT = 0.9990f;

// Keep softmax_ext deep recheck off by default to reduce log noise.
// Enable only when needed:
//   POWERSERVE_LAYER_DIFF_RECHECK_SOFTMAX_UP=1
static bool enable_softmax_up_recheck() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    const char *v = std::getenv("POWERSERVE_LAYER_DIFF_RECHECK_SOFTMAX_UP");
    cached = (v && (
        std::strcmp(v, "1") == 0 ||
        std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "TRUE") == 0 ||
        std::strcmp(v, "on") == 0 ||
        std::strcmp(v, "ON") == 0
    )) ? 1 : 0;
    return cached == 1;
}

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

static inline uint64_t op_in_key(int op_idx, int in_idx) {
    return (uint64_t(uint32_t(op_idx)) << 32) | uint32_t(in_idx);
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

static void print_tensor_meta(const Tensor *t, const char *tag) {
    if (!t) {
        std::printf("    %s: <null>\n", tag);
        return;
    }
    const auto s = t->m_shape;
    if (!t->m_data) {
        std::printf("    %s: dtype=%d shape=[%zu,%zu,%zu,%zu] [NO_DATA]\n",
                    tag, (int)t->m_dtype, s[0], s[1], s[2], s[3]);
        return;
    }
    if (auto *cb = dynamic_cast<CPUBuffer *>(t->m_data.get())) {
        std::printf("    %s: dtype=%d shape=[%zu,%zu,%zu,%zu] strideB=[%zu,%zu,%zu,%zu] [CPU]\n",
                    tag, (int)t->m_dtype, s[0], s[1], s[2], s[3],
                    cb->m_stride[0], cb->m_stride[1], cb->m_stride[2], cb->m_stride[3]);
        return;
    }
    if (auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer *>(t->m_data.get())) {
        const auto st = clb->get_stride();
        const size_t off = clb->get_base_offset();
        std::printf("    %s: dtype=%d shape=[%zu,%zu,%zu,%zu] strideB=[%zu,%zu,%zu,%zu] base_off=%zu [OpenCL]\n",
                    tag, (int)t->m_dtype, s[0], s[1], s[2], s[3],
                    st[0], st[1], st[2], st[3], off);
        return;
    }
    std::printf("    %s: dtype=%d shape=[%zu,%zu,%zu,%zu] [UNKNOWN_BUFFER]\n",
                tag, (int)t->m_dtype, s[0], s[1], s[2], s[3]);
}

struct ByteSnapshot {
    DataType dtype = DataType::FP32;
    Shape shape{};
    std::vector<uint8_t> bytes;
};

static uint64_t fnv1a64(const uint8_t *data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<uint8_t> tensor_to_bytes_any(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    std::vector<uint8_t> out;
    if (!t || !t->m_data) return out;

    if (auto *cb = dynamic_cast<CPUBuffer *>(t->m_data.get())) {
        const size_t bytes = cb->m_stride[3] * t->m_shape[3];
        out.resize(bytes);
        std::memcpy(out.data(), cb->m_data, bytes);
        return out;
    }

    if (auto *ob = dynamic_cast<powerserve::opencl::OpenCLBuffer *>(t->m_data.get())) {
        if (!cl_backend) return out;
        const Stride st = ob->get_stride();
        const size_t bytes = st[3] * t->m_shape[3];
        Tensor tmp_cpu(t->m_dtype, t->m_shape);
        tmp_cpu.m_data = std::make_shared<CPUBuffer>(st, std::malloc(bytes), true);
        cl_backend->copy(&tmp_cpu, t);
        auto *tmp_cb = dynamic_cast<CPUBuffer *>(tmp_cpu.m_data.get());
        POWERSERVE_ASSERT(tmp_cb);
        out.resize(bytes);
        std::memcpy(out.data(), tmp_cb->m_data, bytes);
        return out;
    }

    return out;
}

static void dump_byte_compare(const char *name, const std::vector<uint8_t> &ocl_b, const std::vector<uint8_t> &gg_b) {
    if (ocl_b.size() != gg_b.size() || ocl_b.empty()) {
        std::printf("    %s: byte-compare unavailable (ocl=%zu ggml=%zu)\n", name, ocl_b.size(), gg_b.size());
        return;
    }

    size_t first_diff = ocl_b.size();
    for (size_t i = 0; i < ocl_b.size(); ++i) {
        if (ocl_b[i] != gg_b[i]) {
            first_diff = i;
            break;
        }
    }

    const uint64_t h_ocl = fnv1a64(ocl_b.data(), ocl_b.size());
    const uint64_t h_gg = fnv1a64(gg_b.data(), gg_b.size());
    std::printf("    %s: byte_size=%zu hash_ocl=0x%llx hash_ggml=0x%llx first_diff=%s\n",
                name, ocl_b.size(),
                (unsigned long long)h_ocl, (unsigned long long)h_gg,
                (first_diff < ocl_b.size() ? "yes" : "no"));
    if (first_diff < ocl_b.size()) {
        std::printf("      first_diff_at=%zu ocl=0x%02x ggml=0x%02x\n",
                    first_diff, (unsigned)ocl_b[first_diff], (unsigned)gg_b[first_diff]);
    }
}

static void dump_first_diff_coords(const Tensor *t, size_t first_diff) {
    if (!t || !t->m_data) return;
    if (first_diff == (size_t)-1) return;
    if (t->m_dtype != DataType::FP32) return;
    if (first_diff % sizeof(float) != 0) return;

    size_t st[4] = {0, 0, 0, 0};
    if (auto *cb = dynamic_cast<CPUBuffer *>(t->m_data.get())) {
        st[0] = cb->m_stride[0];
        st[1] = cb->m_stride[1];
        st[2] = cb->m_stride[2];
        st[3] = cb->m_stride[3];
    } else if (auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer *>(t->m_data.get())) {
        const auto s = clb->get_stride();
        st[0] = s[0];
        st[1] = s[1];
        st[2] = s[2];
        st[3] = s[3];
    } else {
        return;
    }

    if (st[0] == 0 || st[1] == 0 || st[2] == 0 || st[3] == 0) return;
    const auto shape = t->m_shape;

    size_t off = first_diff;
    size_t i3 = off / st[3];
    off %= st[3];
    size_t i2 = off / st[2];
    off %= st[2];
    size_t i1 = off / st[1];
    off %= st[1];
    size_t i0 = off / st[0];

    if (i0 < shape[0] && i1 < shape[1] && i2 < shape[2] && i3 < shape[3]) {
        std::printf("      first_diff_idx=[%zu,%zu,%zu,%zu] (i0,i1,i2,i3)\n", i0, i1, i2, i3);
    }
}

static size_t first_diff_index(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
    if (a.size() != b.size() || a.empty()) return (size_t)-1;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return i;
    }
    return (size_t)-1;
}

static void dump_input_slot_compare(
    const char *tag,
    const Tensor *ocl_t,
    const ByteSnapshot *gg_snap,
    powerserve::opencl::OpenCLBackend *cl_backend
) {
    print_tensor_meta(ocl_t, tag);
    if (!gg_snap) {
        std::printf("    %s: ggml snapshot missing\n", tag);
        return;
    }
    std::vector<uint8_t> ocl_b = tensor_to_bytes_any(ocl_t, cl_backend);
    if (ocl_b.empty() || gg_snap->bytes.empty()) {
        std::printf("    %s: byte-compare unavailable (ocl=%zu ggml=%zu)\n",
                    tag, ocl_b.size(), gg_snap->bytes.size());
        return;
    }
    dump_byte_compare(tag, ocl_b, gg_snap->bytes);
    size_t first_diff = first_diff_index(ocl_b, gg_snap->bytes);
    if (first_diff != (size_t)-1) {
        dump_first_diff_coords(ocl_t, first_diff);
    }
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
    bool has_non_finite = false;
    size_t non_finite_count = 0;
};

struct LayerAgg {
    int count = 0;
    int non_finite_ops = 0;
    float max_abs = 0.0f;
    float mean_abs_acc = 0.0f;
    float min_cosine = 1.0f;
};

struct UlpStats {
    size_t total = 0;
    size_t diff_count = 0;
    uint32_t max_ulp = 0;
};

static inline uint32_t float_to_ordered_u32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(uint32_t));
    if (u & 0x80000000u) return 0x80000000u - u;
    return u + 0x80000000u;
}

static UlpStats ulp_stats(const std::vector<float> &a, const std::vector<float> &b) {
    UlpStats s;
    if (a.size() != b.size() || a.empty()) return s;
    s.total = a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        const float av = a[i];
        const float bv = b[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) continue;
        const uint32_t ua = float_to_ordered_u32(av);
        const uint32_t ub = float_to_ordered_u32(bv);
        const uint32_t d = (ua > ub) ? (ua - ub) : (ub - ua);
        if (d > 0) {
            s.diff_count++;
            if (d > s.max_ulp) s.max_ulp = d;
        }
    }
    return s;
}

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
    size_t finite_count = 0;
    float mx = 0.0f;
    size_t mi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float rv = ref[i];
        const float ov = ocl[i];
        if (!std::isfinite(rv) || !std::isfinite(ov)) {
            r.has_non_finite = true;
            r.non_finite_count++;
            continue;
        }
        float d = std::fabs(rv - ov);
        sum += d;
        finite_count++;
        if (d > mx) {
            mx = d;
            mi = i;
        }
        dot += (double)rv * (double)ov;
        nr += (double)rv * (double)rv;
        no += (double)ov * (double)ov;
    }

    if (finite_count == 0) {
        r.max_abs = std::numeric_limits<float>::infinity();
        r.mean_abs = std::numeric_limits<float>::infinity();
        r.cosine = -1.0f;
        r.worst_i = 0;
        r.ref_v = ref[0];
        r.ocl_v = ocl[0];
        return r;
    }

    r.max_abs = mx;
    r.mean_abs = (float)(sum / (double)finite_count);
    r.cosine = (nr > 0.0 && no > 0.0) ? (float)(dot / (std::sqrt(nr) * std::sqrt(no))) : 0.0f;
    r.worst_i = mi;
    r.ref_v = ref[mi];
    r.ocl_v = ocl[mi];
    return r;
}

// CPU reference for softmax_ext that mirrors ggml_compute_forward_soft_max_f32 semantics.
static bool softmax_ext_ref_cpu(
    const Shape &x_shape,
    const std::vector<float> &x,
    const Shape &mask_shape,
    const std::vector<float> &mask,
    float scale,
    float max_bias,
    std::vector<float> &out,
    std::string *err_msg = nullptr
) {
    const int ne00 = (int)x_shape[0];
    const int ne01 = (int)x_shape[1];
    const int ne02 = (int)x_shape[2];
    const int ne03 = (int)x_shape[3];
    if (ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0) {
        if (err_msg) *err_msg = "invalid x shape";
        return false;
    }

    const size_t x_elems = (size_t)ne00 * (size_t)ne01 * (size_t)ne02 * (size_t)ne03;
    if (x.size() != x_elems) {
        if (err_msg) *err_msg = "x size mismatch";
        return false;
    }

    const int me00 = (int)mask_shape[0];
    const int me01 = (int)mask_shape[1];
    const int me02 = (int)mask_shape[2];
    const int me03 = (int)mask_shape[3];
    const size_t m_elems = (size_t)me00 * (size_t)me01 * (size_t)me02 * (size_t)me03;
    if (mask.size() != m_elems) {
        if (err_msg) *err_msg = "mask size mismatch";
        return false;
    }
    if (!(me00 == ne00 && me01 == ne01 && me02 == 1 && me03 == 1)) {
        if (err_msg) *err_msg = "mask shape not [ne00, ne01, 1, 1]";
        return false;
    }

    out.assign(x_elems, 0.0f);

    const uint32_t n_head = (uint32_t)ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)std::floor(std::log2((double)n_head));
    const float m0 = std::pow(2.0f, -(max_bias)        / (float)n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int nrows = ne01 * ne02 * ne03;
    for (int row = 0; row < nrows; ++row) {
        const uint32_t h = (uint32_t)((row / ne01) % ne02);
        const float slope = (max_bias > 0.0f)
            ? (h < n_head_log2 ? std::pow(m0, (float)h + 1.0f)
                               : std::pow(m1, (float)(2 * ((int)h - (int)n_head_log2) + 1)))
            : 1.0f;

        const int x_base = row * ne00;
        const int m_base = (row % ne01) * ne00;

        float maxv = -INFINITY;
        for (int i = 0; i < ne00; ++i) {
            const float v = x[x_base + i] * scale + slope * mask[m_base + i];
            if (v > maxv) maxv = v;
        }

        double sum = 0.0;
        for (int i = 0; i < ne00; ++i) {
            const float v = x[x_base + i] * scale + slope * mask[m_base + i];
            const float e = std::exp(v - maxv);
            out[x_base + i] = e;
            sum += (double)e;
        }

        if (!(sum > 0.0) || !std::isfinite(sum)) {
            if (err_msg) *err_msg = "non-finite or non-positive softmax sum";
            return false;
        }

        const float inv = 1.0f / (float)sum;
        for (int i = 0; i < ne00; ++i) {
            out[x_base + i] *= inv;
        }
    }

    return true;
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
        const bool softmax_recheck = enable_softmax_up_recheck();
        std::unordered_map<uint64_t, std::vector<float>> gg_outs;
        std::unordered_map<uint64_t, ByteSnapshot> gg_out_bytes;
        std::unordered_map<uint64_t, ByteSnapshot> gg_in_bytes;

        set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
            if (softmax_recheck) {
                for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                    Node *in_node = op->prev[pi];
                    const Tensor *in = in_node ? in_node->tensor() : nullptr;
                    if (!in || !in->m_data) continue;

                    ByteSnapshot snap;
                    snap.dtype = in->m_dtype;
                    snap.shape = in->m_shape;
                    snap.bytes = tensor_to_bytes_any(in, /*cl_backend=*/nullptr);
                    gg_in_bytes[op_in_key(op_idx, pi)] = std::move(snap);
                }
            }

            for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                Tensor *out = op->next[oi]->tensor();
                if (!out || !out->m_data) continue;
                if (out->m_dtype == DataType::FP32) {
                    gg_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
                }

                if (softmax_recheck) {
                    ByteSnapshot snap;
                    snap.dtype = out->m_dtype;
                    snap.shape = out->m_shape;
                    snap.bytes = tensor_to_bytes_any(out, /*cl_backend=*/nullptr);
                    gg_out_bytes[op_out_key(op_idx, oi)] = std::move(snap);
                }
            }
        });

        auto ret_g = model_ggml->forward(in_tokens, in_pos, mask, lm_head);
        set_op_after_exec_hook(nullptr);

        auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend *>(
            platform->get_backend(model_ocl->m_config->model_id));
        POWERSERVE_ASSERT(cl_backend);

        std::vector<DiffRec> diffs;
        std::unordered_map<int, LayerAgg> by_layer;
        int layer_cursor = 0;
        bool softmax_up_rechecked = false;

        auto dump_op_inputs = [&](const OpNode *op, int op_idx, const char *tag) {
            if (!op) return;
            std::printf("\n[%s] phase=%s step=%d op#%d type=%s inputs=%zu\n",
                        tag, phase, step, op_idx, op_type_to_string(op->op), op->prev.size());
            for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                const Tensor *in = op->prev[pi] ? op->prev[pi]->tensor() : nullptr;
                const ByteSnapshot *g = nullptr;
                auto git = gg_in_bytes.find(op_in_key(op_idx, pi));
                if (git != gg_in_bytes.end()) g = &git->second;
                std::string itag = "  input[" + std::to_string(pi) + "]";
                dump_input_slot_compare(itag.c_str(), in, g, cl_backend);
            }
        };

        auto dump_op_output_compare = [&](int op_idx, Tensor *out, const char *tag) {
            if (!out || !out->m_data) {
                std::printf("    %s: output null\n", tag);
                return;
            }
            auto gitb = gg_out_bytes.find(op_out_key(op_idx, 0));
            if (gitb == gg_out_bytes.end() || gitb->second.bytes.empty()) {
                std::printf("    %s: ggml output snapshot missing\n", tag);
                return;
            }
            auto cur_b = tensor_to_bytes_any(out, cl_backend);
            if (cur_b.empty()) {
                std::printf("    %s: opencl output copy failed\n", tag);
                return;
            }
            dump_byte_compare(tag, cur_b, gitb->second.bytes);
            size_t first_diff = first_diff_index(cur_b, gitb->second.bytes);
            if (first_diff != (size_t)-1) {
                dump_first_diff_coords(out, first_diff);
            }
        };

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

            if (softmax_recheck &&
                !softmax_up_rechecked &&
                op_idx == RECHECK_SOFTMAX_UP_OP_IDX &&
                op->op == OpType::SOFTMAX_EXT) {
                softmax_up_rechecked = true;
                dump_op_inputs(op, op_idx, "RECHECK-SOFTMAX-UP");
                dump_op_output_compare(op_idx, out, "    output[0]");

                if (ocl.size() == it->second.size() && !ocl.empty()) {
                    auto us = ulp_stats(it->second, ocl);
                    std::printf("    output[0]_ulp: diff_count=%zu/%zu max_ulp=%u\n",
                                us.diff_count, us.total, us.max_ulp);
                }

                if (op->prev.size() >= 2) {
                    const Tensor *in0 = op->prev[0] ? op->prev[0]->tensor() : nullptr;
                    const Tensor *in1 = op->prev[1] ? op->prev[1]->tensor() : nullptr;
                    if (in0 && in1 && in0->m_data && in1->m_data &&
                        in0->m_dtype == DataType::FP32 && in1->m_dtype == DataType::FP32) {
                        auto v0 = tensor_to_f32_vec_any(in0, cl_backend);
                        auto v1 = tensor_to_f32_vec_any(in1, cl_backend);
                        if (!v0.empty() && !v1.empty()) {
                            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();
                            std::vector<float> ref2;
                            std::string err;
                            if (softmax_ext_ref_cpu(in0->m_shape, v0, in1->m_shape, v1, scale, max_bias, ref2, &err) &&
                                ref2.size() == ocl.size() && ref2.size() == it->second.size() && !ref2.empty()) {
                                auto d_ocl_ref2 = diff_vecs(ref2, ocl);
                                auto d_gg_ref2 = diff_vecs(ref2, it->second);
                                std::printf(
                                    "    ref2_compare: ggml_cached_vs_ref2(max_abs=%.6g mean_abs=%.6g cosine=%.8f) "
                                    "opencl_vs_ref2(max_abs=%.6g mean_abs=%.6g cosine=%.8f)\n",
                                    d_gg_ref2.max_abs, d_gg_ref2.mean_abs, d_gg_ref2.cosine,
                                    d_ocl_ref2.max_abs, d_ocl_ref2.mean_abs, d_ocl_ref2.cosine);
                            } else if (!err.empty()) {
                                std::printf("    ref2_compare: skipped (%s)\n", err.c_str());
                            } else {
                                std::printf("    ref2_compare: skipped (size mismatch or empty)\n");
                            }
                        }
                    }
                }
            }

            auto &agg = by_layer[d.layer];
            agg.count++;
            if (d.has_non_finite) {
                agg.non_finite_ops++;
            }
            agg.max_abs = std::max(agg.max_abs, d.max_abs);
            if (std::isfinite(d.mean_abs)) {
                agg.mean_abs_acc += d.mean_abs;
            }
            if (std::isfinite(d.cosine)) {
                agg.min_cosine = std::min(agg.min_cosine, d.cosine);
            }
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
                std::printf("  layer=%d count=%d non_finite_ops=%d max_abs=%.6g mean_abs=%.6g min_cosine=%.8f\n",
                            lid, a.count, a.non_finite_ops, a.max_abs, mean_abs, a.min_cosine);
            }

            const auto &w = diffs.front();
            const bool cosine_fail = (w.max_abs > 1e-8f || w.mean_abs > 1e-8f) && w.cosine < FAIL_COSINE_LT;
            if (w.max_abs > FAIL_MAX_ABS || w.mean_abs > FAIL_MEAN_ABS || cosine_fail) {
                std::printf("[FAIL-TRIGGER] phase=%s step=%d worst: op#%d layer=%d type=%s max_abs=%.6g mean_abs=%.6g cosine=%.8f\n",
                            phase, step, w.op_idx, w.layer, op_type_to_string(w.op), w.max_abs, w.mean_abs, w.cosine);
                if (lm_head && !ret_g.logits_vector.empty() && !ret_o.logits_vector.empty()) {
                    auto lg = ret_g.logits_vector.back();
                    auto lo = ret_o.logits_vector.back();
                    dump_topk(lg, TOPK_PRINT, "ggml");
                    dump_topk(lo, TOPK_PRINT, "opencl");
                } else {
                    std::printf("[FAIL-TRIGGER] logits unavailable (lm_head=%d, ggml_logits=%zu, opencl_logits=%zu), skip topk dump\n",
                                lm_head ? 1 : 0, ret_g.logits_vector.size(), ret_o.logits_vector.size());
                }
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
        if (ret_g.logits_vector.empty() || ret_o.logits_vector.empty()) {
            POWERSERVE_LOG_ERROR("PREFILL returned empty logits (ggml={}, opencl={})",
                                 ret_g.logits_vector.size(), ret_o.logits_vector.size());
            return 1;
        }
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
            if (ret_g.logits_vector.empty() || ret_o.logits_vector.empty()) {
                POWERSERVE_LOG_ERROR("DECODE step={} returned empty logits (ggml={}, opencl={})",
                                     step, ret_g.logits_vector.size(), ret_o.logits_vector.size());
                return 1;
            }
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
