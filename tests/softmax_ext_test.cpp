#include "backend/opencl/opencl_backend.hpp"
#include "backend/ggml/ggml_wrapper.cpp"
#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace powerserve::opencl {

static inline Stride make_contig_stride_bytes(const Shape &shape, size_t elem_bytes) {
    Stride st{};
    st[0] = elem_bytes;
    st[1] = shape[0] * st[0];
    st[2] = shape[1] * st[1];
    st[3] = shape[2] * st[2];
    return st;
}

static inline Tensor make_cpu_tensor_f32(const Shape &shape, std::vector<float> &storage) {
    Tensor t(DataType::FP32, shape);
    storage.resize(t.n_elements(), 0.0f);
    auto stride = make_contig_stride_bytes(shape, sizeof(float));
    t.m_data = std::make_shared<powerserve::CPUBuffer>(stride, storage.data());
    return t;
}

static inline Tensor make_opencl_tensor_f32(OpenCLBackend &b, const Shape &shape) {
    Tensor t(DataType::FP32, shape);
    t.m_data = b.create_buffer(shape, DataType::FP32);
    return t;
}

struct DiffRec {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float cosine = 1.0f;
    size_t worst_i = 0;
    float ref_v = 0.0f;
    float ocl_v = 0.0f;
};

static DiffRec diff_vecs(const std::vector<float> &ref, const std::vector<float> &ocl) {
    DiffRec r;
    if (ref.size() != ocl.size() || ref.empty()) {
        r.max_abs = INFINITY;
        r.mean_abs = INFINITY;
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
        if (!std::isfinite(rv) || !std::isfinite(ov)) continue;
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
        r.max_abs = INFINITY;
        r.mean_abs = INFINITY;
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

static inline void fill_mask_causal(std::vector<float> &mask, const Shape &shape) {
    const size_t ne00 = shape[0];
    const size_t ne01 = shape[1];
    mask.assign(ne00 * ne01, 0.0f);
    for (size_t i01 = 0; i01 < ne01; ++i01) {
        for (size_t i00 = 0; i00 < ne00; ++i00) {
            mask[i01 * ne00 + i00] = (i00 <= i01) ? 0.0f : -INFINITY;
        }
    }
}

static inline void fill_random(std::vector<float> &x, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &v : x) v = dist(rng);
}

struct TestCase {
    std::string name;
    Shape x_shape;
    float scale;
    float max_bias;
    float rand_lo;
    float rand_hi;
    uint32_t seed;
};

static bool run_softmax_ext_case(
    const TestCase &tc,
    powerserve::ggml::GGMLBackend &ggml_backend,
    OpenCLBackend &ocl_backend
) {
    const Shape x_shape = tc.x_shape;
    const Shape m_shape{ x_shape[0], x_shape[1], 1, 1 };

    std::vector<float> x_storage;
    std::vector<float> m_storage;
    std::vector<float> out_gg_storage;
    std::vector<float> out_ocl_storage;

    Tensor x_cpu   = make_cpu_tensor_f32(x_shape, x_storage);
    Tensor m_cpu   = make_cpu_tensor_f32(m_shape, m_storage);
    Tensor out_gg  = make_cpu_tensor_f32(x_shape, out_gg_storage);
    Tensor out_ocl = make_cpu_tensor_f32(x_shape, out_ocl_storage);

    fill_random(x_storage, tc.rand_lo, tc.rand_hi, tc.seed);
    fill_mask_causal(m_storage, m_shape);

    // ggml path
    ggml_backend.softmax_ext(&out_gg, &x_cpu, &m_cpu, tc.scale, tc.max_bias);

    // OpenCL path
    Tensor x_cl = make_opencl_tensor_f32(ocl_backend, x_shape);
    Tensor m_cl = make_opencl_tensor_f32(ocl_backend, m_shape);
    Tensor o_cl = make_opencl_tensor_f32(ocl_backend, x_shape);
    if (!x_cl.m_data || !m_cl.m_data || !o_cl.m_data) {
        POWERSERVE_LOG_ERROR("OpenCL buffer allocation failed for test case {}", tc.name);
        return false;
    }

    ocl_backend.copy(&x_cl, &x_cpu);
    ocl_backend.copy(&m_cl, &m_cpu);
    ocl_backend.softmax_ext(&o_cl, &x_cl, &m_cl, tc.scale, tc.max_bias);
    ocl_backend.copy(&out_ocl, &o_cl);

    DiffRec d = diff_vecs(out_gg_storage, out_ocl_storage);
    std::printf(
        "[softmax_ext_test] case=%s shape=[%zu,%zu,%zu,%zu] scale=%.6g max_bias=%.6g "
        "max_abs=%.6g mean_abs=%.6g cosine=%.8f worst_i=%zu gg=%.6g ocl=%.6g\n",
        tc.name.c_str(),
        x_shape[0], x_shape[1], x_shape[2], x_shape[3],
        tc.scale, tc.max_bias,
        d.max_abs, d.mean_abs, d.cosine, d.worst_i, d.ref_v, d.ocl_v
    );

    const float FAIL_MAX_ABS = 1e-4f;
    const float FAIL_MEAN_ABS = 1e-6f;
    const float FAIL_COSINE_LT = 0.99999f;

    if (d.max_abs > FAIL_MAX_ABS || d.mean_abs > FAIL_MEAN_ABS || d.cosine < FAIL_COSINE_LT) {
        std::vector<float> ref2;
        std::string err;
        if (softmax_ext_ref_cpu(x_shape, x_storage, m_shape, m_storage, tc.scale, tc.max_bias, ref2, &err)) {
            auto d_gg_ref = diff_vecs(ref2, out_gg_storage);
            auto d_ocl_ref = diff_vecs(ref2, out_ocl_storage);
            std::printf(
                "  ref2 diff: ggml(max_abs=%.6g mean_abs=%.6g cosine=%.8f) "
                "opencl(max_abs=%.6g mean_abs=%.6g cosine=%.8f)\n",
                d_gg_ref.max_abs, d_gg_ref.mean_abs, d_gg_ref.cosine,
                d_ocl_ref.max_abs, d_ocl_ref.mean_abs, d_ocl_ref.cosine
            );
        } else {
            std::printf("  ref2 compute failed: %s\n", err.c_str());
        }
        return false;
    }

    return true;
}

} // namespace powerserve::opencl

int main() {
    using powerserve::opencl::OpenCLBackend;
    using powerserve::opencl::TestCase;
    using powerserve::Shape;

    powerserve::ModelConfig::LLMConfig cfg{};
    cfg.dim        = 64;
    cfg.vocab_size = 64;
    cfg.n_layers   = 1;
    cfg.n_heads    = 4;
    cfg.n_kv_heads = 4;
    cfg.kv_dim     = 64;
    cfg.head_size  = 16;
    cfg.seq_len    = 128;

    powerserve::HyperParams hp{};
    hp.n_threads = 4;

    powerserve::ggml::GGMLBackend ggml_backend(cfg, hp);
    ggml_backend.setup_threadpool();

    OpenCLBackend ocl_backend(cfg, hp);
    if (!ocl_backend.initialize()) {
        POWERSERVE_LOG_ERROR("softmax_ext_test: OpenCL backend init failed");
        return 1;
    }

    std::vector<TestCase> cases = {
        {"small_no_alibi", Shape{32, 8, 4, 1}, 0.125f, 0.0f,  -5.0f,  5.0f, 123},
        {"alibi_pow2",     Shape{64, 16, 8, 1}, 0.125f, 8.0f,  -5.0f,  5.0f, 456},
        {"alibi_nonpow2",  Shape{96, 16, 8, 1}, 0.0625f, 8.0f, -5.0f,  5.0f, 789},
        {"large_vals",     Shape{128, 4, 4, 2}, 0.125f, 8.0f,  -50.0f, 50.0f, 999},
    };

    size_t max_ne00 = 0;
    for (const auto &tc : cases) max_ne00 = std::max(max_ne00, (size_t)tc.x_shape[0]);
    const size_t work_bytes = (max_ne00 + 64) * (size_t)hp.n_threads * sizeof(float);
    ggml_backend.setup_work_data(work_bytes);

    bool all_ok = true;
    for (const auto &tc : cases) {
        if (!powerserve::opencl::run_softmax_ext_case(tc, ggml_backend, ocl_backend)) {
            all_ok = false;
        }
    }

    ggml_backend.reset_threadpool();
    return all_ok ? 0 : 1;
}
