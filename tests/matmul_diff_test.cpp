#include "backend/opencl/opencl_backend.hpp"
#include "backend/ggml/ggml_wrapper.cpp"
#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace powerserve::opencl {

struct DiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float cosine = 0.0f;
    int argmax_ref = -1;
    int argmax_test = -1;
};

struct Threshold {
    float atol;
    float rtol;
    float cosine_min;
};

struct TestCase {
    const char *name;
    DataType weight_dtype;
    int K;
    int N;
    int M;
    uint32_t seed;
    Threshold th;
};

static inline bool matmul_diff_dbg_enabled() {
    const char *e = std::getenv("POWERSERVE_OCL_MATMUL_DBG");
    return e && e[0] != '\0' && e[0] != '0';
}

static inline Stride make_contig_stride_bytes(const Shape &shape, size_t elem_bytes) {
    Stride st{};
    st[0] = (int)elem_bytes;
    st[1] = (int)(shape[0] * (size_t)st[0]);
    st[2] = (int)(shape[1] * (size_t)st[1]);
    st[3] = (int)(shape[2] * (size_t)st[2]);
    return st;
}

static inline Tensor make_cpu_tensor_f32(const Shape &shape, std::vector<float> &storage) {
    Tensor t(DataType::FP32, shape);
    storage.resize(t.n_elements(), 0.0f);
    t.m_data = std::make_shared<CPUBuffer>(make_contig_stride_bytes(shape, sizeof(float)), storage.data());
    return t;
}

static inline Tensor make_cpu_tensor_f16(const Shape &shape, std::vector<ggml_fp16_t> &storage) {
    Tensor t(DataType::FP16, shape);
    storage.resize(t.n_elements(), (ggml_fp16_t)0);
    t.m_data = std::make_shared<CPUBuffer>(make_contig_stride_bytes(shape, sizeof(ggml_fp16_t)), storage.data());
    return t;
}

static inline Tensor make_cpu_tensor_q8_0(const Shape &shape, std::vector<block_q8_0> &storage) {
    Tensor t(DataType::GGML_Q8_0, shape);
    const ggml_type qtype = GGML_TYPE_Q8_0;
    const int K = (int)shape[0];
    const size_t row_size = (size_t)ggml_row_size(qtype, K);

    Stride st{};
    st[0] = (int)ggml_type_size(qtype);
    st[1] = (int)row_size;
    st[2] = (int)(shape[1] * (size_t)st[1]);
    st[3] = (int)(shape[2] * (size_t)st[2]);
    t.m_data = std::make_shared<CPUBuffer>(st, storage.data());
    return t;
}

static inline Tensor make_opencl_tensor(OpenCLBackend &backend, const Shape &shape, DataType dt) {
    Tensor t(dt, shape);
    t.m_data = backend.create_buffer(shape, dt);
    return t;
}

static inline uint32_t lcg_next(uint32_t &s) {
    s = 1664525u * s + 1013904223u;
    return s;
}

static inline float rand_symm(uint32_t &s) {
    return ((lcg_next(s) & 0x00FFFFFF) / float(0x01000000)) * 2.0f - 1.0f;
}

static inline void fill_f32(std::vector<float> &v, uint32_t seed, float scale = 0.25f) {
    uint32_t s = seed;
    for (float &x : v) {
        x = rand_symm(s) * scale;
    }
}

static inline int argmax(const std::vector<float> &v) {
    if (v.empty()) return -1;
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

static DiffStats calc_stats(const std::vector<float> &ref, const std::vector<float> &test) {
    DiffStats s{};
    if (ref.size() != test.size() || ref.empty()) {
        s.max_abs = std::numeric_limits<float>::infinity();
        s.mean_abs = std::numeric_limits<float>::infinity();
        s.cosine = 0.0f;
        return s;
    }

    double sum_abs = 0.0;
    double dot = 0.0;
    double nr = 0.0;
    double nt = 0.0;
    float mx = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        const float r = ref[i];
        const float t = test[i];
        const float d = std::fabs(r - t);
        if (d > mx) mx = d;
        sum_abs += d;
        dot += (double)r * (double)t;
        nr += (double)r * (double)r;
        nt += (double)t * (double)t;
    }

    s.max_abs = mx;
    s.mean_abs = (float)(sum_abs / (double)ref.size());
    s.cosine = (nr > 0.0 && nt > 0.0) ? (float)(dot / (std::sqrt(nr) * std::sqrt(nt))) : 0.0f;
    s.argmax_ref = argmax(ref);
    s.argmax_test = argmax(test);
    return s;
}

static bool allclose(const std::vector<float> &a, const std::vector<float> &b, float atol, float rtol, size_t *bad_i, float *bad_diff) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float tol = atol + rtol * std::fabs(b[i]);
        if (!(diff <= tol)) {
            if (bad_i) *bad_i = i;
            if (bad_diff) *bad_diff = diff;
            return false;
        }
    }
    return true;
}

static bool run_ggml_matmul_ref(
    const ModelConfig::LLMConfig &cfg,
    const HyperParams &hp,
    const Tensor *w_cpu,
    const Tensor *x_cpu,
    Tensor *dst_cpu) {
    powerserve::ggml::GGMLBackend ggml_be(cfg, hp);
    ggml_be.setup_threadpool();

    const ggml_type vec_dot = ggml_be.get_vec_dot_type(x_cpu);
    const ggml_type w_type = powerserve::ggml::convert_datatype_to_ggml(w_cpu->m_dtype);
    size_t work = 0;
    if (w_type != vec_dot) {
        work = (size_t)ggml_row_size(vec_dot, (int64_t)w_cpu->n_elements());
    }
    work = std::max(work, (size_t)(1 << 20));
    ggml_be.setup_work_data(work);

    ggml_be.matmul(dst_cpu, w_cpu, x_cpu);
    ggml_be.reset_threadpool();
    return true;
}

static bool should_run_case(const TestCase &) {
    return true;
}

static bool run_case(const ModelConfig::LLMConfig &cfg, const HyperParams &hp, const TestCase &tc) {
    if (!should_run_case(tc)) {
        POWERSERVE_LOG_INFO("[matmul_diff] skip case={} (filter)", tc.name);
        return true;
    }

    POWERSERVE_LOG_INFO("[matmul_diff] run case={} dtype={} K={} N={} M={}",
                        tc.name, (int)tc.weight_dtype, tc.K, tc.N, tc.M);

    OpenCLBackend backend(cfg, hp);
    if (!backend.initialize()) {
        POWERSERVE_LOG_ERROR("[matmul_diff] case={} OpenCL backend initialize failed", tc.name);
        return false;
    }

    Shape w_shape{(size_t)tc.K, (size_t)tc.N, 1, 1};
    Shape x_shape{(size_t)tc.K, (size_t)tc.M, 1, 1};
    Shape d_shape{(size_t)tc.N, (size_t)tc.M, 1, 1};
    if (matmul_diff_dbg_enabled()) {
        const char *expected_q8_path =
            (tc.weight_dtype == DataType::GGML_Q8_0 && tc.M >= 32 && (tc.K % 32) == 0)
                ? "kernel_mul_mm_q8_0_f32_l4_lm"
                : (tc.weight_dtype == DataType::GGML_Q8_0 ? "kernel_mul_mv_q8_0_f32_flat_or_simple" : "non-q8");
        POWERSERVE_LOG_INFO(
            "[matmul_diff][dbg] case={} shapes w=[{},{},{},{}] x=[{},{},{},{}] d=[{},{},{},{}] expected_q8_path={}",
            tc.name,
            w_shape[0], w_shape[1], w_shape[2], w_shape[3],
            x_shape[0], x_shape[1], x_shape[2], x_shape[3],
            d_shape[0], d_shape[1], d_shape[2], d_shape[3],
            expected_q8_path
        );
    }

    std::vector<float> w_f32((size_t)tc.K * (size_t)tc.N, 0.0f);
    std::vector<float> x_f32((size_t)tc.K * (size_t)tc.M, 0.0f);
    fill_f32(w_f32, tc.seed, 0.20f);
    fill_f32(x_f32, tc.seed ^ 0x9e3779b9u, 0.10f);

    std::vector<float> x_cpu_storage;
    Tensor x_cpu = make_cpu_tensor_f32(x_shape, x_cpu_storage);
    std::memcpy(x_cpu_storage.data(), x_f32.data(), x_f32.size() * sizeof(float));

    std::vector<float> w_cpu_f32_storage;
    std::vector<ggml_fp16_t> w_cpu_f16_storage;
    std::vector<block_q8_0> w_cpu_q8_storage;
    Tensor w_cpu(tc.weight_dtype, w_shape);

    if (tc.weight_dtype == DataType::FP32) {
        w_cpu = make_cpu_tensor_f32(w_shape, w_cpu_f32_storage);
        std::memcpy(w_cpu_f32_storage.data(), w_f32.data(), w_f32.size() * sizeof(float));
    } else if (tc.weight_dtype == DataType::FP16) {
        w_cpu = make_cpu_tensor_f16(w_shape, w_cpu_f16_storage);
        for (size_t i = 0; i < w_f32.size(); ++i) {
            w_cpu_f16_storage[i] = ggml_fp32_to_fp16(w_f32[i]);
        }
    } else if (tc.weight_dtype == DataType::GGML_Q8_0) {
        if ((tc.K % 32) != 0) {
            POWERSERVE_LOG_ERROR("[matmul_diff] case={} invalid K for Q8_0 (K must be multiple of 32)", tc.name);
            return false;
        }
        const int nb = tc.K / 32;
        w_cpu_q8_storage.resize((size_t)nb * (size_t)tc.N);
        w_cpu = make_cpu_tensor_q8_0(w_shape, w_cpu_q8_storage);
        for (int n = 0; n < tc.N; ++n) {
            const float *src_row = w_f32.data() + (size_t)tc.K * (size_t)n;
            block_q8_0 *dst_row = w_cpu_q8_storage.data() + (size_t)nb * (size_t)n;
            quantize_row_q8_0(src_row, (void *)dst_row, tc.K);
        }
    } else {
        POWERSERVE_LOG_ERROR("[matmul_diff] unsupported dtype in case={}", tc.name);
        return false;
    }

    Tensor w_dev = make_opencl_tensor(backend, w_shape, tc.weight_dtype);
    Tensor x_dev = make_opencl_tensor(backend, x_shape, DataType::FP32);
    Tensor d_dev = make_opencl_tensor(backend, d_shape, DataType::FP32);
    if (matmul_diff_dbg_enabled()) {
        auto *wcb = dynamic_cast<CPUBuffer *>(w_cpu.m_data.get());
        auto *xcb = dynamic_cast<CPUBuffer *>(x_cpu.m_data.get());
        auto *wdb = dynamic_cast<OpenCLBuffer *>(w_dev.m_data.get());
        auto *xdb = dynamic_cast<OpenCLBuffer *>(x_dev.m_data.get());
        auto *ddb = dynamic_cast<OpenCLBuffer *>(d_dev.m_data.get());
        if (wcb && xcb) {
            POWERSERVE_LOG_INFO(
                "[matmul_diff][dbg] case={} cpu_strideB w=[{},{},{},{}] x=[{},{},{},{}]",
                tc.name,
                wcb->m_stride[0], wcb->m_stride[1], wcb->m_stride[2], wcb->m_stride[3],
                xcb->m_stride[0], xcb->m_stride[1], xcb->m_stride[2], xcb->m_stride[3]
            );
        }
        if (wdb && xdb && ddb) {
            const auto wst = wdb->get_stride();
            const auto xst = xdb->get_stride();
            const auto dst = ddb->get_stride();
            POWERSERVE_LOG_INFO(
                "[matmul_diff][dbg] case={} dev_strideB w=[{},{},{},{}] x=[{},{},{},{}] d=[{},{},{},{}]",
                tc.name,
                wst[0], wst[1], wst[2], wst[3],
                xst[0], xst[1], xst[2], xst[3],
                dst[0], dst[1], dst[2], dst[3]
            );
        }
    }

    backend.copy(&w_dev, &w_cpu);
    backend.copy(&x_dev, &x_cpu);
    backend.matmul(&d_dev, &w_dev, &x_dev);

    std::vector<float> d_ocl_storage;
    Tensor d_ocl_cpu = make_cpu_tensor_f32(d_shape, d_ocl_storage);
    backend.copy(&d_ocl_cpu, &d_dev);

    std::vector<float> d_ref_storage;
    Tensor d_ref_cpu = make_cpu_tensor_f32(d_shape, d_ref_storage);
    run_ggml_matmul_ref(cfg, hp, &w_cpu, &x_cpu, &d_ref_cpu);

    DiffStats st = calc_stats(d_ref_storage, d_ocl_storage);
    size_t bad_i = 0;
    float bad_diff = 0.0f;
    const bool close_ok = allclose(d_ocl_storage, d_ref_storage, tc.th.atol, tc.th.rtol, &bad_i, &bad_diff);
    const bool cosine_ok = st.cosine >= tc.th.cosine_min;

    POWERSERVE_LOG_INFO("[matmul_diff] case={} max_abs={:.6g} mean_abs={:.6g} cosine={:.8f} argmax_ref={} argmax_ocl={}",
                        tc.name, st.max_abs, st.mean_abs, st.cosine, st.argmax_ref, st.argmax_test);

    if (!close_ok || !cosine_ok) {
        POWERSERVE_LOG_ERROR("[matmul_diff] FAIL case={} close_ok={} cosine_ok={} (atol={} rtol={} cosine_min={})",
                             tc.name, (int)close_ok, (int)cosine_ok, tc.th.atol, tc.th.rtol, tc.th.cosine_min);
        if (!close_ok && bad_i < d_ref_storage.size()) {
            POWERSERVE_LOG_ERROR("[matmul_diff] first_bad_idx={} ref={} ocl={} diff={}",
                                 bad_i, d_ref_storage[bad_i], d_ocl_storage[bad_i], bad_diff);
        }

        const size_t head = std::min<size_t>(8, d_ref_storage.size());
        std::printf("[matmul_diff] ref head: ");
        for (size_t i = 0; i < head; ++i) std::printf("%.6f ", (double)d_ref_storage[i]);
        std::printf("\n[matmul_diff] ocl head: ");
        for (size_t i = 0; i < head; ++i) std::printf("%.6f ", (double)d_ocl_storage[i]);
        std::printf("\n");
        return false;
    }

    return true;
}

static bool run_matmul_diff_suite() {
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 256;
    cfg.vocab_size = 256;
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.n_kv_heads = 1;
    cfg.kv_dim     = 256;
    cfg.head_size  = 64;
    cfg.seq_len    = 16;

    HyperParams hp{};
    hp.n_threads = 1;

    const std::vector<TestCase> cases = {
        {"fp16_m1",  DataType::FP16,      256, 320, 1,  1u, {2e-2f, 2e-2f, 0.9990f}},
        {"fp16_m8",  DataType::FP16,      256, 320, 8,  2u, {2e-2f, 2e-2f, 0.9990f}},
        {"fp32_m1",  DataType::FP32,      256, 320, 1,  3u, {2e-3f, 2e-3f, 0.9999f}},
        {"fp32_m8",  DataType::FP32,      256, 320, 8,  4u, {2e-3f, 2e-3f, 0.9999f}},
        // Tighten q8 thresholds to better catch early matmul drift.
        {"q8_0_m1",  DataType::GGML_Q8_0, 256, 320, 1,  5u, {2e-2f, 2e-2f, 0.9990f}},
        {"q8_0_m8",  DataType::GGML_Q8_0, 256, 320, 8,  6u, {2e-2f, 2e-2f, 0.9990f}},
        // Reproduce layer-diff op#2-like geometry (Q8_0 * FP32, M>=32 path).
        {"q8_0_k896_n896_m1",  DataType::GGML_Q8_0, 896, 896, 1,  7u, {2e-2f, 2e-2f, 0.9990f}},
        {"q8_0_k896_n896_m77", DataType::GGML_Q8_0, 896, 896, 77, 8u, {2e-2f, 2e-2f, 0.9990f}},
        {"q8_0_k896_n896_m78", DataType::GGML_Q8_0, 896, 896, 78, 9u, {2e-2f, 2e-2f, 0.9990f}},
    };

    bool ok = true;
    for (const auto &tc : cases) {
        ok = run_case(cfg, hp, tc) && ok;
    }
    return ok;
}

} // namespace powerserve::opencl

int main() {
    const bool ok = powerserve::opencl::run_matmul_diff_suite();
    if (!ok) {
        return 1;
    }
    POWERSERVE_LOG_INFO("[matmul_diff] PASS");
    return 0;
}

