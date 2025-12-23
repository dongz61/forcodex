// src/backend/opencl/opencl_smoke_test.cpp
#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace powerserve::opencl {

// ---------------- tiny helpers ----------------

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
    storage.resize(t.n_elements(), 0.f);
    auto stride = make_contig_stride_bytes(shape, sizeof(float));
    t.m_data = std::make_shared<powerserve::CPUBuffer>(stride, storage.data());
    return t;
}

static inline Tensor make_opencl_tensor_f32(OpenCLBackend &b, const Shape &shape) {
    Tensor t(DataType::FP32, shape);
    // OpenCLBuffer is stored as Tensor::m_data (shared_ptr<BufferBase>)
    t.m_data = b.create_buffer(shape, DataType::FP32);
    return t;
}

static inline bool allclose(const std::vector<float> &a, const std::vector<float> &b, float atol, float rtol,
                            size_t *first_bad_idx = nullptr, float *diff_out = nullptr) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol  = atol + rtol * std::fabs(b[i]);
        if (!(diff <= tol)) {
            if (first_bad_idx) *first_bad_idx = i;
            if (diff_out) *diff_out = diff;
            return false;
        }
    }
    return true;
}

// CPU reference for logits when:
// - emb is [K] vector
// - B is stored as [K,N] row-major (k-major then v), index = k*N + v
// - bias is [N]
static inline void ref_logits_B_KN(
    const std::vector<float> &emb_K,     // [K]
    const std::vector<float> &B_KN,      // [K*N] row-major in K,N
    const std::vector<float> &bias_N,    // [N]
    int K, int N,
    std::vector<float> &out_N            // [N]
) {
    out_N.assign((size_t)N, 0.f);
    for (int v = 0; v < N; ++v) {
        float acc = bias_N.empty() ? 0.f : bias_N[(size_t)v];
        for (int k = 0; k < K; ++k) {
            acc += emb_K[(size_t)k] * B_KN[(size_t)k * (size_t)N + (size_t)v];
        }
        out_N[(size_t)v] = acc;
    }
}

// ---------------- the smoke test ----------------

bool run_opencl_backend_smoke_test() {
    POWERSERVE_LOG_INFO("OpenCL smoke test: start");

    // ---- minimal config just to satisfy backend init / kv init path ----
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 4;    // K
    cfg.vocab_size = 8;    // N
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.n_kv_heads = 1;
    cfg.kv_dim     = 4;
    cfg.head_size  = 4;
    cfg.seq_len    = 16;

    HyperParams hp{};
    hp.n_threads = 1;

    OpenCLBackend backend(cfg, hp);
    if (!backend.initialize()) {
        POWERSERVE_LOG_ERROR("OpenCL smoke test: backend.initialize failed");
        return false;
    }

    const int K = (int)cfg.dim;
    const int N = (int)cfg.vocab_size;
    const int M = 1; // batch=1 for this smoke

    // ---- 1) Embedding weight on CPU: shape {dim, vocab, 1, 1} ----
    Shape emb_w_shape{};
    emb_w_shape[0] = K;
    emb_w_shape[1] = N;
    emb_w_shape[2] = 1;
    emb_w_shape[3] = 1;

    std::vector<float> emb_w_storage;
    Tensor emb_weight_cpu = make_cpu_tensor_f32(emb_w_shape, emb_w_storage);

    // Fill embedding table deterministically.
    // IMPORTANT: get_embedding uses CPUBuffer stride[1] * token to locate row.
    // For our contiguous CPU tensor with shape {K,N}, stride[1]=K*sizeof(float),
    // so "row token" corresponds to contiguous K floats: emb_w[token*K + k].
    for (int token = 0; token < N; ++token) {
        for (int k = 0; k < K; ++k) {
            emb_w_storage[(size_t)token * (size_t)K + (size_t)k] =
                0.01f * (float)token + 0.001f * (float)k;
        }
    }

    // tokens (batch=1)
    std::vector<int> tokens;
    tokens.push_back(3);

    // dst embedding on OpenCL: shape {dim, batch, 1, 1} => {K,M,1,1}
    Shape emb_out_shape{};
    emb_out_shape[0] = K;
    emb_out_shape[1] = (int)tokens.size();
    emb_out_shape[2] = 1;
    emb_out_shape[3] = 1;

    Tensor emb_out_dev = make_opencl_tensor_f32(backend, emb_out_shape);

    backend.get_embedding(&emb_out_dev, &emb_weight_cpu, tokens);

    // ---- 2) lm_head B on CPU then H2D: shape {N,K,1,1} but stored as [K,N] row-major ----
    // This is the CRITICAL layout match to matmul_minimal.
    Shape B_shape{};
    B_shape[0] = N;   // N
    B_shape[1] = K;   // K
    B_shape[2] = 1;
    B_shape[3] = 1;

    std::vector<float> B_cpu_storage;
    Tensor B_cpu = make_cpu_tensor_f32(B_shape, B_cpu_storage);

    // Fill B as [K,N] row-major: index = k*N + v
    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < N; ++v) {
            B_cpu_storage[(size_t)k * (size_t)N + (size_t)v] =
                0.02f * (float)v - 0.003f * (float)k;
        }
    }

    Tensor B_dev = make_opencl_tensor_f32(backend, B_shape);
    backend.copy(&B_dev, &B_cpu); // H2D

    // ---- 3) matmul: C = A * B => shape {N,M,1,1} ----
    Shape logits_shape{};
    logits_shape[0] = N;
    logits_shape[1] = M;
    logits_shape[2] = 1;
    logits_shape[3] = 1;

    Tensor logits_dev = make_opencl_tensor_f32(backend, logits_shape);
    backend.matmul(&logits_dev, &emb_out_dev, &B_dev);

    // ---- 4) add bias (same shape) ----
    Shape bias_shape = logits_shape;

    std::vector<float> bias_cpu_storage;
    Tensor bias_cpu = make_cpu_tensor_f32(bias_shape, bias_cpu_storage);
    for (int v = 0; v < N; ++v) {
        bias_cpu_storage[(size_t)v] = 0.1f - 0.01f * (float)v;
    }

    Tensor bias_dev = make_opencl_tensor_f32(backend, bias_shape);
    backend.copy(&bias_dev, &bias_cpu); // H2D

    Tensor logits_plus_bias_dev = make_opencl_tensor_f32(backend, logits_shape);
    backend.add(&logits_plus_bias_dev, &logits_dev, &bias_dev);

    // ---- 5) D2H logits ----
    std::vector<float> logits_host_storage;
    Tensor logits_host = make_cpu_tensor_f32(logits_shape, logits_host_storage);
    backend.copy(&logits_host, &logits_plus_bias_dev); // D2H

    // ---- 6) CPU reference ----
    std::vector<float> emb_ref((size_t)K, 0.f);
    {
        int token = tokens[0];
        const float *row = emb_w_storage.data() + (size_t)token * (size_t)K;
        std::copy(row, row + K, emb_ref.begin());
    }

    std::vector<float> bias_ref((size_t)N, 0.f);
    for (int v = 0; v < N; ++v) bias_ref[(size_t)v] = bias_cpu_storage[(size_t)v];

    std::vector<float> out_ref;
    ref_logits_B_KN(emb_ref, B_cpu_storage, bias_ref, K, N, out_ref);

    // ---- 7) Compare ----
    const float atol = 1e-4f;
    const float rtol = 1e-4f;
    size_t bad_i = 0;
    float bad_diff = 0.f;
    bool ok = allclose(logits_host_storage, out_ref, atol, rtol, &bad_i, &bad_diff);

    if (!ok) {
        POWERSERVE_LOG_ERROR("OpenCL smoke test: FAILED (logits mismatch)");
        // Avoid "{}" formatting; print a few values for quick diagnosis.
        // If you prefer, replace with printf.
        if (bad_i < logits_host_storage.size()) {
            POWERSERVE_LOG_ERROR("OpenCL smoke test: first mismatch at index");
            // simple debug prints
            // Note: log system may not support formatting reliably; use printf as fallback.
            printf("bad_i=%zu gpu=%f cpu=%f diff=%f\n",
                   bad_i,
                   (double)logits_host_storage[bad_i],
                   (double)out_ref[bad_i],
                   (double)bad_diff);
        }
        // Print first 8 entries
        printf("GPU logits: ");
        for (size_t i = 0; i < std::min<size_t>(8, logits_host_storage.size()); ++i) {
            printf("%f ", (double)logits_host_storage[i]);
        }
        printf("\nCPU logits: ");
        for (size_t i = 0; i < std::min<size_t>(8, out_ref.size()); ++i) {
            printf("%f ", (double)out_ref[i]);
        }
        printf("\n");
        return false;
    }

    POWERSERVE_LOG_INFO("OpenCL smoke test: PASS");
    return true;
}

} // namespace powerserve::opencl

int main() {
    bool ok = powerserve::opencl::run_opencl_backend_smoke_test();
    return ok ? 0 : 1;
}

