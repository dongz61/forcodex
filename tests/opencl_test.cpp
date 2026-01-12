// src/backend/opencl/opencl_smoke_test.cpp
#include "backend/opencl/opencl_backend.hpp"
#include "backend/ggml/ggml_wrapper.cpp"
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

static inline void ref_rmsnorm_f32(
    const std::vector<float> &x,         // [rows*hidden]
    const std::vector<float> &w,         // [hidden]
    int hidden,
    int rows,
    float eps,
    std::vector<float> &out              // [rows*hidden]
) {
    out.resize((size_t)hidden * (size_t)rows);
    const float inv_hidden = 1.0f / (float)hidden;

    for (int r = 0; r < rows; ++r) {
        const float *xr = x.data() + (size_t)r * (size_t)hidden;
        float *yr = out.data() + (size_t)r * (size_t)hidden;

        double sumsq = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const float v = xr[i];
            sumsq += (double)v * (double)v;
        }
        const float mean = (float)(sumsq * inv_hidden);
        const float scale = 1.0f / std::sqrt(mean + eps);

        for (int i = 0; i < hidden; ++i) {
            yr[i] = xr[i] * scale * w[(size_t)i];
        }
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

bool run_opencl_backend_rope_vs_ggml_test() {
    POWERSERVE_LOG_INFO("OpenCL rope-vs-ggml test: start");

    // ---- minimal config just to satisfy backend init ----
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 16;  // hidden dimension (ne0)
    cfg.vocab_size = 8;
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.n_kv_heads = 1;
    cfg.kv_dim     = 16;
    cfg.head_size  = 16;
    cfg.seq_len    = 16;

    // Setup RoPE config
    cfg.rope_config.n_dims       = 16;        // rotate all dims for this test
    cfg.rope_config.n_ctx_orig   = 2048;
    cfg.rope_config.freq_base    = 10000.0f;
    cfg.rope_config.freq_scale   = 1.0f;
    cfg.rope_config.ext_factor   = 0.0f;      // keep it simple first
    cfg.rope_config.attn_factor  = 1.0f;
    cfg.rope_config.beta_fast    = 32.0f;
    cfg.rope_config.beta_slow    = 0.0f;

    // 先测 norm (rope_type=0)，再测 neox (rope_type=2)
    // 这里假设 GGML_ROPE_TYPE_NEOX == 2
    const int rope_types_to_test[2] = {0, 2};

    HyperParams hp{};
    hp.n_threads = 1;

    // ---- init backends ----
    OpenCLBackend ocl(cfg, hp);
    if (!ocl.initialize()) {
        POWERSERVE_LOG_ERROR("OpenCL rope-vs-ggml test: OpenCL backend.initialize failed");
        return false;
    }

    powerserve::ggml::GGMLBackend ggml(cfg, hp);

    // ====== STEP: make GGMLBackend usable standalone (no plan()) ======
    // GGMLBackend::rope() will call m_thread_pool->run(...) :contentReference[oaicite:5]{index=5}
    // but m_thread_pool is nullptr right after construction :contentReference[oaicite:6]{index=6},
    // and wdata/wsize are also empty until setup_work_data() :contentReference[oaicite:7]{index=7}.
    ggml.setup_threadpool();
    // For ROPE, plan() uses sizeof(float) * dst_dim * n_threads as the base work size :contentReference[oaicite:8]{index=8}
    ggml.setup_work_data(sizeof(float) * (size_t)cfg.dim * (size_t)hp.n_threads);
    // ================================================================

    // ---- build test tensor: shape {dim, batch, 1, 1} ----
    const int D = (int)cfg.dim;
    const int T = 3;        // tokens = pos.size()
    const int H = 1;        // heads
    const int B = 1;        // batch (bring-up，先固定 1)

    Shape shape{};
    shape[0] = D;
    shape[1] = H;
    shape[2] = T;
    shape[3] = B;

    // src_storage: size = D * H * T * B = D * T
    std::vector<float> src_storage;
    Tensor src_cpu = make_cpu_tensor_f32(shape, src_storage);

    // 填数据：按 token 维写，每个 token 一行 D
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < D; ++i) {
            src_storage[(size_t)t * (size_t)D + (size_t)i] =
                0.01f * (float)t + 0.001f * (float)i - 0.02f;
        }
    }

    // positions
    std::vector<int> pos = {0, 5, 17};

    // outputs for ggml/opencl
    std::vector<float> out_ggml_storage;
    Tensor out_ggml_cpu = make_cpu_tensor_f32(shape, out_ggml_storage);

    std::vector<float> out_ocl_storage;
    Tensor out_ocl_cpu  = make_cpu_tensor_f32(shape, out_ocl_storage);

    // OpenCL tensors
    Tensor src_dev = make_opencl_tensor_f32(ocl, shape);
    Tensor out_dev = make_opencl_tensor_f32(ocl, shape);

    // Upload src to device
    ocl.copy(&src_dev, &src_cpu);

    // ---- run two rope_type variants ----
    for (int t = 0; t < 2; ++t) {
        cfg.rope_config.rope_type = rope_types_to_test[t];

        // (1) GGML reference: ggml backend rope writes into CPU tensor
        // NOTE: ggml backend expects dst/src on CPU buffers
        Tensor ggml_in  = src_cpu;         // CPU input
        Tensor ggml_out = out_ggml_cpu;    // CPU output
        ggml.rope(&ggml_out, &ggml_in, pos, cfg.rope_config);

        // (2) OpenCL backend rope: uses CPU fallback but takes dev input/out
        ocl.rope(&out_dev, &src_dev, pos, cfg.rope_config);

        // download OpenCL output
        ocl.copy(&out_ocl_cpu, &out_dev);

        // compare
        const float atol = 1e-4f;
        const float rtol = 1e-4f;
        size_t bad_i = 0;
        float bad_diff = 0.f;

        bool ok = allclose(out_ocl_storage, out_ggml_storage, atol, rtol, &bad_i, &bad_diff);
        if (!ok) {
            POWERSERVE_LOG_ERROR("OpenCL rope-vs-ggml test: FAILED (rope_type mismatch)");
            printf("rope_type=%d bad_i=%zu ocl=%f ggml=%f diff=%f\n",
                   cfg.rope_config.rope_type,
                   bad_i,
                   (double)out_ocl_storage[bad_i],
                   (double)out_ggml_storage[bad_i],
                   (double)bad_diff);

            // print head for quick diagnosis
            printf("OCL out head:  ");
            for (size_t i = 0; i < std::min<size_t>(8, out_ocl_storage.size()); ++i) {
                printf("%f ", (double)out_ocl_storage[i]);
            }
            printf("\nGGML out head: ");
            for (size_t i = 0; i < std::min<size_t>(8, out_ggml_storage.size()); ++i) {
                printf("%f ", (double)out_ggml_storage[i]);
            }
            printf("\n");
            return false;
        }

        POWERSERVE_LOG_INFO("OpenCL rope-vs-ggml test: PASS (rope_type={})", cfg.rope_config.rope_type);
    }

    POWERSERVE_LOG_INFO("OpenCL rope-vs-ggml test: PASS");
    return true;
}


bool run_opencl_backend_rmsnorm_test() {
    POWERSERVE_LOG_INFO("OpenCL rmsnorm test: start");

    // Minimal config
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 16;   // hidden
    cfg.vocab_size = 8;
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.n_kv_heads = 1;
    cfg.kv_dim     = 16;
    cfg.head_size  = 16;
    cfg.seq_len    = 16;

    HyperParams hp{};
    hp.n_threads = 1;

    OpenCLBackend backend(cfg, hp);
    if (!backend.initialize()) {
        POWERSERVE_LOG_ERROR("OpenCL rmsnorm test: backend.initialize failed");
        return false;
    }

    const int H = (int)cfg.dim;
    const int M = 3;      // rows (batch-like)
    const float eps = 1e-5f;

    // x_dev: shape {H, M, 1, 1}
    Shape x_shape{};
    x_shape[0] = H;
    x_shape[1] = M;
    x_shape[2] = 1;
    x_shape[3] = 1;

    // host x
    std::vector<float> x_host_storage;
    Tensor x_cpu = make_cpu_tensor_f32(x_shape, x_host_storage);
    for (int r = 0; r < M; ++r) {
        for (int i = 0; i < H; ++i) {
            // deterministic but nontrivial
            x_host_storage[(size_t)r * (size_t)H + (size_t)i] =
                0.01f * (float)r + 0.001f * (float)i - 0.002f;
        }
    }

    Tensor x_dev = make_opencl_tensor_f32(backend, x_shape);
    backend.copy(&x_dev, &x_cpu); // H2D

    // weight_cpu: shape {H,1,1,1}
    Shape w_shape{};
    w_shape[0] = H;
    w_shape[1] = 1;
    w_shape[2] = 1;
    w_shape[3] = 1;

    std::vector<float> w_storage;
    Tensor w_cpu = make_cpu_tensor_f32(w_shape, w_storage);
    for (int i = 0; i < H; ++i) {
        // gamma: vary a bit
        w_storage[(size_t)i] = 1.0f + 0.01f * (float)i;
    }

    // out_dev: shape {H,M,1,1}
    Tensor out_dev = make_opencl_tensor_f32(backend, x_shape);

    // ---- call backend rmsnorm (CPU fallback) ----
    backend.rmsnorm(&out_dev, &x_dev, &w_cpu, eps);

    // ---- D2H out ----
    std::vector<float> out_host_storage;
    Tensor out_cpu = make_cpu_tensor_f32(x_shape, out_host_storage);
    backend.copy(&out_cpu, &out_dev); // D2H

    // ---- CPU reference ----
    std::vector<float> ref_out;
    ref_rmsnorm_f32(x_host_storage, w_storage, H, M, eps, ref_out);

    // ---- compare ----
    const float atol = 1e-4f;
    const float rtol = 1e-4f;
    size_t bad_i = 0;
    float bad_diff = 0.f;
    bool ok = allclose(out_host_storage, ref_out, atol, rtol, &bad_i, &bad_diff);

    if (!ok) {
        POWERSERVE_LOG_ERROR("OpenCL rmsnorm test: FAILED (mismatch)");
        printf("bad_i=%zu gpu=%f cpu=%f diff=%f\n",
               bad_i,
               (double)out_host_storage[bad_i],
               (double)ref_out[bad_i],
               (double)bad_diff);

        printf("GPU out head: ");
        for (size_t i = 0; i < std::min<size_t>(8, out_host_storage.size()); ++i) {
            printf("%f ", (double)out_host_storage[i]);
        }
        printf("\nCPU out head: ");
        for (size_t i = 0; i < std::min<size_t>(8, ref_out.size()); ++i) {
            printf("%f ", (double)ref_out[i]);
        }
        printf("\n");
        return false;
    }

    POWERSERVE_LOG_INFO("OpenCL rmsnorm test: PASS");
    return true;
}
bool run_opencl_backend_softmax_ext_vs_ggml_test() {
    POWERSERVE_LOG_INFO("OpenCL softmax_ext-vs-ggml test: start");

    // ---- minimal config ----
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 16;
    cfg.vocab_size = 8;
    cfg.n_layers   = 1;

    // IMPORTANT: ne02 is used as n_head in ggml slope computation.
    // We'll set cfg.n_heads = ne02 for clarity.
    cfg.n_heads    = 8;
    cfg.n_kv_heads = 8;
    cfg.kv_dim     = 16;
    cfg.head_size  = 16;
    cfg.seq_len    = 16;

    HyperParams hp{};
    hp.n_threads = 1;

    // ---- init backends ----
    OpenCLBackend ocl(cfg, hp);
    if (!ocl.initialize()) {
        POWERSERVE_LOG_ERROR("OpenCL softmax_ext-vs-ggml test: OpenCL backend.initialize failed");
        return false;
    }

    powerserve::ggml::GGMLBackend ggml(cfg, hp);
    // softmax_ext in GGMLBackend uses threadpool similarly to other ops in your wrapper;
    // keep it safe for standalone test.
    ggml.setup_threadpool();
    // work size: allocate at least (nc + CACHE_LINE_SIZE_F32)*n_threads floats like ggml uses
    // Here we use nc=ne00, and give it plenty.
    const size_t ne00 = 64;
    ggml.setup_work_data(sizeof(float) * (ne00 + 64) * (size_t)hp.n_threads);

    // ---- shapes ----
    // x shape: {ne00, ne01, ne02, ne03}
    // ggml uses:
    //   nc = ne00
    //   row count nr = ne01*ne02*ne03
    const int NE00 = 64;   // nc (kv length)
    const int NE01 = 8;    // "n_q" like dimension; also controls mask broadcast row via (i1 % NE01)
    const int NE02 = 8;    // n_head (critical for ALiBi slope)
    const int NE03 = 2;    // batch-ish

    Shape x_shape{};
    x_shape[0] = NE00;
    x_shape[1] = NE01;
    x_shape[2] = NE02;
    x_shape[3] = NE03;

    // mask shape MUST be {NE00, NE01, 1, 1} to match ggml mp=(i1%ne01)*ne00
    Shape m_shape{};
    m_shape[0] = NE00;
    m_shape[1] = NE01;
    m_shape[2] = 1;
    m_shape[3] = 1;

    // ---- host tensors (CPU) ----
    std::vector<float> x_storage;
    Tensor x_cpu = make_cpu_tensor_f32(x_shape, x_storage);

    std::vector<float> m_storage;
    Tensor m_cpu = make_cpu_tensor_f32(m_shape, m_storage);

    // Fill x deterministically (avoid RNG)
    // Layout is contiguous row-major by your make_cpu_tensor_f32: index = i0 + ne00*(i1 + ne01*(i2 + ne02*i3))
    for (int b = 0; b < NE03; ++b) {
        for (int h = 0; h < NE02; ++h) {
            for (int q = 0; q < NE01; ++q) {
                for (int kv = 0; kv < NE00; ++kv) {
                    const size_t idx =
                        (size_t)kv +
                        (size_t)NE00 * ((size_t)q + (size_t)NE01 * ((size_t)h + (size_t)NE02 * (size_t)b));
                    x_storage[idx] = 0.001f * (float)kv + 0.01f * (float)q - 0.02f * (float)h + 0.005f * (float)b;
                }
            }
        }
    }

    // Fill mask as "causal-like" per q row: mask[kv,q] = (kv<=q)?0:-inf
    // NOTE: mask only has {kv,q}, broadcast over head/batch in ggml.
    for (int q = 0; q < NE01; ++q) {
        for (int kv = 0; kv < NE00; ++kv) {
            m_storage[(size_t)kv + (size_t)q * (size_t)NE00] = (kv <= q) ? 0.0f : -INFINITY;
        }
    }

    // ---- reference output (GGML) ----
    std::vector<float> out_ggml_storage;
    Tensor out_ggml_cpu = make_cpu_tensor_f32(x_shape, out_ggml_storage);

    // ---- opencl output (device -> host) ----
    std::vector<float> out_ocl_storage;
    Tensor out_ocl_cpu = make_cpu_tensor_f32(x_shape, out_ocl_storage);

    // ---- device tensors ----
    Tensor x_dev   = make_opencl_tensor_f32(ocl, x_shape);
    Tensor m_dev   = make_opencl_tensor_f32(ocl, m_shape);
    Tensor out_dev = make_opencl_tensor_f32(ocl, x_shape);

    // Upload
    ocl.copy(&x_dev, &x_cpu);
    ocl.copy(&m_dev, &m_cpu);

    // ---- run test cases ----
    struct Case { float scale; float max_bias; };
    const Case cases[] = {
        {1.0f, 0.0f},   // no ALiBi slope
        {0.5f, 8.0f},   // enable ALiBi slope (max_bias > 0)
    };

    for (const auto &tc : cases) {
        // GGML reference (CPU buffers)
        ggml.softmax_ext(&out_ggml_cpu, &x_cpu, &m_cpu, tc.scale, tc.max_bias);

        // OpenCL backend (will run your OpenCLBackend::softmax_ext CPU fallback, but taking dev tensors)
        ocl.softmax_ext(&out_dev, &x_dev, &m_dev, tc.scale, tc.max_bias);

        // Download
        ocl.copy(&out_ocl_cpu, &out_dev);

        // Compare
        const float atol = 1e-5f;
        const float rtol = 1e-5f;
        size_t bad_i = 0;
        float bad_diff = 0.f;

        bool ok = allclose(out_ocl_storage, out_ggml_storage, atol, rtol, &bad_i, &bad_diff);
        if (!ok) {
            POWERSERVE_LOG_ERROR("OpenCL softmax_ext-vs-ggml test: FAILED (mismatch)");
            printf("scale=%f max_bias=%f bad_i=%zu ocl=%f ggml=%f diff=%f\n",
                   (double)tc.scale, (double)tc.max_bias,
                   bad_i,
                   (double)out_ocl_storage[bad_i],
                   (double)out_ggml_storage[bad_i],
                   (double)bad_diff);

            // print a small head slice
            printf("OCL out head:  ");
            for (size_t i = 0; i < std::min<size_t>(8, out_ocl_storage.size()); ++i) {
                printf("%f ", (double)out_ocl_storage[i]);
            }
            printf("\nGGML out head: ");
            for (size_t i = 0; i < std::min<size_t>(8, out_ggml_storage.size()); ++i) {
                printf("%f ", (double)out_ggml_storage[i]);
            }
            printf("\n");
            return false;
        }

        POWERSERVE_LOG_INFO("OpenCL softmax_ext-vs-ggml test: PASS (scale={} max_bias={})",
                            tc.scale, tc.max_bias);
    }

    POWERSERVE_LOG_INFO("OpenCL softmax_ext-vs-ggml test: PASS");
    return true;
}
// ---------- helper: CPU f32 tensor with custom (possibly non-contig) stride ----------
static inline Tensor make_cpu_tensor_f32_strided(
    const Shape &shape,
    const Stride &stride_bytes,
    std::vector<uint8_t> &storage_bytes
) {
    Tensor t(DataType::FP32, shape);

    // allocate enough backing bytes for the last element reachable by strides
    // last_offset = (ne0-1)*st0 + (ne1-1)*st1 + (ne2-1)*st2 + (ne3-1)*st3
    const size_t ne0 = (size_t)shape[0];
    const size_t ne1 = (size_t)shape[1];
    const size_t ne2 = (size_t)shape[2];
    const size_t ne3 = (size_t)shape[3];

    const size_t last_off =
        (ne0 ? (ne0 - 1) * (size_t)stride_bytes[0] : 0) +
        (ne1 ? (ne1 - 1) * (size_t)stride_bytes[1] : 0) +
        (ne2 ? (ne2 - 1) * (size_t)stride_bytes[2] : 0) +
        (ne3 ? (ne3 - 1) * (size_t)stride_bytes[3] : 0);

    const size_t need_bytes = last_off + sizeof(float);
    storage_bytes.assign(need_bytes, 0);

    t.m_data = std::make_shared<powerserve::CPUBuffer>(stride_bytes, storage_bytes.data());
    return t;
}

// ---------- test: quant(Q8_0) weight on CPU + non-contig activation on OpenCL ----------
bool run_opencl_backend_matmul_quant_noncontigA_test() {
    POWERSERVE_LOG_INFO("OpenCL matmul quant(Q8_0) + non-contig A test: start");

    // Minimal config (only used for backend init; matmul itself uses tensor shapes)
    ModelConfig::LLMConfig cfg{};
    cfg.dim        = 64;   // K
    cfg.vocab_size = 64;   // N (not strictly needed here, but keep sane)
    cfg.n_layers   = 1;
    cfg.n_heads    = 1;
    cfg.n_kv_heads = 1;
    cfg.kv_dim     = 64;
    cfg.head_size  = 64;
    cfg.seq_len    = 16;

    HyperParams hp{};
    hp.n_threads = 1;

    OpenCLBackend backend(cfg, hp);
    if (!backend.initialize()) {
        POWERSERVE_LOG_ERROR("OpenCL matmul quant+noncontigA test: backend.initialize failed");
        return false;
    }

    const int K = 64;
    const int N = 64;
    const int M = 4;

    // -------------------------
    // Build B_fp32 on CPU with ggml-friendly layout:
    // shape {K, N, 1, 1}, row-major in i0 (K contiguous), row = output neuron n
    // idx = k + K*n
    // -------------------------
    Shape B_shape{};
    B_shape[0] = K;
    B_shape[1] = N;
    B_shape[2] = 1;
    B_shape[3] = 1;

    std::vector<float> B_f32_storage;
    Tensor B_f32_cpu = make_cpu_tensor_f32(B_shape, B_f32_storage);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            B_f32_storage[(size_t)k + (size_t)K * (size_t)n] =
                0.01f * (float)n - 0.0007f * (float)k;
        }
    }

    // -------------------------
    // Quantize B to Q8_0 (CPUBuffer)
    // Use ggml_row_size() and quantize_row_q8_0() so stride semantics match ggml.
    // -------------------------
    const ggml_type qtype = GGML_TYPE_Q8_0;
    const size_t row_size = ggml_row_size(qtype, K);     // bytes per row (one output neuron)
    const size_t q_bytes  = row_size * (size_t)N;

    std::vector<uint8_t> B_q_storage(q_bytes, 0);

    Stride B_q_stride{};
    B_q_stride[0] = (int)ggml_type_size(qtype);          // e.g. sizeof(block_q8_0)
    B_q_stride[1] = (int)row_size;
    B_q_stride[2] = B_q_stride[1] * B_shape[1];
    B_q_stride[3] = B_q_stride[2] * B_shape[2];

    Tensor B_q_cpu(DataType::GGML_Q8_0, B_shape);
    B_q_cpu.m_data = std::make_shared<powerserve::CPUBuffer>(B_q_stride, B_q_storage.data());

    for (int n = 0; n < N; ++n) {
        const float *src_row = B_f32_storage.data() + (size_t)K * (size_t)n;
        void *dst_row = (void *)(B_q_storage.data() + row_size * (size_t)n);
        quantize_row_q8_0(src_row, dst_row, K);
    }

    // -------------------------
    // Build A on CPU but with intentional padding in stride1 (non-contig)
    // Logical shape: {K, M, 1, 1}
    // Backing layout: row stride = K*sizeof(float)*2 (so it is non-contig)
    // -------------------------
    Shape A_shape{};
    A_shape[0] = K;
    A_shape[1] = M;
    A_shape[2] = 1;
    A_shape[3] = 1;

    const int pad_factor = 2;
    Stride A_pad_stride{};
    A_pad_stride[0] = (int)sizeof(float);
    A_pad_stride[1] = (int)((size_t)K * sizeof(float) * (size_t)pad_factor);
    A_pad_stride[2] = A_pad_stride[1] * A_shape[1];
    A_pad_stride[3] = A_pad_stride[2] * A_shape[2];

    std::vector<uint8_t> A_pad_bytes;
    Tensor A_cpu_pad = make_cpu_tensor_f32_strided(A_shape, A_pad_stride, A_pad_bytes);

    // Also keep a contiguous logical copy for CPU reference
    std::vector<float> A_logical((size_t)K * (size_t)M, 0.f);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float v = 0.02f * (float)m + 0.001f * (float)k - 0.03f;
            A_logical[(size_t)k + (size_t)K * (size_t)m] = v;

            // write into padded CPU tensor memory using its stride
            uint8_t *base = (uint8_t *)A_pad_bytes.data();
            float *ptr = (float *)(base + (size_t)m * (size_t)A_pad_stride[1] + (size_t)k * (size_t)A_pad_stride[0]);
            *ptr = v;
        }
    }

    // -------------------------
    // Create a device buffer big enough to honor the padded stride.
    // Allocate as {K, M*pad_factor, 1, 1}, then "view" it as {K, M, 1, 1} by overriding stride.
    // -------------------------
    Shape A_alloc_shape{};
    A_alloc_shape[0] = K;
    A_alloc_shape[1] = M * pad_factor;
    A_alloc_shape[2] = 1;
    A_alloc_shape[3] = 1;

    Tensor A_dev_alloc = make_opencl_tensor_f32(backend, A_alloc_shape);

    Tensor A_dev(DataType::FP32, A_shape);
    A_dev.m_data = A_dev_alloc.m_data;

    // Override device stride to make it non-contig for the logical {K,M} view
    {
        auto &buf = A_dev.get<OpenCLBuffer>();
        buf.m_stride = A_pad_stride;
    }

    // Upload A (H2D) — this MUST exercise your non-contig H2D copy path
    backend.copy(&A_dev, &A_cpu_pad);

    // -------------------------
    // Matmul on backend: C = A * B
    // A: OpenCL FP32 non-contig
    // B: CPU Q8_0
    // C: OpenCL FP32 contig
    // -------------------------
    Shape C_shape{};
    C_shape[0] = N;
    C_shape[1] = M;
    C_shape[2] = 1;
    C_shape[3] = 1;

    Tensor C_dev = make_opencl_tensor_f32(backend, C_shape);
    backend.matmul(&C_dev, &B_q_cpu, &A_dev);

    // D2H result
    std::vector<float> C_host_storage;
    Tensor C_cpu = make_cpu_tensor_f32(C_shape, C_host_storage);
    backend.copy(&C_cpu, &C_dev);

    // -------------------------
    // CPU reference (float): C[n,m] = sum_k A[k,m] * B[k,n]
    // where B is in ggml-friendly layout idx = k + K*n
    // -------------------------
    std::vector<float> C_ref((size_t)N * (size_t)M, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const float a = A_logical[(size_t)k + (size_t)K * (size_t)m];
                const float b = B_f32_storage[(size_t)k + (size_t)K * (size_t)n];
                acc += (double)a * (double)b;
            }
            C_ref[(size_t)n + (size_t)N * (size_t)m] = (float)acc;
        }
    }

    // Compare
    const float atol = 5e-3f;   // quant introduces error; keep reasonable
    const float rtol = 5e-3f;
    size_t bad_i = 0;
    float bad_diff = 0.f;

    bool ok = allclose(C_host_storage, C_ref, atol, rtol, &bad_i, &bad_diff);
    if (!ok) {
        POWERSERVE_LOG_ERROR("OpenCL matmul quant+noncontigA test: FAILED (mismatch)");
        printf("bad_i=%zu ocl=%f ref=%f diff=%f\n",
               bad_i,
               (double)C_host_storage[bad_i],
               (double)C_ref[bad_i],
               (double)bad_diff);

        printf("OCL head: ");
        for (size_t i = 0; i < std::min<size_t>(8, C_host_storage.size()); ++i) {
            printf("%f ", (double)C_host_storage[i]);
        }
        printf("\nREF head: ");
        for (size_t i = 0; i < std::min<size_t>(8, C_ref.size()); ++i) {
            printf("%f ", (double)C_ref[i]);
        }
        printf("\n");
        return false;
    }

    POWERSERVE_LOG_INFO("OpenCL matmul quant(Q8_0) + non-contig A test: PASS");
    return true;
}



} // namespace powerserve::opencl

int main() {
    // bool ok1 = powerserve::opencl::run_opencl_backend_smoke_test();
    // bool ok2 = powerserve::opencl::run_opencl_backend_rope_vs_ggml_test();
    // bool ok3 = powerserve::opencl::run_opencl_backend_softmax_ext_vs_ggml_test();
    bool ok4 = powerserve::opencl::run_opencl_backend_matmul_quant_noncontigA_test();
    // return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}
