#include "backend/opencl/opencl_backend.hpp"
#include "backend/ggml/ggml_wrapper.cpp"
#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "ggml.h"

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
    // Quantize B to Q8_0 (CPUBuffer) -- ALIGNED storage
    // -------------------------
    const ggml_type qtype = GGML_TYPE_Q8_0;
    const int nb = K / 32;                 // blocks per row for Q8_0 (K must be multiple of 32)
    POWERSERVE_ASSERT(K % 32 == 0);

    std::vector<block_q8_0> B_q_blocks((size_t)nb * (size_t)N);

    // stride follows ggml nb[] convention: nb0=type_size, nb1=row_size
    const size_t row_size = (size_t)ggml_row_size(qtype, K);  // should be nb * sizeof(block_q8_0)
    POWERSERVE_ASSERT(row_size == (size_t)nb * sizeof(block_q8_0));

    Stride B_q_stride{};
    B_q_stride[0] = (int)ggml_type_size(qtype);          // 34
    B_q_stride[1] = (int)row_size;                       // 68 for K=64
    B_q_stride[2] = B_q_stride[1] * B_shape[1];
    B_q_stride[3] = B_q_stride[2] * B_shape[2];


    Tensor B_q_cpu(DataType::GGML_Q8_0, B_shape);
    B_q_cpu.m_data = std::make_shared<powerserve::CPUBuffer>(B_q_stride, B_q_blocks.data());

    for (int n = 0; n < N; ++n) {
        const float *src_row = B_f32_storage.data() + (size_t)K * (size_t)n;
        block_q8_0 *dst_row  = B_q_blocks.data() + (size_t)nb * (size_t)n;
        quantize_row_q8_0(src_row, (void *)dst_row, K);
    }


    // ---- SELF CHECK: dequant first row after quantize ----
    {
        std::vector<float> deq0((size_t)K, 0.f);
        const block_q8_0 *row0 = (const block_q8_0 *)(B_q_blocks.data()); // 第 0 行
        dequantize_row_q8_0(row0, deq0.data(), K);

        printf("[TEST] B_q row0 dequant head: ");
        for (int i = 0; i < 8; ++i) printf("%f ", (double)deq0[i]);
        printf("\n");

        // 再把 block0 的 d / qs 打出来
        float d0 = ggml_fp16_to_fp32(row0[0].d);
        uint16_t raw = *(const uint16_t *)&row0[0].d;
        printf("[TEST] B_q block0 raw_d=0x%04x\n", raw);

        printf("[TEST] B_q block0 d0=%f qs0..7=", (double)d0);
        for (int i = 0; i < 8; ++i) printf("%d ", (int)row0[0].qs[i]);
        printf("\n");
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
    // backend.matmul expects weight (B) first, activation (A) second
    backend.matmul(&C_dev, &B_q_cpu, &A_dev);

    // D2H result
    std::vector<float> C_host_storage;
    Tensor C_cpu = make_cpu_tensor_f32(C_shape, C_host_storage);
    backend.copy(&C_cpu, &C_dev);

    // -------------------------
    // GGML reference (quant Q8_0 * noncontig A):
    // C_ggml = matmul(B_q_cpu, A_cpu_pad)  -> same path as OpenCL uses logically
    // -------------------------
    std::vector<float> C_ggml_storage;
    Tensor C_ggml_cpu = make_cpu_tensor_f32(C_shape, C_ggml_storage);

    {
        powerserve::ggml::GGMLBackend ggml_be(cfg, hp);
        ggml_be.setup_threadpool();
        {
            const ggml_type vec_dot = ggml_be.get_vec_dot_type(&A_cpu_pad); // x = activation
            const ggml_type w_type  = powerserve::ggml::convert_datatype_to_ggml(B_q_cpu.m_dtype); // weight = B

            size_t work = 0;
            if (w_type != vec_dot) {
                work = (size_t)ggml_row_size(vec_dot, (int64_t)B_q_cpu.n_elements());
            }

            work = std::max(work, (size_t)(1 << 20)); // 1MB

            ggml_be.setup_work_data(work);
        }
        ggml_be.matmul(&C_ggml_cpu, &B_q_cpu, &A_cpu_pad);
    }

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
    const float atol = 5e-3f;
    const float rtol = 5e-3f;
    size_t bad_i = 0;
    float bad_diff = 0.f;

    // (1) OpenCL vs GGML (this is your alignment target)
    bool ok_ocl_vs_ggml = allclose(C_host_storage, C_ggml_storage, atol, rtol, &bad_i, &bad_diff);
    if (!ok_ocl_vs_ggml) {
        POWERSERVE_LOG_ERROR("OpenCL vs GGML (Q8_0 x noncontig A) mismatch");
        printf("bad_i=%zu ocl=%f ggml=%f diff=%f\n",
               bad_i,
               (double)C_host_storage[bad_i],
               (double)C_ggml_storage[bad_i],
               (double)bad_diff);

        printf("OCL head:  ");
        for (size_t i = 0; i < std::min<size_t>(8, C_host_storage.size()); ++i) printf("%f ", (double)C_host_storage[i]);
        printf("\nGGML head: ");
        for (size_t i = 0; i < std::min<size_t>(8, C_ggml_storage.size()); ++i) printf("%f ", (double)C_ggml_storage[i]);
        printf("\nREF head:  ");
        for (size_t i = 0; i < std::min<size_t>(8, C_ref.size()); ++i) printf("%f ", (double)C_ref[i]);
        printf("\n");

        // Debug: verify D2H pack for non-contig A
        std::vector<float> A_dev_host_storage;
        Tensor A_dev_host = make_cpu_tensor_f32(A_shape, A_dev_host_storage);
        backend.copy(&A_dev_host, &A_dev);
        printf("A_dev D2H head: ");
        for (size_t i = 0; i < std::min<size_t>(8, A_dev_host_storage.size()); ++i) printf("%f ", (double)A_dev_host_storage[i]);
        printf("\nA_logical head: ");
        for (size_t i = 0; i < std::min<size_t>(8, A_logical.size()); ++i) printf("%f ", (double)A_logical[i]);
        printf("\n");

        return false;
    }

    // (2) GGML vs CPU FP32 ref (quantization sanity)
    bool ok_ggml_vs_ref = allclose(C_ggml_storage, C_ref, atol, rtol, &bad_i, &bad_diff);
    if (!ok_ggml_vs_ref) {
        POWERSERVE_LOG_ERROR("GGML(Q8_0 x noncontig A) vs CPU FP32 ref mismatch (quant error too large?)");
        printf("bad_i=%zu ggml=%f ref=%f diff=%f\n",
               bad_i,
               (double)C_ggml_storage[bad_i],
               (double)C_ref[bad_i],
               (double)bad_diff);

        printf("GGML head: ");
        for (size_t i = 0; i < std::min<size_t>(8, C_ggml_storage.size()); ++i) printf("%f ", (double)C_ggml_storage[i]);
        printf("\nREF head:  ");
        for (size_t i = 0; i < std::min<size_t>(8, C_ref.size()); ++i) printf("%f ", (double)C_ref[i]);
        printf("\n");
        return false;
    }

    // (3) OpenCL vs CPU FP32 ref (total error)
    bool ok_ocl_vs_ref = allclose(C_host_storage, C_ref, atol, rtol, &bad_i, &bad_diff);
    if (!ok_ocl_vs_ref) {
        POWERSERVE_LOG_ERROR("OpenCL vs CPU FP32 ref mismatch (total error)");
        printf("bad_i=%zu ocl=%f ref=%f diff=%f\n",
               bad_i,
               (double)C_host_storage[bad_i],
               (double)C_ref[bad_i],
               (double)bad_diff);
        return false;
    }


    POWERSERVE_LOG_INFO("OpenCL matmul quant(Q8_0) + non-contig A test: PASS");
    return true;
}


} // namespace powerserve::opencl

int main() {
    bool ok4 = powerserve::opencl::run_opencl_backend_matmul_quant_noncontigA_test();
    return (ok4 ) ? 0 : 1;
}
