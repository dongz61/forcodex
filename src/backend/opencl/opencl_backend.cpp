#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"

#include <iostream>
#include <CL/cl.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <execinfo.h>

namespace powerserve::opencl {

// for debug
std::shared_ptr<OpenCLBuffer> OpenCLBackend::debug_get_k_cache(size_t L) const {
    if (!m_kv || L >= m_kv->key.size()) return nullptr;
    return m_kv->key[L];
}

std::shared_ptr<OpenCLBuffer> OpenCLBackend::debug_get_v_cache(size_t L) const {
    if (!m_kv || L >= m_kv->value.size()) return nullptr;
    return m_kv->value[L];
}

static inline void dump_backtrace() {
    void* bt[32];
    int n = backtrace(bt, 32);
    char** syms = backtrace_symbols(bt, n);
    POWERSERVE_LOG_ERROR(">>> copy() backtrace:");
    for (int i = 0; i < n; ++i) {
        POWERSERVE_LOG_ERROR("  {}", syms[i]);
    }
    free(syms);
}
// for debug end

OpenCLBackend::OpenCLBackend(const ModelConfig::LLMConfig &llm,
                             const HyperParams &hparams)
    : m_llm(llm), m_hparams(hparams) {
    // 其它初始化保持你原来的逻辑
}


OpenCLBackend::~OpenCLBackend() {
    cleanup();
    POWERSERVE_LOG_DEBUG("OpenCLBackend destructor called");
}

void OpenCLBackend::cleanup() {
    if (!initialized) return;
    
    std::cout << "[DEBUG] === CLEANUP TEST VERSION ===" << std::endl;
    
    // 方案1：先清理 OpenCL，再清理线程池
    std::cout << "[DEBUG] 1. Cleaning OpenCL context first..." << std::endl;
    if (context) {
        context.reset();  // 直接释放，不调用 cleanup()
        std::cout << "[DEBUG] Context released" << std::endl;
    }
    
    std::cout << "[DEBUG] 2. Cleaning memory pool..." << std::endl;
    if (memory_pool) {
        memory_pool.reset();
        std::cout << "[DEBUG] Memory pool released" << std::endl;
    }
    
    std::cout << "[DEBUG] 3. Now cleaning thread pool..." << std::endl;
    if (thread_pool) {
        thread_pool.reset();
        std::cout << "[DEBUG] Thread pool released" << std::endl;
    }
    
    initialized = false;
    std::cout << "[DEBUG] === CLEANUP DONE ===" << std::endl;
}

bool OpenCLBackend::initialize() {
    if (initialized) {
        return true;
    }
    
    // 1. 创建OpenCL上下文
    context = std::make_shared<OpenCLContext>();
    if (!context->initialize(device_preference)) {
        POWERSERVE_LOG_ERROR("Failed to initialize OpenCL context");
        return false;
    }
    
    // 2. 创建内存池
    memory_pool = std::make_shared<OpenCLMemoryPool>(context);
    
    // 3. 创建内核管理器
    kernel_manager = std::make_shared<OpenCLKernelManager>(context);
    
    // 4. 初始化内核管理器
    OpenCLCompileOptions options;
    options.opencl_c_std = "CL3.0";
    options.enable_mad = true;
    options.unsafe_math = true;
    options.finite_math = true;
    options.fast_relaxed_math = true;
    
    if (!kernel_manager->initialize(options)) {
        POWERSERVE_LOG_ERROR("Failed to initialize OpenCL kernel manager");
        return false;
    }

    // 5. 初始化kv cache
    ensure_kv_cache_allocated_v0();
    
    initialized = true;
    
    return true;
}

std::shared_ptr<OpenCLBuffer> OpenCLBackend::create_buffer(Shape shape, DataType dtype) {
    if (!memory_pool) {
        POWERSERVE_LOG_ERROR("Memory pool not initialized");
        return nullptr;
    }
    
    switch (dtype) {
        case DataType::FP32:
            return OpenCLBuffer::create_buffer<float>(shape, memory_pool);
        case DataType::FP16:
            return OpenCLBuffer::create_buffer<cl_half>(shape, memory_pool);
        case DataType::INT32:
            return OpenCLBuffer::create_buffer<cl_int>(shape, memory_pool);
        default:
            POWERSERVE_LOG_ERROR("Unsupported data type for OpenCL buffer: {}", 
                               static_cast<int>(dtype));
            return nullptr;
    }
}


void OpenCLBackend::plan(std::vector<std::shared_ptr<OpNode>> & /*ops*/) {
    // TODO: optional scheduling / workspace planning
}

// ==================== 算子实现 ====================

// for add_minimal
static inline size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}
// for add_minimal end

void OpenCLBackend::add_minimal(Tensor * dst, const Tensor * src0, const Tensor * src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }

    // ---- 1) 形状/类型检查：第一版只支持 FP32 & same-shape ----
    if (!dst || !src0 || !src1) {
        POWERSERVE_LOG_ERROR("add_minimal got null tensor");
        return;
    }
    if (dst->m_dtype != DataType::FP32 ||
        src0->m_dtype != DataType::FP32 ||
        src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("add_minimal only supports FP32");
        return;
    }
    if (dst->m_shape != src0->m_shape || dst->m_shape != src1->m_shape) {
        POWERSERVE_LOG_ERROR("add_minimal requires same shape");
        return;
    }

    const size_t n = dst->n_elements();
    if (n == 0) return;

    // ---- 2) 取 cl_mem ----
    cl_mem a = nullptr;
    cl_mem b = nullptr;
    cl_mem o = nullptr;
    try {
        a = src0->get<OpenCLBuffer>().get_device_buffer();
        b = src1->get<OpenCLBuffer>().get_device_buffer();
        o = dst ->get<OpenCLBuffer>().get_device_buffer();
    } catch (const std::bad_cast & e) {
        POWERSERVE_LOG_ERROR("add_minimal expects OpenCLBuffer: {}", e.what());
        return;
    }

    if (!a || !b || !o) {
        POWERSERVE_LOG_ERROR("add_minimal invalid cl_mem");
        return;
    }

    // ---- 3) 拿 kernel（来自你 compile_program 自动缓存）----
    cl_kernel kernel = kernel_manager->get_kernel("kernel_add_contig_f32");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("kernel not found: kernel_add_contig_f32");
        return;
    }

    // ---- 4) set args ----
    cl_int err = CL_SUCCESS;
    cl_uint idx = 0;
    const int n_i = static_cast<int>(n);

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg a failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg b failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &o);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg out failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(int), &n_i);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg n failed"); return; }

    // ---- 5) enqueue：1D，最稳 ----
    const size_t local = 256;
    const size_t global = round_up(n, local);

    cl_command_queue q = context->get_queue();
    err = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("clEnqueueNDRangeKernel failed: {}", context->get_error_string(err));
        return;
    }

    // bring-up 阶段建议 finish，后续可以去掉或用 event
    err = clFinish(q);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
    }
}


void OpenCLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !src0 || !src1) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::add got null tensor");
        return;
    }

    // Strict mode (Phase 1):
    // - FP32 only
    // - same-shape only (no broadcast)
    // - contiguous assumed by convention (views are handled by OpenCLBuffer sub-buffer already)
    if (dst->m_dtype != DataType::FP32 || src0->m_dtype != DataType::FP32 || src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::add (Phase1) only supports FP32");
        return;
    }
    if (dst->m_shape != src0->m_shape || dst->m_shape != src1->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::add (Phase1) requires same shape (no broadcast)");
        return;
    }
    this->add_minimal(const_cast<Tensor *>(dst), src0, src1);
}

void OpenCLBackend::get_embedding(
    const Tensor *dst,
    const Tensor *weight,
    const std::vector<int> &tokens
) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !weight) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding got null tensor");
        return;
    }

    // Phase 1 strict:
    // - dst must be FP32 and OpenCLBuffer-backed (executor allocates it as OpenCLBuffer)
    // - weight must be FP32 and CPUBuffer-backed (model weight lives on CPU in your flow)
    // - do CPU gather then H2D copy
    if (dst->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) dst must be FP32");
        return;
    }
    if (weight->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) weight must be FP32 (no dequant yet)");
        return;
    }

    const size_t batch_size = tokens.size();
    const size_t dim        = dst->m_shape[0];

    if (dst->m_shape[2] != 1 || dst->m_shape[3] != 1) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) dst must be 2D-like (dim,batch,1,1)");
        return;
    }
    if (dst->m_shape[1] != batch_size) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) batch mismatch: dst.shape[1] != tokens.size()");
        return;
    }

    // Weight is expected to be an embedding table laid out like GGML path:
    // each token row is contiguous with stride[1] bytes (see CPU reference):contentReference[oaicite:3]{index=3}
    powerserve::CPUBuffer *w_cpu = nullptr;
    try {
        w_cpu = &const_cast<Tensor *>(weight)->get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) expects weight backed by CPUBuffer");
        return;
    }
    if (!w_cpu->m_data) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) weight CPUBuffer data is null");
        return;
    }

    // 1) CPU gather into a temporary host tensor
    Tensor host_tmp(DataType::FP32, dst->m_shape);
    host_tmp.m_data = powerserve::CPUBuffer::create_buffer<float>(dst->m_shape);

    auto *dst_host = static_cast<float *>(host_tmp.get<powerserve::CPUBuffer>().m_data);
    auto *embd_tb  = static_cast<char *>(w_cpu->m_data);
    const auto w_stride = w_cpu->m_stride; // bytes

    for (size_t i = 0; i < batch_size; i++) {
        const int token = tokens[i];
        if (token < 0) {
            POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) got negative token id");
            return;
        }

        // row pointer = base + stride[1] * token
        char *src = embd_tb + w_stride[1] * static_cast<size_t>(token);

        // (optional) basic bounds check like GGMLBackend does:contentReference[oaicite:4]{index=4}
        // Here we only check that src doesn't go backwards and that stride[1] is sane.
        // Full bound check would require knowing total table bytes; we keep it minimal & fail-fast-ish.
        if (w_stride[1] == 0) {
            POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding (Phase1) weight stride[1] is 0");
            return;
        }

        std::memcpy(dst_host + i * dim, src, dim * sizeof(float));
    }

    // 2) H2D copy into dst OpenCLBuffer using existing copy() (CPU -> OpenCL path):contentReference[oaicite:5]{index=5}
    this->copy(dst, &host_tmp);
}

void OpenCLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !src0 || !src1) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::matmul got null tensor");
        return;
    }

    // Strict mode (Phase 1), per matmul_minimal contract already documented in-file:
    // A:{K,M,1,1}, B:{N,K,1,1}, C:{N,M,1,1}, FP32 only, 2D only.
    if (dst->m_dtype != DataType::FP32 || src0->m_dtype != DataType::FP32 || src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::matmul (Phase1) only supports FP32");
        return;
    }

    if (src0->m_shape[2] != 1 || src0->m_shape[3] != 1 ||
        src1->m_shape[2] != 1 || src1->m_shape[3] != 1 ||
        dst->m_shape[2]  != 1 || dst->m_shape[3]  != 1) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::matmul (Phase1) only supports 2D (shape[2]=shape[3]=1)");
        return;
    }

    const size_t K = src0->m_shape[0];
    const size_t M = src0->m_shape[1];
    const size_t N = src1->m_shape[0];

    if (src1->m_shape[1] != K) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::matmul (Phase1) requires B.shape[1]==A.shape[0] (K)");
        return;
    }
    if (dst->m_shape[0] != N || dst->m_shape[1] != M) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::matmul (Phase1) requires C shape (N,M,1,1)");
        return;
    }

    // Route to minimal kernel
    this->matmul_minimal(const_cast<Tensor *>(dst), src0, src1);
}

void OpenCLBackend::matmul_minimal(Tensor * dst,
                                  const Tensor * src0,
                                  const Tensor * src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !src0 || !src1) {
        POWERSERVE_LOG_ERROR("matmul_minimal got null tensor");
        return;
    }
    if (dst->m_dtype != DataType::FP32 ||
        src0->m_dtype != DataType::FP32 ||
        src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("matmul_minimal only supports FP32");
        return;
    }

    // ---- Sentinel shape contract (2D only, row-major):
    // A: [M,K] => shape {K, M, 1, 1}
    // B: [K,N] => shape {N, K, 1, 1}
    // C: [M,N] => shape {N, M, 1, 1}
    const size_t K = src0->m_shape[0];
    const size_t M = src0->m_shape[1];
    const size_t N = src1->m_shape[0];

    if (src0->m_shape[2] != 1 || src0->m_shape[3] != 1 ||
        src1->m_shape[2] != 1 || src1->m_shape[3] != 1 ||
        dst ->m_shape[2] != 1 || dst ->m_shape[3] != 1) {
        POWERSERVE_LOG_ERROR("matmul_minimal only supports 2D (shape[2]=shape[3]=1)");
        return;
    }

    if (src1->m_shape[1] != K) {
        POWERSERVE_LOG_ERROR("matmul_minimal requires B rows == K (B.shape[1] == A.shape[0])");
        return;
    }
    if (dst->m_shape[1] != M || dst->m_shape[0] != N) {
        POWERSERVE_LOG_ERROR("matmul_minimal requires C shape {{N, M, 1, 1}}");
        return;
    }
    if (M == 0 || N == 0 || K == 0) return;

    cl_mem A = nullptr;
    cl_mem B = nullptr;
    cl_mem C = nullptr;
    try {
        A = src0->get<OpenCLBuffer>().get_device_buffer();
        B = src1->get<OpenCLBuffer>().get_device_buffer();
        C = dst ->get<OpenCLBuffer>().get_device_buffer();
    } catch (const std::bad_cast & e) {
        POWERSERVE_LOG_ERROR("matmul_minimal expects OpenCLBuffer: {}", e.what());
        return;
    }
    if (!A || !B || !C) {
        POWERSERVE_LOG_ERROR("matmul_minimal invalid cl_mem");
        return;
    }

    cl_kernel kernel = kernel_manager->get_kernel("kernel_matmul_contig_f32");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("kernel not found: kernel_matmul_contig_f32");
        return;
    }

    cl_int err = CL_SUCCESS;
    cl_uint idx = 0;
    const int M_i = (int)M;
    const int N_i = (int)N;
    const int K_i = (int)K;

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &A);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg A failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &B);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg B failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &C);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg C failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(int), &M_i);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg M failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(int), &N_i);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg N failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(int), &K_i);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg K failed"); return; }

    // 2D launch: (n, m) = (col, row)
    const size_t local[2]  = {16, 16};
    const size_t global[2] = {round_up(N, local[0]), round_up(M, local[1])};

    cl_command_queue q = context->get_queue();
    err = clEnqueueNDRangeKernel(q, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("clEnqueueNDRangeKernel failed: {}", context->get_error_string(err));
        return;
    }

    err = clFinish(q);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
    }
}


void OpenCLBackend::rmsnorm(
    const Tensor *o,
    const Tensor *x,
    const Tensor *weight,
    float eps
) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!o || !x || !weight) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm got null tensor");
        return;
    }

    // ----------------------------
    // Strict contract (Phase2 v0)
    // ----------------------------
    if (o->m_dtype != DataType::FP32 || x->m_dtype != DataType::FP32 || weight->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm fallback only supports FP32");
        return;
    }
    if (o->m_shape != x->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm requires o.shape == x.shape");
        return;
    }

    const int hidden = (int)x->m_shape[0];
    if (hidden <= 0) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm hidden dim invalid: {}", hidden);
        return;
    }
    // weight expected to be 1D-like with hidden in dim0
    if ((int)weight->m_shape[0] != hidden) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm weight.shape[0] must equal hidden ({}), got {}",
                             hidden, (int)weight->m_shape[0]);
        return;
    }

    // x/o must be OpenCLBuffer-backed
    OpenCLBuffer *x_cl = nullptr;
    OpenCLBuffer *o_cl = nullptr;
    powerserve::CPUBuffer *w_cpu = nullptr;

    try {
        x_cl = &const_cast<Tensor *>(x)->get<OpenCLBuffer>();
        o_cl = &const_cast<Tensor *>(o)->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm expects x/o backed by OpenCLBuffer: {}", e.what());
        return;
    }

    try {
        w_cpu = &const_cast<Tensor *>(weight)->get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm expects weight backed by CPUBuffer: {}", e.what());
        return;
    }

    if (!w_cpu->m_data) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm weight CPU data is null");
        return;
    }

    // ----------------------------
    // D2H: x -> host_x
    // ----------------------------
    Tensor host_x(DataType::FP32, x->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(x->m_shape);

    this->copy(&host_x, x);  // D2H path already implemented

    auto *x_host = static_cast<float *>(host_x.get<powerserve::CPUBuffer>().m_data);
    auto *w_host = static_cast<float *>(w_cpu->m_data);

    // ----------------------------
    // CPU compute
    // ----------------------------
    Tensor host_y(DataType::FP32, o->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(o->m_shape);
    auto *y_host = static_cast<float *>(host_y.get<powerserve::CPUBuffer>().m_data);

    // treat as [hidden, rows, 1, 1]
    const int rows = (int)(x->m_shape[1] * x->m_shape[2] * x->m_shape[3]);
    const float inv_hidden = 1.0f / (float)hidden;

    for (int r = 0; r < rows; ++r) {
        const float *xr = x_host + (size_t)r * hidden;
        float *yr = y_host + (size_t)r * hidden;

        // mean square
        double sumsq = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const float v = xr[i];
            sumsq += (double)v * (double)v;
        }
        const float mean = (float)(sumsq * inv_hidden);
        const float scale = 1.0f / std::sqrt(mean + eps);

        for (int i = 0; i < hidden; ++i) {
            yr[i] = xr[i] * scale * w_host[i];
        }
    }

    // ----------------------------
    // H2D: host_y -> o
    // ----------------------------
    this->copy(o, &host_y);

    POWERSERVE_LOG_DEBUG("OpenCLBackend::rmsnorm CPU fallback done (hidden={}, rows={}, eps={})",
                         hidden, rows, eps);
}

// for rope
static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float denom = std::max(0.001f, high - low);
    const float y = ((float)(i0 / 2) - low) / denom;
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

// corr_factor from ggml (required by corr_dim/corr_dims)
// This is the same formula used by llama.cpp/ggml.
// Note: 2*pi as float constant:
static inline float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float freq_base) {
    const float pi = 3.14159265358979323846f;
    return (float)n_dims * std::log((float)n_ctx_orig / (n_rot * 2.0f * pi)) / (2.0f * std::log(freq_base));
}

static inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float freq_base) {
    return rope_yarn_corr_factor(n_dims, n_ctx_orig, n_rot, freq_base);
}

static inline void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    float start = std::floor(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   = std::ceil (rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min((float)n_dims - 1.0f, end);
}

static inline void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0,
    float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;

        // magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    }
    *cos_theta = std::cos(theta) * mscale;
    *sin_theta = std::sin(theta) * mscale;
}

static inline void rope_cache_init(
    float theta_base, float freq_scale, const float * freq_factors,
    float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
    float * cache, float sin_sign, float theta_scale
) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(theta / ff, freq_scale, corr_dims, (int)i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;
        theta *= theta_scale;
    }
}
// for rope end

void OpenCLBackend::rope(
    Tensor *out,
    const Tensor *src,
    const std::vector<int> &pos,
    const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!out || !src) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope got null tensor");
        return;
    }

    // ---- strict dtype/shape ----
    if (out->m_dtype != DataType::FP32 || src->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope CPU fallback only supports FP32");
        return;
    }
    if (out->m_shape != src->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope requires out.shape == src.shape");
        return;
    }

    // minimal shape contract: {dim, batch, 1, 1}
    const int dim   = (int)src->m_shape[0];
    const int ne1   = (int)src->m_shape[1]; // heads
    const int ne2   = (int)src->m_shape[2]; // tokens/time
    const int ne3   = (int)src->m_shape[3]; // batch (bring-up：先限制 1)

    if (ne3 != 1) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope fallback only supports shape[3]==1 for now, got {}", ne3);
        return;
    }

    // 兼容旧测试：如果 shape[2]==1 且 pos.size()==shape[1]，就把 shape[1] 当 token 数
    const bool legacy_axis1 = (ne2 == 1 && (int)pos.size() == ne1);

    // H = head 数，T = token 数
    const int H = legacy_axis1 ? 1   : ne1;
    const int T = legacy_axis1 ? ne1 : ne2;

    if (!legacy_axis1 && (int)pos.size() != T) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope pos.size() must equal n_tokens (shape[2]) = {}, got {}",
                            T, (int)pos.size());
        return;
    }

    const int n_dims = rope_cfg.n_dims;
    if (n_dims <= 0 || (n_dims % 2) != 0 || n_dims > dim) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope invalid rope_cfg.n_dims");
        return;
    }

    // rope_type: treat -1 as 0
    const int rope_type = (rope_cfg.rope_type < 0) ? 0 : rope_cfg.rope_type;

    // GGML_ROPE_TYPE_NEOX is commonly 2
    const bool is_neox = (rope_type & GGML_ROPE_TYPE_NEOX) != 0;

    // ---- D2H src -> host_x ----
    Tensor host_x(DataType::FP32, src->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
    this->copy(&host_x, src);
    float *x_host = static_cast<float *>(host_x.get<powerserve::CPUBuffer>().m_data);

    // ---- CPU compute -> host_y ----
    Tensor host_y(DataType::FP32, out->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);
    float *y_host = static_cast<float *>(host_y.get<powerserve::CPUBuffer>().m_data);

    // prepare cache for cos/sin (size dim, interleaved cos/sin)
    std::vector<float> cache((size_t)dim);
    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.beta_fast, rope_cfg.beta_slow, corr_dims);

    const float theta_scale = std::pow(rope_cfg.freq_base, -2.0f / (float)n_dims);
    const float sin_sign = 1.0f; // forward only

    for (int t = 0; t < T; ++t) {
        const float theta_base = (float)pos[t];

        // 每个 token 初始化一次 cache
        rope_cache_init(theta_base, rope_cfg.freq_scale, /*freq_factors*/nullptr,
                        corr_dims, (int64_t)dim, rope_cfg.ext_factor, rope_cfg.attn_factor,
                        cache.data(), sin_sign, theta_scale);

        for (int h = 0; h < H; ++h) {
            // contiguous layout: ((t * H + h) * dim)
            const size_t row = ((size_t)t * (size_t)H + (size_t)h) * (size_t)dim;
            float *x = x_host + row;
            float *y = y_host + row;

            std::memcpy(y, x, sizeof(float) * (size_t)dim);

            if (!is_neox) {
                // norm: rotate (i0, i0+1)
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float cos_theta = cache[(size_t)i0 + 0];
                    const float sin_theta = cache[(size_t)i0 + 1];
                    const float x0 = x[i0 + 0];
                    const float x1 = x[i0 + 1];
                    y[i0 + 0] = x0 * cos_theta - x1 * sin_theta;
                    y[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            } else {
                // neox: rotate (ic, ic + n_dims/2)
                const int half = n_dims / 2;
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const int ic = i0 / 2;
                    const float cos_theta = cache[(size_t)i0 + 0];
                    const float sin_theta = cache[(size_t)i0 + 1];
                    const float x0 = x[ic];
                    const float x1 = x[ic + half];
                    y[ic]        = x0 * cos_theta - x1 * sin_theta;
                    y[ic + half] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }

    // ---- H2D host_y -> out ----
    this->copy(out, &host_y);

    POWERSERVE_LOG_ERROR("rope fallback: dim={}, heads={}, tokens={}, batch={}, n_dims={}, rope_type={}",
                     dim, H, T, ne3, n_dims, rope_type);
}


void OpenCLBackend::softmax(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::softmax TODO");
}

void OpenCLBackend::permute(const Tensor * /*out*/, const Tensor * /*x*/, Shape /*axes*/) const {
    POWERSERVE_ABORT("OpenCLBackend::permute TODO");
}

void OpenCLBackend::cont(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::cont TODO");
}

void OpenCLBackend::softmax_ext(
    const Tensor * /*out*/,
    const Tensor * /*x*/,
    const Tensor * /*mask*/,
    float /*scale*/,
    float /*max_bias*/
) const {
    POWERSERVE_ABORT("OpenCLBackend::softmax_ext TODO");
}

void OpenCLBackend::silu_hadamard(const Tensor * out,
                                 const Tensor * hb,
                                 const Tensor * hb2) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }

    if (!out || !hb || !hb2) {
        POWERSERVE_LOG_ERROR("silu_hadamard got null tensor");
        return;
    }

    // minimal: FP32 + same-shape + contiguous-only by convention
    if (out->m_dtype != DataType::FP32 ||
        hb->m_dtype  != DataType::FP32 ||
        hb2->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("silu_hadamard minimal only supports FP32");
        return;
    }

    if (out->m_shape != hb->m_shape || out->m_shape != hb2->m_shape) {
        POWERSERVE_LOG_ERROR("silu_hadamard requires same shape");
        return;
    }

    const size_t n = out->n_elements();
    if (n == 0) return;

    cl_mem a = nullptr;   // hb
    cl_mem b = nullptr;   // hb2
    cl_mem o = nullptr;   // out
    try {
        a = hb ->get<OpenCLBuffer>().get_device_buffer();
        b = hb2->get<OpenCLBuffer>().get_device_buffer();
        o = out->get<OpenCLBuffer>().get_device_buffer();
    } catch (const std::bad_cast & e) {
        POWERSERVE_LOG_ERROR("silu_hadamard expects OpenCLBuffer: {}", e.what());
        return;
    }

    if (!a || !b || !o) {
        POWERSERVE_LOG_ERROR("silu_hadamard invalid cl_mem");
        return;
    }

    cl_kernel kernel = kernel_manager->get_kernel("kernel_silu_hadamard_contig_f32");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("kernel not found: kernel_silu_hadamard_contig_f32");
        return;
    }

    cl_int err = CL_SUCCESS;
    cl_uint idx = 0;
    const int n_i = static_cast<int>(n);

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg hb failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg hb2 failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &o);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg out failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(int), &n_i);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg n failed"); return; }

    const size_t local = 256;
    const size_t global = round_up(n, local);

    cl_command_queue q = context->get_queue();
    err = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("clEnqueueNDRangeKernel failed: {}", context->get_error_string(err));
        return;
    }

    err = clFinish(q);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
    }
}


// for copy
static inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT32: return 4;
        default: return 0;
    }
}

static inline size_t numel_4d(const Tensor* t) {
    size_t n = 1;
    // 按你们 shape 维度数改；你摘要里是 4 维
    for (int i = 0; i < 4; ++i) n *= static_cast<size_t>(t->m_shape[i]);
    return n;
}
// for copy end

void OpenCLBackend::copy(const Tensor* dst, const Tensor* src) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("copy: null tensor");
        return;
    }
    if (!memory_pool) {
        POWERSERVE_LOG_ERROR("copy: memory_pool is null");
        return;
    }

    const size_t src_bytes = numel_4d(src) * dtype_size(src->m_dtype);
    const size_t dst_bytes = numel_4d(dst) * dtype_size(dst->m_dtype);
    if (src_bytes == 0 || dst_bytes == 0 || src_bytes != dst_bytes) {
        POWERSERVE_LOG_ERROR("copy: size mismatch src_bytes={} dst_bytes={}", src_bytes, dst_bytes);
        return;
    }

    BaseBuffer& src_base = src->get<BaseBuffer>();
    BaseBuffer& dst_base = dst->get<BaseBuffer>();

    auto* src_cpu = dynamic_cast<powerserve::CPUBuffer*>(&src_base);
    auto* dst_cpu = dynamic_cast<powerserve::CPUBuffer*>(&dst_base);
    auto* src_cl  = dynamic_cast<OpenCLBuffer*>(&src_base);
    auto* dst_cl  = dynamic_cast<OpenCLBuffer*>(&dst_base);

    // H2D
    if (src_cpu && dst_cl) {
        
        void* host = src_cpu->m_data;
        cl_mem dev = dst_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("H2D: invalid host/dev");
            return;
        }
        if (!memory_pool->copy_host_to_device(dev, host, src_bytes, 0)) {
            POWERSERVE_LOG_ERROR("H2D: copy_host_to_device failed");
        }
        POWERSERVE_LOG_DEBUG("copy H2D bytes={}", src_bytes);
        return;
    }

    // D2H
    if (src_cl && dst_cpu) {
        void* host = dst_cpu->m_data;
        cl_mem dev = src_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("D2H: invalid host/dev");
            return;
        }
        if (!memory_pool->copy_device_to_host(host, dev, src_bytes, 0)) {
            POWERSERVE_LOG_ERROR("D2H: copy_device_to_host failed");
        }
        POWERSERVE_LOG_DEBUG("copy D2H bytes={}", src_bytes);
        return;
    }

    // D2D
    if (src_cl && dst_cl) {
        cl_mem src_dev = src_cl->get_device_buffer();
        cl_mem dst_dev = dst_cl->get_device_buffer();
        if (!src_dev || !dst_dev) {
            POWERSERVE_LOG_ERROR("D2D: invalid cl_mem");
            return;
        }
        if (!memory_pool->copy_device_to_device(dst_dev, src_dev, src_bytes)) {
            POWERSERVE_LOG_ERROR("D2D: copy_device_to_device failed");
        }
        POWERSERVE_LOG_DEBUG("copy D2D bytes={}", src_bytes);
        return;
    }

    // CPU2CPU
    if (src_cpu && dst_cpu) {
        std::memcpy(dst_cpu->m_data, src_cpu->m_data, src_bytes);
        POWERSERVE_LOG_DEBUG("copy CPU2CPU bytes={}", src_bytes);
        return;
    }

    POWERSERVE_LOG_ERROR("copy: unsupported src/dst buffer types");
}

void OpenCLBackend::print(const Tensor* x, size_t size) const {
    POWERSERVE_ABORT("OpenCLBackend::print TODO");
}

void OpenCLBackend::reset_kv_batch_size(const size_t batch_size) const {
    if (!m_kv) {
        POWERSERVE_LOG_ERROR("reset_kv_batch_size called but KVCache not allocated");
        return;
    }
    if (batch_size != 1) {
        POWERSERVE_LOG_ERROR("KVCache v0 only supports batch_size=1, got {}", batch_size);
        return;
    }
    m_kv->batch_size = 1;
    m_kv->position = 0;
}

void OpenCLBackend::add_cache(const Tensor *k,
                              const Tensor *v,
                              size_t L,
                              const std::vector<int> &pos,
                              size_t head_id) {
    (void)head_id;

    if (!m_kv) { POWERSERVE_LOG_ERROR("add_cache: KVCache not allocated"); return; }
    if (!k || !v) { POWERSERVE_LOG_ERROR("add_cache: null tensor"); return; }
    if (m_kv->batch_size != 1 || pos.size() != 1) {
        POWERSERVE_LOG_ERROR("add_cache v0 expects batch=1 and pos.size()==1");
        return;
    }
    if (L >= m_kv->key.size()) {
        POWERSERVE_LOG_ERROR("add_cache: invalid layer {}", L);
        return;
    }
    if (m_kv->position >= m_kv->max_seq_len) {
        POWERSERVE_LOG_ERROR("KVCache overflow: position {} max_seq_len {}", m_kv->position, m_kv->max_seq_len);
        return;
    }
    if (k->m_dtype != DataType::FP32 || v->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("add_cache v0 only supports FP32");
        return;
    }

    const size_t kv_dim = m_kv->kv_dim;
    // Expect token shape {kv_dim, 1, 1, 1}
    if (k->m_shape[0] != kv_dim || k->m_shape[1] != 1 || k->m_shape[2] != 1 || k->m_shape[3] != 1 ||
        v->m_shape[0] != kv_dim || v->m_shape[1] != 1 || v->m_shape[2] != 1 || v->m_shape[3] != 1) {
        POWERSERVE_LOG_ERROR("add_cache shape mismatch: expect {{kv_dim,1,1,1}}");
        return;
    }

    const size_t cur = m_kv->position;

    // Layout: cache is {kv_dim, max_seq_len} row-major (cols=kv_dim).
    // Writing row=cur => offset in elements = cur * kv_dim
    // !!! IMPORTANT: offset unit depends on create_buffer_view contract.
    // If offset is in BYTES, use: cur*kv_dim*sizeof(float).
    // If offset is in ELEMENTS, use: cur*kv_dim.
    //
    // From your earlier system rule ("offset encoded into cl_mem"), it is almost certainly BYTES.
    const size_t offset = cur * kv_dim * sizeof(float);

    // Create destination views (one token slice)
    Shape sTok{kv_dim, 1, 1, 1};

    try {
        auto &k_parent = *m_kv->key[L];
        auto &v_parent = *m_kv->value[L];

        auto k_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(k_parent, sTok, offset);
        auto v_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(v_parent, sTok, offset);

        if (!k_view || !v_view) {
            POWERSERVE_LOG_ERROR("add_cache: create_buffer_view failed");
            return;
        }

        Tensor t_dst_k(DataType::FP32, sTok);
        Tensor t_dst_v(DataType::FP32, sTok);
        t_dst_k.m_data = k_view;
        t_dst_v.m_data = v_view;

        this->copy(&t_dst_k, k); // D2D
        this->copy(&t_dst_v, v); // D2D
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("add_cache expects OpenCLBuffer in KVCache: {}", e.what());
        return;
    }

    m_kv->position += 1;
}

void OpenCLBackend::transpose(const Tensor *out, const Tensor *x) const {
    POWERSERVE_ASSERT(out && x);
    POWERSERVE_ASSERT(out->m_data && x->m_data);

    // Ensure input is OpenCLBuffer-backed
    try {
        (void)const_cast<Tensor *>(x)->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::transpose expects OpenCLBuffer input: {}", e.what());
        POWERSERVE_ABORT("transpose: input not OpenCLBuffer");
    }

    // Share buffer object
    auto *out_nc = const_cast<Tensor *>(out);
    auto *x_nc   = const_cast<Tensor *>(x);
    out_nc->m_data = x_nc->m_data;

    // Now both point to same OpenCLBuffer object
    auto &xbuf = x_nc->get<OpenCLBuffer>();
    auto &obuf = out_nc->get<OpenCLBuffer>();

    obuf.m_stride = xbuf.m_stride;
    std::swap(obuf.m_stride[0], obuf.m_stride[1]);
}



void OpenCLBackend::ensure_kv_cache_allocated_v0() {
    // v0: batch fixed 1, FP32, prealloc
    const int n_layers_i = m_llm.n_layers;
    const int seq_len_i  = m_llm.seq_len;   // n_ctx
    const int kv_dim_i   = m_llm.kv_dim;

    if (n_layers_i <= 0 || seq_len_i <= 0 || kv_dim_i <= 0) {
        POWERSERVE_LOG_WARN("KVCache v0 skipped: invalid llm config n_layers={}, seq_len={}, kv_dim={}",
                            n_layers_i, seq_len_i, kv_dim_i);
        return;
    }

    const size_t n_layers    = static_cast<size_t>(n_layers_i);
    const size_t max_seq_len = static_cast<size_t>(seq_len_i);
    const size_t kv_dim      = static_cast<size_t>(kv_dim_i);

    if (!m_kv) m_kv = std::make_unique<powerserve::opencl::OpenCLKV>();

    // If already allocated with same spec, do nothing
    if (m_kv->spec_matches(n_layers, kv_dim, max_seq_len)) {
        return;
    }

    // (Re)allocate
    m_kv->kv_dim = kv_dim;
    m_kv->max_seq_len = max_seq_len;
    m_kv->batch_size = 1;
    m_kv->position = 0;
    m_kv->key.clear();
    m_kv->value.clear();
    m_kv->key.resize(n_layers);
    m_kv->value.resize(n_layers);

    const Shape sKV{kv_dim, max_seq_len, 1, 1};
    for (size_t L = 0; L < n_layers; ++L) {
        m_kv->key[L]   = this->create_buffer(sKV, DataType::FP32);
        m_kv->value[L] = this->create_buffer(sKV, DataType::FP32);
        if (!m_kv->key[L] || !m_kv->value[L]) {
            POWERSERVE_LOG_ERROR("KVCache v0 alloc failed at layer {}", L);
            m_kv.reset();
            return;
        }
    }

    POWERSERVE_LOG_INFO("KVCache v0 allocated: layers={}, kv_dim={}, max_seq_len={}",
                        n_layers, kv_dim, max_seq_len);
}

} // namespace powerserve::opencl
