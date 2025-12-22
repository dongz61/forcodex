#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"

#include <iostream>
#include <CL/cl.h>

namespace powerserve::opencl {

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


void OpenCLBackend::add(const Tensor * /*dst*/, const Tensor * /*src0*/, const Tensor * /*src1*/) const {
    POWERSERVE_ABORT("OpenCLBackend::add TODO");
}

void OpenCLBackend::get_embedding(
    const Tensor * /*dst*/,
    const Tensor * /*weight*/,
    const std::vector<int> & /*tokens*/
) const {
    POWERSERVE_ABORT("OpenCLBackend::get_embedding TODO");
}

void OpenCLBackend::matmul(const Tensor * /*dst*/, const Tensor * /*src0*/, const Tensor * /*src1*/) const {
    POWERSERVE_ABORT("OpenCLBackend::matmul TODO");
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
    const Tensor * /*o*/,
    const Tensor * /*x*/,
    const Tensor * /*weight*/,
    float /*eps*/
) const {
    POWERSERVE_ABORT("OpenCLBackend::rmsnorm TODO");
}

void OpenCLBackend::rope(
    Tensor * /*out*/,
    const Tensor * /*src*/,
    const std::vector<int> & /*pos*/,
    const ModelConfig::LLMConfig::RopeConfig & /*rope_cfg*/
) const {
    POWERSERVE_ABORT("OpenCLBackend::rope TODO");
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

    // 1) bytes 计算（建议你们项目里封装成 Tensor::nbytes()）
    const size_t src_bytes = numel_4d(src) * dtype_size(src->m_dtype);
    const size_t dst_bytes = numel_4d(dst) * dtype_size(dst->m_dtype);
    if (src_bytes == 0 || dst_bytes == 0 || src_bytes != dst_bytes) {
        POWERSERVE_LOG_ERROR("copy: size mismatch src_bytes={} dst_bytes={}", src_bytes, dst_bytes);
        return;
    }

    // 2) 取 buffer（按你们 Tensor 接口改）
    BaseBuffer& src_base = src->get<BaseBuffer>();
    BaseBuffer& dst_base = dst->get<BaseBuffer>();

    auto* src_cpu = dynamic_cast<powerserve::CPUBuffer*>(&src_base);
    auto* dst_cpu = dynamic_cast<powerserve::CPUBuffer*>(&dst_base);
    auto* src_cl  = dynamic_cast<OpenCLBuffer*>(&src_base);
    auto* dst_cl  = dynamic_cast<OpenCLBuffer*>(&dst_base);

    cl_command_queue q = context->get_queue();
    if (!q) {
        POWERSERVE_LOG_ERROR("copy: OpenCL queue is null");
        return;
    }

    cl_int err = CL_SUCCESS;

    // --- H2D: CPU -> OpenCL ---
    if (src_cpu && dst_cl) {
        void* host = src_cpu->m_data; 
        cl_mem dev = dst_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("H2D: invalid host/dev");
            return;
        }

        err = clEnqueueWriteBuffer(
            q, dev,
            CL_TRUE,   // bring-up 阶段用 blocking，最稳
            0,         // !!! 你们 sub-buffer 模式下永远 0
            src_bytes,
            host,
            0, nullptr, nullptr
        );
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("H2D clEnqueueWriteBuffer failed: {}", context->get_error_string(err));
            return;
        }

        POWERSERVE_LOG_DEBUG("copy H2D bytes={}", src_bytes);
        return;
    }

    // --- D2H: OpenCL -> CPU ---
    if (src_cl && dst_cpu) {
        void* host = dst_cpu->m_data; 
        cl_mem dev = src_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("D2H: invalid host/dev");
            return;
        }

        err = clEnqueueReadBuffer(
            q, dev,
            CL_TRUE,
            0,         // !!! sub-buffer 模式下永远 0
            src_bytes,
            host,
            0, nullptr, nullptr
        );
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("D2H clEnqueueReadBuffer failed: {}", context->get_error_string(err));
            return;
        }

        POWERSERVE_LOG_DEBUG("copy D2H bytes={}", src_bytes);
        return;
    }

    // --- D2D: OpenCL -> OpenCL（含 sub-buffer/view）---
    if (src_cl && dst_cl) {
        cl_mem src_dev = src_cl->get_device_buffer();
        cl_mem dst_dev = dst_cl->get_device_buffer();
        if (!src_dev || !dst_dev) {
            POWERSERVE_LOG_ERROR("D2D: invalid cl_mem");
            return;
        }

        err = clEnqueueCopyBuffer(
            q,
            src_dev,
            dst_dev,
            0, 0,       // !!! 两边都 0，因为 cl_mem 已经是正确的 sub-buffer
            src_bytes,
            0, nullptr, nullptr
        );
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("D2D clEnqueueCopyBuffer failed: {}", context->get_error_string(err));
            return;
        }

        // bring-up 阶段可以 finish，保证确定性；后面再优化成 event/异步
        err = clFinish(q);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
        }

        POWERSERVE_LOG_DEBUG("copy D2D bytes={}", src_bytes);
        return;
    }

    // --- CPU -> CPU（可选兜底） ---
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

void OpenCLBackend::transpose(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::transpose TODO");
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

// for debug
std::shared_ptr<OpenCLBuffer> OpenCLBackend::debug_get_k_cache(size_t L) const {
    if (!m_kv || L >= m_kv->key.size()) return nullptr;
    return m_kv->key[L];
}
std::shared_ptr<OpenCLBuffer> OpenCLBackend::debug_get_v_cache(size_t L) const {
    if (!m_kv || L >= m_kv->value.size()) return nullptr;
    return m_kv->value[L];
}

} // namespace powerserve::opencl
