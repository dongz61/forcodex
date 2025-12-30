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

#define CL_CHECK(call) \
    do { \
        cl_int _err = (call); \
        if (_err != CL_SUCCESS) { \
            POWERSERVE_LOG_ERROR("OpenCL error at {}:{} - {}: {}", \
                __FILE__, __LINE__, #call, context->get_error_string(_err)); \
            return; \
        } \
    } while (0)


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

static inline const char* dtype_name(DataType t) {
    switch (t) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::INT32: return "INT32";
        default: return "OTHER";
    }
}

static inline void log_tensor_meta(OpenCLBackend *self, const char *tag, const Tensor *t, int n = 4) {
    if (!t) {
        POWERSERVE_LOG_ERROR("[MATMUL][{}] <null>", tag);
        return;
    }
    bool cont = self->is_contiguous(t, n);

    powerserve::Stride st{};
    bool has_stride = false;
    try {
        const auto &buf = const_cast<Tensor*>(t)->get<OpenCLBuffer>();
        st = buf.m_stride;
        has_stride = true;
    } catch (...) {
        try {
            const auto &buf = const_cast<Tensor*>(t)->get<powerserve::CPUBuffer>();
            st = buf.m_stride;
            has_stride = true;
        } catch (...) {
            has_stride = false;
        }
    }

    if (has_stride) {
        POWERSERVE_LOG_ERROR("[MATMUL][{}] dtype={} shape={{{},{},{},{}}} strideB={{{},{},{},{}}} cont={}",
            tag, dtype_name(t->m_dtype),
            t->m_shape[0], t->m_shape[1], t->m_shape[2], t->m_shape[3],
            st[0], st[1], st[2], st[3],
            cont);
    } else {
        POWERSERVE_LOG_ERROR("[MATMUL][{}] dtype={} shape={{{},{},{},{}}} stride=<unknown> cont={}",
            tag, dtype_name(t->m_dtype),
            t->m_shape[0], t->m_shape[1], t->m_shape[2], t->m_shape[3],
            cont);
    }
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
    
    m_ggml_fallback.reset();
    m_ggml_fallback_wsize = 0;

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

    // ---- setup reusable GGML fallback backend ----
    if (!m_ggml_fallback) {
        m_ggml_fallback = std::make_unique<powerserve::ggml::GGMLBackend>(m_llm, m_hparams);
        // GGMLBackend::matmul uses m_thread_pool->run(...), so threadpool must exist
        m_ggml_fallback->setup_threadpool();  // GGMLBackend::setup_threadpool() creates ThreadPool :contentReference[oaicite:4]{index=4}
    }
    
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

// contiguous check helpers
static inline powerserve::Stride make_contig_stride_bytes(const powerserve::Shape &shape, size_t elem) {
    powerserve::Stride s{};
    s[0] = elem;
    for (size_t i = 1; i < shape.size(); ++i) s[i] = s[i - 1] * shape[i - 1];
    return s;
}

bool OpenCLBackend::is_contiguous(const Tensor *tensor, int n) const {
    if (!tensor) return false;
    const size_t elem = powerserve::get_type_size(tensor->m_dtype);

    // Only care first n dims (caller decides), but we store strides for full rank=4
    const auto expected = make_contig_stride_bytes(tensor->m_shape, elem);

    try {
        const auto &buf = const_cast<Tensor*>(tensor)->get<OpenCLBuffer>();
        for (int i = 0; i < n; ++i) {
            if (buf.m_stride[i] != expected[i]) return false;
        }
        return true;
    } catch (...) {
        // CPU path (optional, keeps function generic)
        try {
            const auto &buf = const_cast<Tensor*>(tensor)->get<powerserve::CPUBuffer>();
            for (int i = 0; i < n; ++i) {
                if (buf.m_stride[i] != expected[i]) return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }
}

static inline void pack_contiguous_cpu_f32(
    powerserve::opencl::OpenCLBackend *self,
    const Tensor *src,
    Tensor *dst_contig_dev
) {
    POWERSERVE_ASSERT(self && src && dst_contig_dev);

    if (src->m_dtype != DataType::FP32 || dst_contig_dev->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("pack_contiguous_cpu_f32 only supports FP32 (got src={}, dst={})",
                             (int)src->m_dtype, (int)dst_contig_dev->m_dtype);
        return;
    }
    if (dst_contig_dev->m_shape != src->m_shape) {
        POWERSERVE_LOG_ERROR("pack_contiguous_cpu_f32 shape mismatch");
        return;
    }

    // ----------------------------
    // D2H: src -> host_src
    // ----------------------------
    Tensor host_src(DataType::FP32, src->m_shape);
    host_src.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);

    self->copy(&host_src, src);

    // Get src strides in bytes (from CPUBuffer after D2H copy)
    powerserve::CPUBuffer *src_cpu = nullptr;
    try {
        src_cpu = &host_src.get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("pack_contiguous_cpu_f32: host_src is not CPUBuffer? {}", e.what());
        return;
    }

    const auto s = src->m_shape;         // {ne0,ne1,ne2,ne3}
    const auto nb = src_cpu->m_stride;   // bytes stride for host_src

    const size_t ne0 = s[0], ne1 = s[1], ne2 = s[2], ne3 = s[3];
    const size_t elem = sizeof(float);

    // ----------------------------
    // CPU pack -> host_contig
    // ----------------------------
    Tensor host_contig(DataType::FP32, src->m_shape);
    host_contig.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);

    float *dst_ptr = static_cast<float *>(host_contig.get<powerserve::CPUBuffer>().m_data);
    const char *src_base = static_cast<const char *>(src_cpu->m_data);

    // destination is standard contiguous row-major in your stride convention:
    // linear index = (((i3*ne2 + i2)*ne1 + i1)*ne0 + i0)
    size_t out_idx = 0;
    for (size_t i3 = 0; i3 < ne3; ++i3) {
        for (size_t i2 = 0; i2 < ne2; ++i2) {
            for (size_t i1 = 0; i1 < ne1; ++i1) {
                // Pointer to the start of this (i1,i2,i3) slice in src
                const char *src_row = src_base
                    + (size_t)i3 * (size_t)nb[3]
                    + (size_t)i2 * (size_t)nb[2]
                    + (size_t)i1 * (size_t)nb[1];

                for (size_t i0 = 0; i0 < ne0; ++i0) {
                    const char *p = src_row + (size_t)i0 * (size_t)nb[0];
                    float v;
                    std::memcpy(&v, p, elem);
                    dst_ptr[out_idx++] = v;
                }
            }
        }
    }

    // ----------------------------
    // H2D: host_contig -> dst_contig_dev
    // ----------------------------
    self->copy(dst_contig_dev, &host_contig);
}

static inline const Tensor * ensure_contiguous_or_pack_f32(
    powerserve::opencl::OpenCLBackend *self,
    const Tensor *src,
    int n_dims_check,
    Tensor &tmp_dev
) {
    POWERSERVE_ASSERT(self && src);
    if (self->is_contiguous(src, n_dims_check)) {
        return src;
    }

    // Allocate a temporary OpenCLBuffer-backed tensor with same shape/dtype
    tmp_dev = Tensor(src->m_dtype, src->m_shape);
    tmp_dev.m_data = self->create_buffer(src->m_shape, src->m_dtype);
    if (!tmp_dev.m_data) {
        POWERSERVE_LOG_ERROR("ensure_contiguous: failed to allocate temp OpenCL buffer");
        return src; // fallback: return src, but caller should be aware this may break
    }

    pack_contiguous_cpu_f32(self, src, &tmp_dev);

    // Safety: packed result must be contiguous
    if (!self->is_contiguous(&tmp_dev, n_dims_check)) {
        POWERSERVE_LOG_ERROR("ensure_contiguous: pack produced non-contiguous tensor unexpectedly");
    }
    return &tmp_dev;
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

void OpenCLBackend::add_broadcast(Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    if (!initialized) {    
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    try {
        // 获取 OpenCL 缓冲区
        auto& src0_buffer = src0->get<OpenCLBuffer>();
        auto& src1_buffer = src1->get<OpenCLBuffer>();
        auto& dst_buffer = dst->get<OpenCLBuffer>();
        
        // 获取形状和步长信息
        Shape src0_shape = src0->m_shape;
        Shape src1_shape = src1->m_shape;
        Shape dst_shape = dst->m_shape;
        
        Stride src0_stride = src0_buffer.get_stride();
        Stride src1_stride = src1_buffer.get_stride();
        Stride dst_stride = dst_buffer.get_stride();
        
        // 参数命名参考 llama.cpp
        const int ne00 = static_cast<int>(src0_shape[0]);
        const int ne01 = static_cast<int>(src0_shape[1]);
        const int ne02 = static_cast<int>(src0_shape[2]);
        const int ne03 = static_cast<int>(src0_shape[3]);
        
        const int ne10 = static_cast<int>(src1_shape[0]);
        const int ne11 = static_cast<int>(src1_shape[1]);
        const int ne12 = static_cast<int>(src1_shape[2]);
        const int ne13 = static_cast<int>(src1_shape[3]);
        
        const int ne0 = static_cast<int>(dst_shape[0]);
        const int ne1 = static_cast<int>(dst_shape[1]);
        const int ne2 = static_cast<int>(dst_shape[2]);
        const int ne3 = static_cast<int>(dst_shape[3]);
        
        const cl_ulong nb00 = static_cast<cl_ulong>(src0_stride[0]);
        const cl_ulong nb01 = static_cast<cl_ulong>(src0_stride[1]);
        const cl_ulong nb02 = static_cast<cl_ulong>(src0_stride[2]);
        const cl_ulong nb03 = static_cast<cl_ulong>(src0_stride[3]);
        
        const cl_ulong nb10 = static_cast<cl_ulong>(src1_stride[0]);
        const cl_ulong nb11 = static_cast<cl_ulong>(src1_stride[1]);
        const cl_ulong nb12 = static_cast<cl_ulong>(src1_stride[2]);
        const cl_ulong nb13 = static_cast<cl_ulong>(src1_stride[3]);
        
        const cl_ulong nb0 = static_cast<cl_ulong>(dst_stride[0]);
        const cl_ulong nb1 = static_cast<cl_ulong>(dst_stride[1]);
        const cl_ulong nb2 = static_cast<cl_ulong>(dst_stride[2]);
        const cl_ulong nb3 = static_cast<cl_ulong>(dst_stride[3]);
        
        // 获取设备缓冲区
        cl_mem src0_data = src0_buffer.get_device_buffer();
        cl_mem src1_data = src1_buffer.get_device_buffer();
        cl_mem dst_data = dst_buffer.get_device_buffer();
        
        if (!src0_data || !src1_data || !dst_data) {
            POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for add");
            return;
        }
        
        bool bcast_row = false;
        if (src1_shape[0] == src0_shape[0] &&
            src1_shape[1] == 1 &&
            src1_shape[2] == 1 &&
            src1_shape[3] == 1 &&
            (ne00 % 4 == 0)) {

            // dim0 连续：nb10 == sizeof(float)
            // 注意：你的 stride 是 bytes
            const bool src1_contig_dim0 = (nb10 == sizeof(float));

            // 如果你支持 sub-buffer offset，最好还要求 offset1 16B 对齐
            const bool align_ok = true; // 你目前 offset1=0，天然对齐
            bcast_row = src1_contig_dim0 && align_ok;
        }
        
        // 选择正确的内核
        cl_kernel kernel = nullptr;
        std::string kernel_name;
        
        if (dst->m_dtype == DataType::FP32 && 
            src0->m_dtype == DataType::FP32 && 
            src1->m_dtype == DataType::FP32) {
            
            if (bcast_row) {
                kernel_name = "kernel_add_row";
                kernel = kernel_manager->get_kernel(kernel_name);
            } else {
                kernel_name = "kernel_add";
                kernel = kernel_manager->get_kernel(kernel_name);
            }
        } 
        // 可以后续添加 FP16 支持
        
        if (!kernel) {
            POWERSERVE_LOG_ERROR("Add kernel not found: {}", kernel_name);
            return;
        }
        
        // 设置内核参数
        cl_int err;
        cl_uint arg_index = 0;
        
        // 所有版本的通用偏移（目前都设为0）
        cl_ulong offset0 = 0;
        cl_ulong offset1 = 0;
        cl_ulong offsetd = 0;
        
        if (bcast_row) {
            // 行广播版本（7个参数）
            // kernel_add_row(src0, offset0, src1, offset1, dst, offsetd, ne)
            
            // 参数0: src0 buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
            CL_CHECK(err);
            
            // 参数1: offset0
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
            CL_CHECK(err);
            
            // 参数2: src1 buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);
            CL_CHECK(err);
            
            // 参数3: offset1
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);
            CL_CHECK(err);
            
            // 参数4: dst buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
            CL_CHECK(err);
            
            // 参数5: offsetd
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
            CL_CHECK(err);
            
            // 参数6: ne (元素数/4)
            int ne = ne00 / 4;
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne);
            CL_CHECK(err);
            
        } else {
            // 普通版本（30个参数，参考 llama.cpp）
            // kernel_add(src0, offset0, src1, offset1, dst, offsetd, 
            //            ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
            //            ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
            //            ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3)
            
            // 参数0-5: buffers 和 offsets
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);   // 0
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);   // 1
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);   // 2
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);   // 3
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);    // 4
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);   // 5
            CL_CHECK(err);
            
            // 参数6-9: ne00, ne01, ne02, ne03
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne00);  // 6
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne01);  // 7
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne02);  // 8
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne03);  // 9
            CL_CHECK(err);
            
            // 参数10-13: nb00, nb01, nb02, nb03
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb00);  // 10
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb01);  // 11
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb02);  // 12
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb03);  // 13
            CL_CHECK(err);
            
            // 参数14-17: ne10, ne11, ne12, ne13
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne10);  // 14
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne11);  // 15
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne12);  // 16
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne13);  // 17
            CL_CHECK(err);
            
            // 参数18-21: nb10, nb11, nb12, nb13
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb10);  // 18
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb11);  // 19
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb12);  // 20
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb13);  // 21
            CL_CHECK(err);
            
            // 参数22-25: ne0, ne1, ne2, ne3
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne0);  // 22
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne1);  // 23
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne2);  // 24
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne3);  // 25
            CL_CHECK(err);
            
            // 参数26-29: nb0, nb1, nb2, nb3
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb0);  // 26
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb1);  // 27
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb2);  // 28
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb3);  // 29
            CL_CHECK(err);
        }
        
        // 计算工作组大小（严格按照 llama.cpp 的逻辑）
        if (bcast_row) {
            // 行广播版本
            int n = dst->n_elements() / 4;  // 使用 float4
            size_t global_work_size[] = {static_cast<size_t>(n), 1, 1};
            size_t local_work_size[] = {64, 1, 1};
            
            err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                         1, nullptr, global_work_size, local_work_size,
                                         0, nullptr, nullptr);
            CL_CHECK(err);
            
        } else {
            // ✅ 普通版本：对齐 add.cl::kernel_add 的 get_group_id 语义
            const size_t nth = 64;              // local threads along dim0
            size_t local_work_size[3]  = { nth, 1, 1 };
            size_t global_work_size[3] = {
                static_cast<size_t>(ne01) * nth, // so get_group_id(0) ranges [0, ne01)
                static_cast<size_t>(ne02),
                static_cast<size_t>(ne03)
            };

            err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                        3, nullptr, global_work_size, local_work_size,
                                        0, nullptr, nullptr);
            CL_CHECK(err);
        }

        
        // 等待完成
        err = clFinish(context->get_queue());
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
            // 不返回，继续执行
        }
        
        
    } catch (const std::bad_cast& e) {
        POWERSERVE_LOG_ERROR("Invalid buffer type for add: {}", e.what());
    } catch (const std::exception& e) {
        POWERSERVE_LOG_ERROR("Exception in add: {}", e.what());
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

    // Phase1: FP32 only
    if (dst->m_dtype != DataType::FP32 ||
        src0->m_dtype != DataType::FP32 ||
        src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::add only supports FP32");
        return;
    }

    auto *self = const_cast<OpenCLBackend*>(this);

    // ✅ 快路径：完全同 shape → 你已有的 minimal kernel
    if (dst->m_shape == src0->m_shape && dst->m_shape == src1->m_shape) {
        Tensor tmp0, tmp1;
        const Tensor *src0_c = ensure_contiguous_or_pack_f32(self, src0, 4, tmp0);
        const Tensor *src1_c = ensure_contiguous_or_pack_f32(self, src1, 4, tmp1);
        self->add_minimal(const_cast<Tensor *>(dst), src0_c, src1_c);
        return;
    }

    self->add_broadcast(const_cast<Tensor *>(dst), src0, src1);
}

void OpenCLBackend::get_embedding(const Tensor *dst,
                                  const Tensor *weight,
                                  const std::vector<int> &tokens) const {
    // 1) basic checks
    if (dst->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding dst must be FP32");
        return;
    }

    auto dst_device = dynamic_cast<OpenCLBuffer *>(dst->m_data.get());
    if (!dst_device) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding dst must be OpenCLBuffer");
        return;
    }

    auto weight_host = dynamic_cast<CPUBuffer *>(weight->m_data.get());
    if (!weight_host) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding weight must be CPUBuffer");
        return;
    }

    const size_t dim = weight->m_shape[0];
    const size_t batch_size = tokens.size();

    // 2) ensure reusable ggml backend ready
    POWERSERVE_ASSERT(m_ggml_fallback && "m_ggml_fallback must be initialized in OpenCLBackend::initialize()");

    // optional: ensure minimum workspace
    constexpr size_t kMinWSize = 1 * 1024 * 1024;
    if (m_ggml_fallback_wsize < kMinWSize) {
        m_ggml_fallback->setup_work_data(kMinWSize);
        m_ggml_fallback_wsize = kMinWSize;
    }

    // 3) run ggml embedding on CPU
    Tensor host_tmp(DataType::FP32, dst->m_shape);
    host_tmp.m_data = CPUBuffer::create_buffer<float>(dst->m_shape);
    m_ggml_fallback->get_embedding(&host_tmp, weight, tokens);

    // 4) H2D copy
    this->copy(dst, &host_tmp);
}

// matmul cpu fallback helpers
static inline void cpu_gemm_f32_colmajorNK(
    float*       C,    // [N, M] but stored as your tensor layout {N, M}
    const float* A,    // {K, M}
    const float* B,    // {N, K}
    int K, int M, int N
) {
    // C(n,m) = sum_k B(n,k) * A(k,m)
    // This matches your matmul_minimal contract:
    // A shape {K,M}, B shape {N,K}, C shape {N,M} :contentReference[oaicite:7]{index=7}
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            const float* b_row = B + (size_t)n * (size_t)K;
            const float* a_col = A + (size_t)m * (size_t)K; // A is [K,M], contiguous in K
            for (int k = 0; k < K; ++k) {
                acc += b_row[k] * a_col[k];
            }
            C[(size_t)m * (size_t)N + (size_t)n] = acc;
        }
    }
}

static inline powerserve::BufferPtr create_cpu_buffer_for_dtype(powerserve::DataType dt, const powerserve::Shape &shape) {
    using powerserve::CPUBuffer;
    switch (dt) {
    case powerserve::DataType::FP32:
        return CPUBuffer::create_buffer<float>(shape);
    case powerserve::DataType::FP16:
        return CPUBuffer::create_buffer<uint16_t>(shape);
    case powerserve::DataType::INT32:
        return CPUBuffer::create_buffer<int32_t>(shape);
    case powerserve::DataType::INT64:
        return CPUBuffer::create_buffer<int64_t>(shape);
    default:
        POWERSERVE_ABORT("create_cpu_buffer_for_dtype: unsupported dtype {}", (int)dt);
    }
}

void OpenCLBackend::matmul_cpu_ggml_fallback(
    const Tensor *dst,
    const Tensor *src0,
    const Tensor *src1
) const {
    using powerserve::ggml::convert_to_ggml;

    // --- Reuse repo's existing pattern: dynamic_cast<CPUBuffer*> to check CPU-backed ---
    auto is_cpu_tensor = [](const Tensor *t) -> bool {
        return dynamic_cast<powerserve::CPUBuffer *>(t->m_data.get()) != nullptr;
    };

    // ---- 1) Prepare host views for src0/src1 ----
    const Tensor *a_host = src0;
    const Tensor *b_host = src1;

    Tensor host_a; // only used if src0 is not CPU-backed
    Tensor host_b; // only used if src1 is not CPU-backed

    // A: if not CPU, do D2H copy
    if (!is_cpu_tensor(src0)) {
        host_a = Tensor(src0->m_dtype, src0->m_shape);
        host_a.m_data = create_cpu_buffer_for_dtype(src0->m_dtype, src0->m_shape);
        this->copy(&host_a, src0);
        a_host = &host_a;
    }

    // B: if not CPU, do D2H copy
    // guard: quant tensor on device is not supported (since quant copy not implemented)
    if (!is_cpu_tensor(src1)) {
        if (src1->m_dtype == DataType::GGML_Q4_0 || src1->m_dtype == DataType::GGML_Q8_0) {
            POWERSERVE_ABORT(
                "matmul_cpu_ggml_fallback: quant tensor (dtype={}) is on device, but quant D2H copy not implemented",
                (int)src1->m_dtype
            );
        }
        host_b = Tensor(src1->m_dtype, src1->m_shape);
        host_b.m_data = create_cpu_buffer_for_dtype(src1->m_dtype, src1->m_shape);
        this->copy(&host_b, src1);
        b_host = &host_b;
    }

    // ---- 2) host output tensor (always CPU) ----
    Tensor host_c(dst->m_dtype, dst->m_shape);
    host_c.m_data = create_cpu_buffer_for_dtype(dst->m_dtype, dst->m_shape);

    // ---- 3) ggml mul_mat (reusable GGMLBackend, correct params/workspace/threadpool) ----
    POWERSERVE_ASSERT(m_ggml_fallback && "m_ggml_fallback must be initialized in OpenCLBackend::initialize()");

    // workspace sizing (copy from GGMLBackend::plan() logic for MAT_MUL) :contentReference[oaicite:6]{index=6}
    const enum ggml_type vec_dot_type = m_ggml_fallback->get_vec_dot_type(b_host);
    const enum ggml_type w_type = powerserve::ggml::convert_datatype_to_ggml(a_host->m_dtype);

    size_t required_wsize = 0;
    if (w_type != vec_dot_type) {
        required_wsize = ggml_row_size(vec_dot_type, a_host->n_elements());
    }

    // only grow, never shrink
    if (required_wsize > m_ggml_fallback_wsize) {
        m_ggml_fallback->setup_work_data(required_wsize); // will add cache line padding internally :contentReference[oaicite:7]{index=7}
        m_ggml_fallback_wsize = required_wsize;
    }

    // run ggml matmul
    m_ggml_fallback->matmul(&host_c, a_host, b_host);

    // ---- 4) H2D: host_c -> dst ----
    this->copy(dst, &host_c);
}



void OpenCLBackend::matmul_batched_cpu_f32_fallback(
    const Tensor *dst,
    const Tensor *src0,
    const Tensor *src1
) const {
    // ---- D2H: src0/src1 -> host tensors ----
    Tensor host_a(DataType::FP32, src0->m_shape);
    host_a.m_data = powerserve::CPUBuffer::create_buffer<float>(src0->m_shape);
    this->copy(&host_a, src0);

    Tensor host_b(DataType::FP32, src1->m_shape);
    host_b.m_data = powerserve::CPUBuffer::create_buffer<float>(src1->m_shape);
    this->copy(&host_b, src1);

    auto *a_host = static_cast<float *>(host_a.get<powerserve::CPUBuffer>().m_data);
    auto *b_host = static_cast<float *>(host_b.get<powerserve::CPUBuffer>().m_data);

    // ---- CPU output ----
    Tensor host_c(DataType::FP32, dst->m_shape);
    host_c.m_data = powerserve::CPUBuffer::create_buffer<float>(dst->m_shape);
    auto *c_host = static_cast<float *>(host_c.get<powerserve::CPUBuffer>().m_data);

    // Shapes (your contract):
    // A {K, M, H2, H3}
    // B {N, K, H2, H3}
    // C {N, M, H2, H3} :contentReference[oaicite:9]{index=9}
    const int K = (int)src0->m_shape[0];
    const int M = (int)src0->m_shape[1];
    const int N = (int)src1->m_shape[0];

    const int H2 = (int)dst->m_shape[2];
    const int H3 = (int)dst->m_shape[3];

    // Basic shape sanity
    if ((int)src1->m_shape[1] != K) {
        auto *self = const_cast<OpenCLBackend *>(this);
        log_tensor_meta(self, "dst(in)", dst);
        log_tensor_meta(self, "A(in)",   src0);
        log_tensor_meta(self, "B(in)",   src1);
        POWERSERVE_LOG_ERROR("matmul_batched_cpu_f32_fallback: B.shape[1]!=K");
        return;
    }
    if ((int)dst->m_shape[0] != N || (int)dst->m_shape[1] != M) {
        POWERSERVE_LOG_ERROR("matmul_batched_cpu_f32_fallback: C shape mismatch");
        return;
    }
    if ((int)src0->m_shape[2] != H2 || (int)src0->m_shape[3] != H3 ||
        (int)src1->m_shape[2] != H2 || (int)src1->m_shape[3] != H3) {
        POWERSERVE_LOG_ERROR("matmul_batched_cpu_f32_fallback: batch dims mismatch");
        return;
    }

    const size_t a_batch_elems = (size_t)K * (size_t)M;
    const size_t b_batch_elems = (size_t)N * (size_t)K;
    const size_t c_batch_elems = (size_t)N * (size_t)M;

    for (int i3 = 0; i3 < H3; ++i3) {
        for (int i2 = 0; i2 < H2; ++i2) {
            const size_t batch = (size_t)i3 * (size_t)H2 + (size_t)i2;

            const float* A = a_host + batch * a_batch_elems;
            const float* B = b_host + batch * b_batch_elems;
            float*       C = c_host + batch * c_batch_elems;

            cpu_gemm_f32_colmajorNK(C, A, B, K, M, N);
        }
    }

    // ---- H2D: host_c -> dst ----
    this->copy(dst, &host_c);

    POWERSERVE_LOG_DEBUG("OpenCLBackend::matmul batched CPU fallback done (K={},M={},N={},H2={},H3={})",
                         K, M, N, H2, H3);
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

    auto *self = const_cast<OpenCLBackend *>(this);

    // ---- Try OpenCL minimal kernel only when it's safe ----
    const bool all_f32 = (dst->m_dtype == DataType::FP32 &&
                          src0->m_dtype == DataType::FP32 &&
                          src1->m_dtype == DataType::FP32);

    if (all_f32) {
        Tensor tmpA_dev, tmpB_dev;
        const int n_dims_check = 4;

        const Tensor *A = ensure_contiguous_or_pack_f32(self, src0, n_dims_check, tmpA_dev);
        const Tensor *B = ensure_contiguous_or_pack_f32(self, src1, n_dims_check, tmpB_dev);

        // 2D-only kernel path
        if (A->m_shape[2] == 1 && A->m_shape[3] == 1 &&
            B->m_shape[2] == 1 && B->m_shape[3] == 1 &&
            dst->m_shape[2] == 1 && dst->m_shape[3] == 1) {

            const size_t K = A->m_shape[0];
            const size_t M = A->m_shape[1];
            const size_t N = B->m_shape[0];

            if (B->m_shape[1] == K &&
                dst->m_shape[0] == N && dst->m_shape[1] == M) {

                self->matmul_minimal(const_cast<Tensor *>(dst), A, B);
                return;
            }
        }
    }

    // ---- General fallback: ggml mul_mat (supports FP32/FP16/quant + arbitrary layout) ----
    self->matmul_cpu_ggml_fallback(dst, src0, src1);
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
}


void OpenCLBackend::softmax(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::softmax TODO");
}

void OpenCLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    // Old code aborts: :contentReference[oaicite:5]{index=5}
    if (!initialized) POWERSERVE_ABORT("OpenCL backend not initialized");
    if (!out || !x)   POWERSERVE_ABORT("permute got null tensor");

    // permute = view: share the same underlying buffer, only rewrite stride
    auto *dst = const_cast<Tensor*>(out);

    // require OpenCL buffers
    auto &xbuf = const_cast<Tensor*>(x)->get<OpenCLBuffer>();
    (void)xbuf.get_device_buffer(); // touch to validate

    // share underlying cl_mem (including sub-buffer views)
    dst->m_data = x->m_data;

    auto &obuf = dst->get<OpenCLBuffer>();
    // new_stride[i] = old_stride[axes[i]]  (same as ggml: :contentReference[oaicite:6]{index=6})
    Stride new_stride{};
    for (size_t i = 0; i < axes.size(); ++i) new_stride[i] = xbuf.m_stride[axes[i]];
    obuf.m_stride = new_stride;
}

void OpenCLBackend::cont(const Tensor *out, const Tensor *x) const {
    if (!initialized) POWERSERVE_ABORT("OpenCL backend not initialized");
    if (!out || !x)   POWERSERVE_ABORT("cont got null tensor");

    // If already contiguous, a plain copy is fine
    if (is_contiguous(x, 4)) {
        this->copy(out, x);
        return;
    }

    const size_t elem = powerserve::get_type_size(x->m_dtype);

    // D2H: read raw physical buffer bytes into host_in (contiguous physical)
    Tensor host_in(x->m_dtype, x->m_shape);
    Tensor host_out(x->m_dtype, x->m_shape);

    // allocate CPU buffers with correct element type size
    switch (x->m_dtype) {
    case DataType::FP32:
        host_in.m_data  = powerserve::CPUBuffer::create_buffer<float>(x->m_shape);
        host_out.m_data = powerserve::CPUBuffer::create_buffer<float>(x->m_shape);
        break;
    case DataType::FP16:
        host_in.m_data  = powerserve::CPUBuffer::create_buffer<uint16_t>(x->m_shape);
        host_out.m_data = powerserve::CPUBuffer::create_buffer<uint16_t>(x->m_shape);
        break;
    case DataType::INT32:
        host_in.m_data  = powerserve::CPUBuffer::create_buffer<int32_t>(x->m_shape);
        host_out.m_data = powerserve::CPUBuffer::create_buffer<int32_t>(x->m_shape);
        break;
    default:
        POWERSERVE_ABORT("cont: unsupported dtype={}", (int)x->m_dtype);
    }

    this->copy(&host_in, x);

    // Reorder: logical -> contiguous
    const auto &shape  = x->m_shape;
    const auto &stride = const_cast<Tensor*>(x)->get<OpenCLBuffer>().m_stride; // bytes

    const char *src = reinterpret_cast<const char*>(host_in.get<powerserve::CPUBuffer>().m_data);
    char *dst       = reinterpret_cast<char*>(host_out.get<powerserve::CPUBuffer>().m_data);

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0, ++idx) {
                    const size_t off = i0 * stride[0] + i1 * stride[1] + i2 * stride[2] + i3 * stride[3];
                    std::memcpy(dst + idx * elem, src + off, elem);
                }
            }
        }
    }

    // H2D
    this->copy(out, &host_out);
}

static inline uint32_t floor_log2_u32(uint32_t x) {
    // x>0
    uint32_t r = 0;
    while ((1u << (r + 1)) <= x) ++r;
    return r;
}

static void softmax_ext_cpu_f32_ggml_semantics(
    float *dst,            // contiguous, same shape as src0
    const float *src0,     // contiguous
    const float *src1,     // contiguous mask, expected shape [ne00, ne01, 1, 1]
    int ne00, int ne01, int ne02, int ne03,
    float scale,
    float max_bias
) {
    const uint32_t n_head = (uint32_t)ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)floor_log2_u32(n_head);

    const float m0 = std::pow(2.0f, -(max_bias)        / (float)n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int nc = ne00;
    const int nr = ne01 * ne02 * ne03;

    std::vector<float> wp((size_t)nc);

    for (int i1 = 0; i1 < nr; ++i1) {
        const uint32_t h = (uint32_t)((i1 / ne01) % ne02); // same as ggml: (i1/ne01)%ne02

        const float slope =
            (max_bias > 0.0f)
            ? (h < n_head_log2
                ? std::pow(m0, (float)(h + 1))
                : std::pow(m1, (float)(2*(h - n_head_log2) + 1)))
            : 1.0f;

        const float *sp = src0 + (size_t)i1 * (size_t)nc;
        float *dp       = dst  + (size_t)i1 * (size_t)nc;

        // broadcast mask across rows: row chosen by (i1 % ne01)
        const float *mp = src1 ? (src1 + (size_t)(i1 % ne01) * (size_t)ne00) : nullptr;

        // wp = sp; wp *= scale; wp += slope*mask
        for (int i = 0; i < nc; ++i) {
            float v = sp[i] * scale;
            if (mp) v += slope * mp[i];
            wp[i] = v;
        }

        // max
        float mx = -INFINITY;
        for (int i = 0; i < nc; ++i) mx = std::max(mx, wp[i]);

        // exp + sum (write exp to dp temporarily)
        float sum = 0.0f;
        for (int i = 0; i < nc; ++i) {
            float e = std::exp(wp[i] - mx);
            dp[i] = e;
            sum += e;
        }

        // normalize
        const float inv = 1.0f / sum;
        for (int i = 0; i < nc; ++i) {
            dp[i] *= inv;
        }
    }
}
void OpenCLBackend::softmax_ext(
    const Tensor *out,
    const Tensor *x,
    const Tensor *mask,
    float scale,
    float max_bias
) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    POWERSERVE_ASSERT(out && x && mask);

    if (out->m_dtype != DataType::FP32 || x->m_dtype != DataType::FP32 || mask->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::softmax_ext (Phase1) only supports FP32");
        return;
    }

    auto *self = const_cast<OpenCLBackend *>(this);

    // 0) ensure contiguous (strict: 4 dims)
    const int n_dims_check = 4;
    Tensor tmp_x_dev, tmp_mask_dev;
    const Tensor *x_dev    = ensure_contiguous_or_pack_f32(self, x,    n_dims_check, tmp_x_dev);
    const Tensor *m_dev    = ensure_contiguous_or_pack_f32(self, mask, n_dims_check, tmp_mask_dev);

    // shape locals
    const int ne00 = (int)x_dev->m_shape[0];
    const int ne01 = (int)x_dev->m_shape[1];
    const int ne02 = (int)x_dev->m_shape[2];
    const int ne03 = (int)x_dev->m_shape[3];

    // out must match x (ggml asserts src0 same shape as dst)
    if (out->m_shape != x_dev->m_shape) {
        POWERSERVE_LOG_ERROR("softmax_ext: out shape != x shape");
        return;
    }

    // ggml mask broadcast expects src1 laid out as [ne00, ne01] (and broadcast over head/batch)
    // i.e., shape [ne00, ne01, 1, 1]
    if (!(m_dev->m_shape[0] == x_dev->m_shape[0] &&
          m_dev->m_shape[1] == x_dev->m_shape[1] &&
          m_dev->m_shape[2] == 1 &&
          m_dev->m_shape[3] == 1)) {
        POWERSERVE_LOG_WARN(
            "softmax_ext: mask shape [{},{},{},{}] not [ne00,ne01,1,1]=[{},{},1,1]; "
            "ggml semantics will not match unless you feed that shape",
            (int)m_dev->m_shape[0], (int)m_dev->m_shape[1], (int)m_dev->m_shape[2], (int)m_dev->m_shape[3],
            ne00, ne01
        );
        return;
    }

    // 1) D2H
    Tensor host_x(DataType::FP32, x_dev->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(x_dev->m_shape);
    self->copy(&host_x, x_dev);

    Tensor host_m(DataType::FP32, m_dev->m_shape);
    host_m.m_data = powerserve::CPUBuffer::create_buffer<float>(m_dev->m_shape);
    self->copy(&host_m, m_dev);

    Tensor host_out(DataType::FP32, out->m_shape);
    host_out.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

    const float *x_buf = (const float *)host_x.get<CPUBuffer>().m_data;
    const float *m_buf = (const float *)host_m.get<CPUBuffer>().m_data;
    float *o_buf       = (float *)host_out.get<CPUBuffer>().m_data;

    // 2) CPU compute (ggml-aligned)
    softmax_ext_cpu_f32_ggml_semantics(
        o_buf, x_buf, m_buf,
        ne00, ne01, ne02, ne03,
        scale, max_bias
    );

    // 3) H2D
    self->copy(out, &host_out);
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
        return;
    }

    // CPU2CPU
    if (src_cpu && dst_cpu) {
        std::memcpy(dst_cpu->m_data, src_cpu->m_data, src_bytes);
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
