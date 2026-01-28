#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"              


#include "core/logger.hpp"
#include "ggml-quants.h"
#include "ggml.h"

#include <iostream>
#include <CL/cl.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <execinfo.h>
#include <fmt/core.h>
#include <mutex>
#include <cstdlib>   // getenv, atoi


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

static inline bool kv_dbg_enabled() {
    static int on = []() -> int {
        const char* e = std::getenv("POWERSERVE_KV_DBG");
        return e ? std::atoi(e) : 0;
    }();
    return on != 0;
}

static inline void log_tensor_meta(OpenCLBackend *self, const char *tag, const Tensor *t, int n = 4) {
    if (!t) {
        POWERSERVE_LOG_ERROR("[MATMUL][{}] <null>", tag);
        return;
    }
    bool cont = self->is_contiguous(t, n);

    powerserve::Stride st{};
    bool has_stride = false;
    BaseBuffer& base = const_cast<Tensor*>(t)->get<BaseBuffer>();
    if (auto *buf = dynamic_cast<OpenCLBuffer*>(&base)) {
        st = buf->m_stride;
        has_stride = true;
    } else if (auto *buf = dynamic_cast<powerserve::CPUBuffer*>(&base)) {
        st = buf->m_stride;
        has_stride = true;
    } else {
        has_stride = false;
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

// ---- ggml-compat helpers for quant nbytes/stride ----
static inline bool is_ggml_quant_dtype(powerserve::DataType dt) {
    using powerserve::DataType;
    return dt == DataType::GGML_Q4_0 || dt == DataType::GGML_Q8_0;
}

// contiguous logical tensor bytes in ggml layout
static inline size_t ggml_compat_nbytes(powerserve::DataType dt, const powerserve::Shape &s) {
    const size_t ne0 = (size_t)s[0];
    const size_t ne1 = (size_t)s[1];
    const size_t ne2 = (size_t)s[2];
    const size_t ne3 = (size_t)s[3];

    if (is_ggml_quant_dtype(dt)) {
        const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(dt);
        // ggml_nbytes for quant types is row_size(ne0) * ne1 * ne2 * ne3
        return (size_t)ggml_row_size(gt, (int64_t)ne0) * ne1 * ne2 * ne3;
    }

    const size_t elem = powerserve::get_type_size(dt); // fp32=4, fp16=2...
    POWERSERVE_ASSERT(elem > 0);
    return elem * ne0 * ne1 * ne2 * ne3;
}

static inline powerserve::Stride ggml_compat_contig_stride_bytes(powerserve::DataType dt, const powerserve::Shape &s) {
    powerserve::Stride st{};
    if (is_ggml_quant_dtype(dt)) {
        const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(dt);
        st[0] = (size_t)ggml_type_size(gt);                 // Q8_0 => 34
        st[1] = (size_t)ggml_row_size(gt, (int64_t)s[0]);   // Q8_0 row bytes
    } else {
        const size_t elem = powerserve::get_type_size(dt);
        POWERSERVE_ASSERT(elem > 0);
        st[0] = elem;
        st[1] = st[0] * (size_t)s[0];
    }
    st[2] = st[1] * (size_t)s[1];
    st[3] = st[2] * (size_t)s[2];
    return st;
}

static void ensure_ggml_global_init_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_init_params p{};
        p.mem_size   = 1024 * 1024; 
        p.mem_buffer = NULL;
        ggml_context* ctx = ggml_init(p);
        ggml_free(ctx);
    });
}


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
    
    ensure_ggml_global_init_once(); 

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
    options.enable_mad = false;
    options.unsafe_math = false;
    options.finite_math = false;
    options.fast_relaxed_math = false;
    
    POWERSERVE_LOG_INFO("[OpenCL] global compile_options: {}", options.to_string());

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

bool OpenCLBackend::is_contiguous(const Tensor *t, int n) const {
    if (!t) return false;
    if (n <= 0) return true;

    // 1) 计算 expected stride（单位：bytes）
    size_t expected[GGML_MAX_DIMS] = {0};

    // nb0：第 0 维的“步长”= 一个元素/一个block的字节数
    size_t nb0 = 0;
    bool is_quant = (t->m_dtype == DataType::GGML_Q4_0 || t->m_dtype == DataType::GGML_Q8_0);

    if (is_quant) {
        const enum ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(t->m_dtype);
        nb0 = (size_t)ggml_type_size(gt);               // e.g. Q8_0: 34
        expected[0] = nb0;
        if (n > 1) {
            expected[1] = (size_t)ggml_row_size(gt, (int64_t)t->m_shape[0]); // nb1（关键）
        }
    } else {
        nb0 = powerserve::get_type_size(t->m_dtype);    // fp32=4, fp16=2...
        expected[0] = nb0;
        if (n > 1) {
            expected[1] = expected[0] * (size_t)t->m_shape[0];
        }
    }

    // 2) 维度 >=2 的 stride 都是前一维 stride * 前一维长度（循环形式）
    for (int i = 2; i < n; ++i) {
        expected[i] = expected[i - 1] * (size_t)t->m_shape[i - 1];
    }

    // 3) 取出实际 stride 并比较
    const size_t *actual = nullptr;
    BaseBuffer& base = const_cast<Tensor*>(t)->get<BaseBuffer>();
    if (auto *buf = dynamic_cast<OpenCLBuffer*>(&base)) {
        actual = buf->m_stride.data();
    } else if (auto *buf = dynamic_cast<powerserve::CPUBuffer*>(&base)) {
        actual = buf->m_stride.data();
    } else {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        if ((size_t)actual[i] != expected[i]) return false;
    }
    return true;
}

static inline void cpy_tensor_cl(const OpenCLBackend* self,
                                 const Tensor* src,
                                 const Tensor* dst) {
    POWERSERVE_ASSERT(self && src && dst);

    // CL_CHECK requires a variable named `context` in scope
    auto* context = self->context.get();
    POWERSERVE_ASSERT(context != nullptr);

    // Must be OpenCL tensors
    // NOTE: Tensor::get<BaseBuffer>() is non-const in this repo, so we const_cast locally.
    auto* src_cl = dynamic_cast<OpenCLBuffer*>(
        &const_cast<Tensor*>(src)->get<BaseBuffer>());

    auto* dst_cl = dynamic_cast<OpenCLBuffer*>(
        &const_cast<Tensor*>(dst)->get<BaseBuffer>());

    if (!src_cl || !dst_cl) {
        POWERSERVE_LOG_ERROR("cpy_tensor_cl: src/dst not OpenCLBuffer");
        return;
    }

    // Same logical shape required
    if (src->m_shape != dst->m_shape) {
        POWERSERVE_ABORT(
            "cpy_tensor_cl: shape mismatch src=[{},{},{},{}] dst=[{},{},{},{}]",
            src->m_shape[0], src->m_shape[1], src->m_shape[2], src->m_shape[3],
            dst->m_shape[0], dst->m_shape[1], dst->m_shape[2], dst->m_shape[3]
        );
    }


    // Kernel dispatch by dtype pair
    cl_kernel k = self->kernel_manager->get_cpy_kernel(src->m_dtype, dst->m_dtype);
    if (!k) {
        POWERSERVE_LOG_ERROR("cpy_tensor_cl: unsupported dtype pair src={} dst={}",
                             (int)src->m_dtype, (int)dst->m_dtype);
        return;
    }

    cl_mem src_mem = src_cl->get_device_buffer();
    cl_mem dst_mem = dst_cl->get_device_buffer();
    if (!src_mem || !dst_mem) {
        POWERSERVE_LOG_ERROR("cpy_tensor_cl: invalid cl_mem");
        return;
    }

    // shape (4D)
    const int ne00 = (int)src->m_shape[0];
    const int ne01 = (int)src->m_shape[1];
    const int ne02 = (int)src->m_shape[2];
    const int ne03 = (int)src->m_shape[3];

    const int ne0  = (int)dst->m_shape[0];
    const int ne1  = (int)dst->m_shape[1];
    const int ne2  = (int)dst->m_shape[2];
    const int ne3  = (int)dst->m_shape[3];

    // strides in bytes (from OpenCLBuffer stride metadata)
    const auto sst = src_cl->get_stride();
    const cl_ulong nb00 = (cl_ulong)sst[0];
    const cl_ulong nb01 = (cl_ulong)sst[1];
    const cl_ulong nb02 = (cl_ulong)sst[2];
    const cl_ulong nb03 = (cl_ulong)sst[3];

    const auto dstst = dst_cl->get_stride();
    const cl_ulong nb0 = (cl_ulong)dstst[0];
    const cl_ulong nb1 = (cl_ulong)dstst[1];
    const cl_ulong nb2 = (cl_ulong)dstst[2];
    const cl_ulong nb3 = (cl_ulong)dstst[3];

    // Offsets:
    // Scheme-B: views are NOT sub-buffers, so we must pass base offsets explicitly.
    const cl_ulong off0 = (cl_ulong)src_cl->get_base_offset();
    const cl_ulong offd = (cl_ulong)dst_cl->get_base_offset();

    cl_uint arg = 0;
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_mem),   &src_mem));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &off0));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_mem),   &dst_mem));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &offd));

    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb03));

    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb3));

    // Work sizes:
    // This matches llama.cpp's cpy.cl style where each work-item handles (i1,i2,i3)
    // and the kernel loops over i0 internally.
    const size_t local[3]  = { 1, 1, 1 };
    const size_t global[3] = { (size_t)ne01, (size_t)ne02, (size_t)ne03 };

    CL_CHECK(clEnqueueNDRangeKernel(self->context->get_queue(),
                                    k, 3, nullptr, global, local,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(self->context->get_queue()));
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

    cpy_tensor_cl(self, src, &tmp_dev);

    // Safety: packed result must be contiguous
    if (!self->is_contiguous(&tmp_dev, n_dims_check)) {
        POWERSERVE_LOG_ERROR("ensure_contiguous: pack produced non-contiguous tensor unexpectedly");
    }
    return &tmp_dev;
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

    // 1) 先在 device 上把 non-contig view pack 成 contig（stride-aware）
    Tensor tmp_dev;
    const Tensor *src_packed = ensure_contiguous_or_pack_f32(self, src, /*n_dims_check=*/4, tmp_dev);

    // 2) D2H：读回已经连续的 src_packed
    Tensor host_contig(DataType::FP32, src->m_shape);
    host_contig.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
    self->copy(&host_contig, src_packed);

    // 3) H2D：写入 dst_contig_dev（连续）
    self->copy(dst_contig_dev, &host_contig);
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

    // Scheme-B: pass base offsets explicitly (bytes)
    const cl_ulong off0 = (cl_ulong)src0->get<OpenCLBuffer>().get_base_offset();
    const cl_ulong off1 = (cl_ulong)src1->get<OpenCLBuffer>().get_base_offset();
    const cl_ulong offd = (cl_ulong)dst ->get<OpenCLBuffer>().get_base_offset();

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg a failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off0);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg off0 failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg b failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off1);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg off1 failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &o);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg out failed"); return; }

    err = clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offd);
    if (err != CL_SUCCESS) { POWERSERVE_LOG_ERROR("set arg offd failed"); return; }

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
        
        cl_ulong offset0 = (cl_ulong)src0_buffer.get_base_offset();
        cl_ulong offset1 = (cl_ulong)src1_buffer.get_base_offset();
        cl_ulong offsetd = (cl_ulong)dst_buffer.get_base_offset();

        
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
        
        if (bcast_row) {
            // 行广播版本 bring-up：local=1，保证合法
            int n = dst->n_elements() / 4;
            if (n <= 0) return;

            size_t global_work_size[] = { static_cast<size_t>(n), 1, 1 };
            size_t local_work_size[]  = { 1, 1, 1 };

            err = clEnqueueNDRangeKernel(
                context->get_queue(),
                kernel,
                1,
                nullptr,
                global_work_size,
                local_work_size,
                0,
                nullptr,
                nullptr
            );
            CL_CHECK(err);

        } else {
            // 普通版本 bring-up：local=1，保证合法
            if (ne01 <= 0 || ne02 <= 0 || ne03 <= 0) return;

            size_t global_work_size[3] = {
                static_cast<size_t>(ne01),   // 注意：local=1 时 global 直接等于 group-count
                static_cast<size_t>(ne02),
                static_cast<size_t>(ne03)
            };
            size_t local_work_size[3]  = { 1, 1, 1 };

            err = clEnqueueNDRangeKernel(
                context->get_queue(),
                kernel,
                3,
                nullptr,
                global_work_size,
                local_work_size,
                0,
                nullptr,
                nullptr
            );
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

    // IMPORTANT:
    // In this repo, matmul fallback must keep operand order as (A=weight, B=activation).
    // Swapping them triggers ggml.c assertion (GGML_ASSERT(ne0 == ne01)).
    const Tensor *w_host = a_host; // weight (often quant)
    const Tensor *x_host = b_host; // activation (usually FP32)

    // ---- shape sanity check (fail fast with readable logs instead of ggml assert) ----
    // Expected (ggml-style) here: w:[K,N], x:[K,M]  => dst:[N,M]
    const int64_t K_w = (int64_t)w_host->m_shape[0];
    const int64_t N_w = (int64_t)w_host->m_shape[1];
    const int64_t K_x = (int64_t)x_host->m_shape[0];
    const int64_t M_x = (int64_t)x_host->m_shape[1];

    const int64_t N_dst = (int64_t)dst->m_shape[0];
    const int64_t M_dst = (int64_t)dst->m_shape[1];

    if (!(K_w == K_x && N_w == N_dst && M_x == M_dst)) {
        POWERSERVE_LOG_ERROR(
            "matmul_cpu_ggml_fallback shape mismatch: "
            "w=[K={},N={}] x=[K={},M={}] dst=[N={},M={}]",
            (long long)K_w, (long long)N_w,
            (long long)K_x, (long long)M_x,
            (long long)N_dst, (long long)M_dst
        );
        POWERSERVE_ABORT("matmul_cpu_ggml_fallback: abort due to incompatible shapes (would trigger ggml assert)");
    }

    // ---- workspace sizing ----
    // Your tests allocate at least sizeof(float) * (K + 64) * n_threads for ggml ops
    // (same idea as ggml's scratch usage). See tests/opencl_test.cpp. :contentReference[oaicite:0]{index=0}
    const size_t n_threads = (size_t)m_hparams.n_threads;
    size_t required_wsize = sizeof(float) * (size_t)(K_w + 64) * n_threads;

    // Also keep your original "type conversion" workspace logic (only increases, never hurts)
    {
        const enum ggml_type vec_dot_type = m_ggml_fallback->get_vec_dot_type(x_host);
        const enum ggml_type w_type       = powerserve::ggml::convert_datatype_to_ggml(w_host->m_dtype);
        if (w_type != vec_dot_type) {
            const size_t extra = (size_t)ggml_row_size(vec_dot_type, (int64_t)w_host->n_elements());
            required_wsize = std::max(required_wsize, extra);
        }
    }

    // only grow, never shrink
    if (required_wsize > m_ggml_fallback_wsize) {
        m_ggml_fallback->setup_work_data(required_wsize);
        m_ggml_fallback_wsize = required_wsize;
    }

    // run ggml matmul (KEEP ORDER: weight first, activation second)
    m_ggml_fallback->matmul(&host_c, w_host, x_host);

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

    // --------------------------------------------
    // Inline helper: check if tensor holds OpenCLBuffer
    // --------------------------------------------
    auto is_opencl = [](const Tensor *t) -> bool {
#if defined(POWERSERVE_WITH_OPENCL)
        return t && t->m_data &&
               dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get()) != nullptr;
#else
        (void)t;
        return false;
#endif
    };

    // ---- Try OpenCL minimal kernel only when it's safe ----
    const bool all_f32 = (dst->m_dtype  == DataType::FP32 &&
                          src0->m_dtype == DataType::FP32 &&
                          src1->m_dtype == DataType::FP32);

    const bool all_opencl = is_opencl(dst) && is_opencl(src0) && is_opencl(src1);

    if (all_f32 && all_opencl) {
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
    // Important: ggml fallback requires CPU tensors.
    // We must materialize OpenCL inputs to CPU, and if dst is OpenCL, copy the result back.

    const Tensor *A_cpu = src0;  // src0 may be CPU quant weight (dtype=6), keep as-is
    const Tensor *B_cpu = src1;

    Tensor tmpB_cpu;
    if (is_opencl(src1)) {
        // D2H src1 into tmpB_cpu (FP32)
        tmpB_cpu = Tensor(DataType::FP32, src1->m_shape);
        tmpB_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(src1->m_shape);
        self->copy(&tmpB_cpu, src1);
        B_cpu = &tmpB_cpu;
    }

    // Run ggml fallback matmul on CPU
    self->matmul_cpu_ggml_fallback(dst, A_cpu, B_cpu);
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
    if (!initialized || !m_ggml_fallback) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm not ready");
        return;
    }
    if (!o || !x || !weight) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm got null tensor");
        return;
    }
    if (o->m_dtype != DataType::FP32 || x->m_dtype != DataType::FP32 || weight->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm strict only supports FP32");
        return;
    }
    if (o->m_shape != x->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm requires o.shape == x.shape");
        return;
    }

    // 1) D2H: x -> host_x
    Tensor host_x(DataType::FP32, x->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(x->m_shape);
    this->copy(&host_x, x);

    // 2) 确保 weight 在 CPU；如果不是，就 D2H 一份
    const Tensor *host_w_ptr = weight;
    Tensor host_w;
    try {
        (void)const_cast<Tensor*>(weight)->get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &) {
        host_w = Tensor(DataType::FP32, weight->m_shape);
        host_w.m_data = powerserve::CPUBuffer::create_buffer<float>(weight->m_shape);
        this->copy(&host_w, weight);
        host_w_ptr = &host_w;
    }

    // 3) GGML 计算
    Tensor host_y(DataType::FP32, o->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(o->m_shape);
    m_ggml_fallback->rmsnorm(&host_y, &host_x, host_w_ptr, eps);

    // 4) H2D: host_y -> o
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
    if (!m_ggml_fallback) {
        POWERSERVE_LOG_ERROR("m_ggml_fallback is null (initialize() not called?)");
        return;
    }
    if (!out || !src) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope got null tensor");
        return;
    }

    // strict：先只支持 FP32 + same shape（和你原来一致）
    if (out->m_dtype != DataType::FP32 || src->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope fallback only supports FP32");
        return;
    }
    if (out->m_shape != src->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope requires out.shape == src.shape");
        return;
    }

    // 1) D2H: src -> host_x (CPUBuffer)
    Tensor host_x(DataType::FP32, src->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
    this->copy(&host_x, src);

    // 2) CPU: 用 GGML 同源实现算 rope
    Tensor host_y(DataType::FP32, out->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

    m_ggml_fallback->rope(&host_y, &host_x, pos, rope_cfg);

    // 3) H2D: host_y -> out
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
        POWERSERVE_ABORT("OpenCL backend not initialized");
    }
    if (!out || !hb || !hb2) {
        POWERSERVE_ABORT("silu_hadamard got null tensor");
    }

    // dtype hard constraint (minimal kernel)
    if (out->m_dtype != DataType::FP32 ||
        hb->m_dtype  != DataType::FP32 ||
        hb2->m_dtype != DataType::FP32) {
        POWERSERVE_ABORT("silu_hadamard only supports FP32");
    }

    // shape hard constraint
    if (out->m_shape != hb->m_shape || out->m_shape != hb2->m_shape) {
        POWERSERVE_ABORT("silu_hadamard requires same shape");
    }

    // ---- NEW: layout hard constraints (match GGML preconditions) ----
    // 1) contiguous
    POWERSERVE_ASSERT(is_contiguous(out, 0));
    POWERSERVE_ASSERT(is_contiguous(hb, 0));
    POWERSERVE_ASSERT(is_contiguous(hb2, 0));

    const size_t n = out->n_elements();
    if (n == 0) return;

    // DEBUG: force CPU fallback for SILU_HADAMARD
    // ============================
    {
        // 1) Prepare CPU tensors (contiguous FP32)
        Tensor hb_cpu(DataType::FP32, hb->m_shape);
        hb_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(hb->m_shape);

        Tensor hb2_cpu(DataType::FP32, hb2->m_shape);
        hb2_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(hb2->m_shape);

        Tensor out_cpu(DataType::FP32, out->m_shape);
        out_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

        // 2) D2H: copy OpenCL -> CPU
        // (OpenCLBackend::copy already supports CPU<->OpenCL in this repo’s flow)
        this->copy(&hb_cpu, hb);
        this->copy(&hb2_cpu, hb2);

        // 3) Compute on CPU using GGML ref implementation if available
        if (m_ggml_fallback) {
            m_ggml_fallback->silu_hadamard(&out_cpu, &hb_cpu, &hb2_cpu);
        } else {
            // Fallback to the same formula as GGMLBackend::silu_hadamard
            float *out_data = static_cast<float *>(out_cpu.get<CPUBuffer>().m_data);
            float *hb_data  = static_cast<float *>(hb_cpu.get<CPUBuffer>().m_data);
            float *hb2_data = static_cast<float *>(hb2_cpu.get<CPUBuffer>().m_data);
            for (size_t j = 0; j < hb_cpu.n_elements(); ++j) {
                float val = hb_data[j];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= hb2_data[j];
                out_data[j] = val;
            }
        }

        // 4) H2D: copy CPU -> OpenCL output
        this->copy(out, &out_cpu);
        return;
    }


    cl_mem a = nullptr;
    cl_mem b = nullptr;
    cl_mem o = nullptr;
    try {
        a = hb ->get<OpenCLBuffer>().get_device_buffer();
        b = hb2->get<OpenCLBuffer>().get_device_buffer();
        o = out->get<OpenCLBuffer>().get_device_buffer();
    } catch (const std::bad_cast & e) {
        POWERSERVE_ABORT("silu_hadamard expects OpenCLBuffer: {}", e.what());
    }

    if (!a || !b || !o) {
        POWERSERVE_ABORT("silu_hadamard invalid cl_mem");
    }

    cl_kernel kernel = kernel_manager->get_kernel("kernel_silu_hadamard_contig_f32");
    if (!kernel) {
        POWERSERVE_ABORT("kernel not found: kernel_silu_hadamard_contig_f32");
    }

    cl_int err = CL_SUCCESS;
    cl_uint idx = 0;

    // also consider uint to avoid int overflow long term
    const cl_uint n_u = (cl_uint)n;

    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &a); if (err != CL_SUCCESS) POWERSERVE_ABORT("set arg hb failed");
    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &b); if (err != CL_SUCCESS) POWERSERVE_ABORT("set arg hb2 failed");
    err = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &o); if (err != CL_SUCCESS) POWERSERVE_ABORT("set arg out failed");
    err = clSetKernelArg(kernel, idx++, sizeof(cl_uint), &n_u); if (err != CL_SUCCESS) POWERSERVE_ABORT("set arg n failed");

    const size_t local = 256;
    const size_t global = round_up(n, local);

    cl_command_queue q = context->get_queue();
    err = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_ABORT("clEnqueueNDRangeKernel failed: {}", context->get_error_string(err));
    }

    err = clFinish(q);
    if (err != CL_SUCCESS) {
        POWERSERVE_ABORT("clFinish failed: {}", context->get_error_string(err));
    }
}


// for copy
static inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        default: return 0;
    }
}

static inline size_t numel_4d(const Tensor* t) {
    size_t n = 1;
    // 按你们 shape 维度数改；你摘要里是 4 维
    for (int i = 0; i < 4; ++i) n *= static_cast<size_t>(t->m_shape[i]);
    return n;
}

static inline bool is_cpy_kernel_supported(powerserve::DataType t) {
    return t == DataType::FP16 || t == DataType::FP32;
}

static inline bool pack_cpu_strided_to_contig(const Tensor* src, void* dst_contig) {
    POWERSERVE_ASSERT(src != nullptr);
    POWERSERVE_ASSERT(dst_contig != nullptr);

    const size_t elem = dtype_size(src->m_dtype);
    if (elem == 0) {
        POWERSERVE_LOG_ERROR("pack_cpu_strided_to_contig: unsupported dtype {}", (int)src->m_dtype);
        return false;
    }

    powerserve::CPUBuffer* src_cpu = nullptr;
    try {
        src_cpu = &const_cast<Tensor*>(src)->get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("pack_cpu_strided_to_contig: src is not CPUBuffer? {}", e.what());
        return false;
    }

    const auto s = src->m_shape;
    const auto nb = src_cpu->m_stride;
    const size_t ne0 = s[0], ne1 = s[1], ne2 = s[2], ne3 = s[3];

    const char* src_base = static_cast<const char*>(src_cpu->m_data);
    char* dst_ptr = static_cast<char*>(dst_contig);

    size_t out_idx = 0;
    for (size_t i3 = 0; i3 < ne3; ++i3) {
        for (size_t i2 = 0; i2 < ne2; ++i2) {
            for (size_t i1 = 0; i1 < ne1; ++i1) {
                const char* src_row = src_base
                    + (size_t)i3 * (size_t)nb[3]
                    + (size_t)i2 * (size_t)nb[2]
                    + (size_t)i1 * (size_t)nb[1];

                for (size_t i0 = 0; i0 < ne0; ++i0) {
                    const char* p = src_row + (size_t)i0 * (size_t)nb[0];
                    std::memcpy(dst_ptr + out_idx * elem, p, elem);
                    ++out_idx;
                }
            }
        }
    }

    return true;
}

static inline bool unpack_contig_to_cpu_strided(const void* src_contig, const Tensor* dst) {
    POWERSERVE_ASSERT(src_contig != nullptr);
    POWERSERVE_ASSERT(dst != nullptr);

    const size_t elem = dtype_size(dst->m_dtype);
    if (elem == 0) {
        POWERSERVE_LOG_ERROR("unpack_contig_to_cpu_strided: unsupported dtype {}", (int)dst->m_dtype);
        return false;
    }

    powerserve::CPUBuffer* dst_cpu = nullptr;
    try {
        dst_cpu = &const_cast<Tensor*>(dst)->get<powerserve::CPUBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("unpack_contig_to_cpu_strided: dst is not CPUBuffer? {}", e.what());
        return false;
    }

    const auto s = dst->m_shape;
    const auto nb = dst_cpu->m_stride;
    const size_t ne0 = s[0], ne1 = s[1], ne2 = s[2], ne3 = s[3];

    const char* src_ptr = static_cast<const char*>(src_contig);
    char* dst_base = static_cast<char*>(dst_cpu->m_data);

    size_t in_idx = 0;
    for (size_t i3 = 0; i3 < ne3; ++i3) {
        for (size_t i2 = 0; i2 < ne2; ++i2) {
            for (size_t i1 = 0; i1 < ne1; ++i1) {
                char* dst_row = dst_base
                    + (size_t)i3 * (size_t)nb[3]
                    + (size_t)i2 * (size_t)nb[2]
                    + (size_t)i1 * (size_t)nb[1];

                for (size_t i0 = 0; i0 < ne0; ++i0) {
                    char* p = dst_row + (size_t)i0 * (size_t)nb[0];
                    std::memcpy(p, src_ptr + in_idx * elem, elem);
                    ++in_idx;
                }
            }
        }
    }

    return true;
}

static inline Tensor make_contig_dev_tensor(
    OpenCLBackend* self,
    powerserve::DataType dtype,
    const powerserve::Shape& shape
) {
    Tensor t(dtype, shape);
    auto buf = self->create_buffer(shape, dtype);
    t.m_data = std::static_pointer_cast<BaseBuffer>(buf);
    return t;
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

    const size_t src_bytes = ggml_compat_nbytes(src->m_dtype, src->m_shape);
    const size_t dst_bytes = ggml_compat_nbytes(dst->m_dtype, dst->m_shape);


    if (src_bytes == 0 || dst_bytes == 0 || src_bytes != dst_bytes) {
        POWERSERVE_LOG_ERROR("copy: size mismatch src_bytes={} dst_bytes={}", src_bytes, dst_bytes);
        return;
    }

    const bool shape_match = src->m_shape == dst->m_shape;
    if (!shape_match) {
        const bool src_contig = is_contiguous(src, 4);
        const bool dst_contig = is_contiguous(dst, 4);

        if (!src_contig || !dst_contig) {
            POWERSERVE_LOG_ERROR("copy: shape mismatch with non-contiguous src/dst is unsupported");
            return;
        }

        BaseBuffer& src_base = src->get<BaseBuffer>();
        BaseBuffer& dst_base = dst->get<BaseBuffer>();

        auto* src_cpu = dynamic_cast<powerserve::CPUBuffer*>(&src_base);
        auto* dst_cpu = dynamic_cast<powerserve::CPUBuffer*>(&dst_base);
        auto* src_cl  = dynamic_cast<OpenCLBuffer*>(&src_base);
        auto* dst_cl  = dynamic_cast<OpenCLBuffer*>(&dst_base);

        if (src_cpu && dst_cpu) {
            std::memcpy(dst_cpu->m_data, src_cpu->m_data, src_bytes);
            return;
        }

        if (src_cpu && dst_cl) {
            cl_mem dev = dst_cl->get_device_buffer();
            if (!dev || !src_cpu->m_data) {
                POWERSERVE_LOG_ERROR("copy: invalid host/dev for shape-mismatch H2D");
                return;
            }
            const size_t dst_off = dst_cl->get_base_offset();
            if (!memory_pool->copy_host_to_device(dev, src_cpu->m_data, src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("copy: shape-mismatch H2D copy_host_to_device failed");
            }
            clFinish(context->get_queue());
            return;
        }

        if (src_cl && dst_cpu) {
            cl_mem dev = src_cl->get_device_buffer();
            if (!dev || !dst_cpu->m_data) {
                POWERSERVE_LOG_ERROR("copy: invalid host/dev for shape-mismatch D2H");
                return;
            }
            const size_t src_off = src_cl->get_base_offset();
            if (!memory_pool->copy_device_to_host(dst_cpu->m_data, dev, src_bytes, src_off)) {
                POWERSERVE_LOG_ERROR("copy: shape-mismatch D2H copy_device_to_host failed");
            }
            return;
        }

        if (src_cl && dst_cl) {
            cl_mem src_dev = src_cl->get_device_buffer();
            cl_mem dst_dev = dst_cl->get_device_buffer();
            if (!src_dev || !dst_dev) {
                POWERSERVE_LOG_ERROR("copy: invalid cl_mem for shape-mismatch D2D");
                return;
            }

            const size_t src_off = src_cl->get_base_offset();
            const size_t dst_off = dst_cl->get_base_offset();
            if (src_off == 0 && dst_off == 0) {
                if (!memory_pool->copy_device_to_device(dst_dev, src_dev, src_bytes)) {
                    POWERSERVE_LOG_ERROR("copy: shape-mismatch D2D copy_device_to_device failed");
                }
                return;
            }

            std::vector<uint8_t> host(src_bytes);
            if (!memory_pool->copy_device_to_host(host.data(), src_dev, src_bytes, src_off)) {
                POWERSERVE_LOG_ERROR("copy: shape-mismatch D2H staging failed");
                return;
            }
            if (!memory_pool->copy_host_to_device(dst_dev, host.data(), src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("copy: shape-mismatch H2D staging failed");
            }
            clFinish(context->get_queue());
            return;
        }

        POWERSERVE_LOG_ERROR("copy: shape mismatch with unsupported buffer types");
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

        const bool src_contig = is_contiguous(src, 4);
        const bool dst_contig = is_contiguous(dst, 4);

        // Fast-path: contiguous dst -> linear write
        if (src_contig && dst_contig) {
            const size_t dst_off = dst_cl->get_base_offset();  // bytes
            if (!memory_pool->copy_host_to_device(dev, host, src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("H2D: copy_host_to_device failed");
            }
            clFinish(context->get_queue());
            return;
        }


        std::vector<uint8_t> host_contig;
        const void* host_src = host;
        if (!src_contig) {
            host_contig.resize(src_bytes);
            if (!pack_cpu_strided_to_contig(src, host_contig.data())) {
                POWERSERVE_LOG_ERROR("H2D: failed to pack CPU strided tensor");
                return;
            }
            host_src = host_contig.data();
        }

        if (dst_contig) {
            const size_t dst_off = dst_cl->get_base_offset();  // bytes
            if (!memory_pool->copy_host_to_device(dev, host_src, src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("H2D: copy_host_to_device failed");
            }
            clFinish(context->get_queue());
            return;
        }

        // Slow-path: non-contiguous dst -> write to staging contig -> scatter on device
        if (!is_cpy_kernel_supported(src->m_dtype) || !is_cpy_kernel_supported(dst->m_dtype)) {
            POWERSERVE_LOG_ERROR("H2D: non-contiguous copy requires cpy kernel, unsupported dtype src={} dst={}",
                                (int)src->m_dtype, (int)dst->m_dtype);
            return;
        }

        Tensor staging = make_contig_dev_tensor(const_cast<OpenCLBackend*>(this),
                                        dst->m_dtype,
                                        dst->m_shape);

        // 1) write host -> staging (staging contiguous so raw write ok)
        auto &staging_buf = staging.get<OpenCLBuffer>();
        cl_mem staging_mem = staging_buf.get_device_buffer();
        const size_t st_off = staging_buf.get_base_offset();
        if (!memory_pool->copy_host_to_device(staging_mem, host_src, src_bytes, st_off)) {
            POWERSERVE_LOG_ERROR("H2D: staging copy_host_to_device failed");
            dump_backtrace();
            std::abort();
        }

        // 2) scatter staging(contig) -> dst(strided) using cpy kernel
        // so for scatter we must call the same kernel but swap args:
        // dst is "dst", staging is "src".
        cpy_tensor_cl(this, &staging, dst);
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

        const bool src_contig = is_contiguous(src, 4);
        const bool dst_contig = is_contiguous(dst, 4);

        // Slow-path: non-contiguous src -> pack on device -> linear read
        Tensor staging;
        const Tensor* read_src = src;
        if (!src_contig) {
            if (!is_cpy_kernel_supported(src->m_dtype)) {
                POWERSERVE_LOG_ERROR("D2H: non-contiguous copy requires cpy kernel, unsupported dtype={}",
                                    (int)src->m_dtype);
                return;
            }

            staging = make_contig_dev_tensor(const_cast<OpenCLBackend*>(this),
                                            dst->m_dtype,
                                            dst->m_shape);

            // pack: src(strided) -> staging(contig)
            cpy_tensor_cl(this, src, &staging);

            // staging must be contiguous
            POWERSERVE_ASSERT(is_contiguous(&staging, 4));
            read_src = &staging;
        }

        BaseBuffer& read_base = const_cast<Tensor*>(read_src)->get<BaseBuffer>();
        auto* read_cl = dynamic_cast<OpenCLBuffer*>(&read_base);
        if (!read_cl) {
            POWERSERVE_LOG_ERROR("D2H: expected OpenCLBuffer for read_src");
            return;
        }
        cl_mem read_mem = read_cl->get_device_buffer();

        if (dst_contig) {
            const size_t src_off = read_cl->get_base_offset();  // bytes
            if (!memory_pool->copy_device_to_host(host, read_mem, src_bytes, src_off)) {
                POWERSERVE_LOG_ERROR("D2H: copy_device_to_host failed");
            }
            return;
        }


        std::vector<uint8_t> host_contig(src_bytes);
        const size_t src_off = read_cl->get_base_offset();  // bytes
        if (!memory_pool->copy_device_to_host(host_contig.data(), read_mem, src_bytes, src_off)) {
            POWERSERVE_LOG_ERROR("D2H: staging copy_device_to_host failed");
            return;
        }

        if (!unpack_contig_to_cpu_strided(host_contig.data(), dst)) {
            POWERSERVE_LOG_ERROR("D2H: failed to unpack to CPU strided tensor");
        }
        return;
    }

    // D2D
    if (src_cl && dst_cl) {
        
        const size_t src_off = src_cl->get_base_offset();  // bytes
        const size_t dst_off = dst_cl->get_base_offset();  // bytes

        cl_mem src_dev = src_cl->get_device_buffer();
        cl_mem dst_dev = dst_cl->get_device_buffer();
        if (!src_dev || !dst_dev) {
            POWERSERVE_LOG_ERROR("D2D: invalid cl_mem");
            return;
        }
        if (src_off != 0 || dst_off != 0 || !is_contiguous(src, 4) || !is_contiguous(dst, 4)) {
            if (!is_cpy_kernel_supported(src->m_dtype) || !is_cpy_kernel_supported(dst->m_dtype)) {
                POWERSERVE_LOG_ERROR("D2D: non-trivial copy requires cpy kernel, unsupported dtype src={} dst={}",
                                    (int)src->m_dtype, (int)dst->m_dtype);
                return;
            }
            cpy_tensor_cl(this, src, dst);
            return;
        }
        if (!memory_pool->copy_device_to_device(dst_dev, src_dev, src_bytes)) {
            POWERSERVE_LOG_ERROR("D2D: copy_device_to_device failed");
            dump_backtrace();
            std::abort();
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

void OpenCLBackend::cont(const Tensor *out, const Tensor *x) const {
    if (!initialized) POWERSERVE_ABORT("OpenCL backend not initialized");
    if (!out || !x)   POWERSERVE_ABORT("cont got null tensor");

    POWERSERVE_ASSERT(is_contiguous(out, 4));

    const size_t x_bytes   = ggml_compat_nbytes(x->m_dtype, x->m_shape);
    const size_t out_bytes = ggml_compat_nbytes(out->m_dtype, out->m_shape);
    if (x_bytes == 0 || out_bytes == 0 || x_bytes != out_bytes) {
        POWERSERVE_ABORT("cont: nbytes mismatch x_bytes={} out_bytes={}", x_bytes, out_bytes);
    }

    // 如果 x 已经 contiguous，copy 就够了（copy 支持 shape 不同但 nbytes 相同）
    if (is_contiguous(x, 4)) {
        this->copy(out, x);
        return;
    }

    // --- Step 1: 先把 x(strided) pack 成 contiguous 的 tmp（shape 必须与 x 相同） ---
    Tensor tmp = make_contig_dev_tensor(const_cast<OpenCLBackend*>(this),
                                        x->m_dtype,
                                        x->m_shape);   // contiguous + same shape as x
    cpy_tensor_cl(this, x, &tmp); 

    // --- Step 2: 把 tmp 的连续字节搬到 out（shape 可以不同，只要 nbytes 相同） ---
    // 获取 OpenCLBuffer
    auto *tmp_cl = dynamic_cast<OpenCLBuffer*>(&tmp.get<BaseBuffer>());
    auto *out_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(out)->get<BaseBuffer>());
    if (!tmp_cl || !out_cl) {
        POWERSERVE_ABORT("cont: expected OpenCLBuffer for tmp/out");
    }

    cl_mem src_dev = tmp_cl->get_device_buffer();
    cl_mem dst_dev = out_cl->get_device_buffer();
    if (!src_dev || !dst_dev) {
        POWERSERVE_ABORT("cont: invalid cl_mem src_dev/dst_dev");
    }

    const size_t src_off = tmp_cl->get_base_offset(); 
    const size_t dst_off = out_cl->get_base_offset(); 

    // 快路径：两边 offset 都是 0 → 直接 D2D memcpy
    if (src_off == 0 && dst_off == 0) {
        if (!memory_pool->copy_device_to_device(dst_dev, src_dev, x_bytes)) {
            POWERSERVE_ABORT("cont: copy_device_to_device failed");
        }
        return;
    }

    // 慢路径但正确：D2H + H2D（支持 offset）
    std::vector<uint8_t> host(x_bytes);
    if (!memory_pool->copy_device_to_host(host.data(), src_dev, x_bytes, src_off)) {
        POWERSERVE_ABORT("cont: copy_device_to_host failed");
    }
    if (!memory_pool->copy_host_to_device(dst_dev, host.data(), x_bytes, dst_off)) {
        POWERSERVE_ABORT("cont: copy_host_to_device failed");
    }
    clFinish(context->get_queue());
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
    POWERSERVE_LOG_INFO("add_cache");

    if (!m_kv) { POWERSERVE_LOG_ERROR("add_cache: KVCache not allocated"); return; }
    if (!k || !v) { POWERSERVE_LOG_ERROR("add_cache: null tensor"); return; }

    if (m_kv->batch_size != 1 || pos.size() != 1) {
        POWERSERVE_LOG_ERROR("add_cache v0 expects batch=1 and pos.size()==1");
        return;
    }

    // ---- slot：必须跟模型的 token position 对齐；不要只用 m_kv->position ----
    if (pos[0] < 0) {
        POWERSERVE_LOG_ERROR("add_cache: invalid pos[0]={}", pos[0]);
        return;
    }
    const size_t slot = (size_t)pos[0];

    if (L >= m_kv->key.size()) {
        POWERSERVE_LOG_ERROR("add_cache: invalid layer {}", L);
        return;
    }
    if (slot >= m_kv->max_seq_len) {
        POWERSERVE_LOG_ERROR("KVCache overflow: slot {} max_seq_len {}", slot, m_kv->max_seq_len);
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
        POWERSERVE_LOG_ERROR("add_cache shape mismatch: expect {{kv_dim,1,1,1}} (kv_dim={})", kv_dim);
        return;
    }

    // ---- offset：bytes；create_buffer_view 会把 offset 加到 parent.base_offset 上（Scheme-B） ----
    const size_t token_bytes = kv_dim * sizeof(float);
    const size_t offset      = slot * token_bytes;

    Shape sTok{kv_dim, 1, 1, 1};

    try {
        auto &k_parent = *m_kv->key[L];
        auto &v_parent = *m_kv->value[L];

        // ---- 关键一致性检查：pos vs position（不一致就很可能“写错行、读到全0”）----
        if (slot != m_kv->position) {
            // 不直接 abort：先把现场打出来，让你确认是不是这里导致错位
            POWERSERVE_LOG_WARN("[KV][ADD_CACHE] position mismatch: slot(pos[0])={} m_kv->position={} (L={}, kv_dim={}, max_seq_len={})",
                                    slot, m_kv->position, L, kv_dim, m_kv->max_seq_len);
        }

        // ---- 边界（双保险；create_buffer_view 内部也会检查）----
        if (offset + token_bytes > k_parent.get_size()) {
            POWERSERVE_LOG_ERROR("[KV][ADD_CACHE] K view out of range: L={} slot={} offset={} token_bytes={} parent_size={}",
                                 L, slot, offset, token_bytes, k_parent.get_size());
            return;
        }
        if (offset + token_bytes > v_parent.get_size()) {
            POWERSERVE_LOG_ERROR("[KV][ADD_CACHE] V view out of range: L={} slot={} offset={} token_bytes={} parent_size={}",
                                 L, slot, offset, token_bytes, v_parent.get_size());
            return;
        }

        auto k_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(k_parent, sTok, offset);
        auto v_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(v_parent, sTok, offset);

        if (!k_view || !v_view) {
            POWERSERVE_LOG_ERROR("add_cache: create_buffer_view failed (L={}, slot={}, offset={})", L, slot, offset);
            return;
        }

        POWERSERVE_LOG_INFO(
                "[KV][ADD_CACHE] L={} slot={} kv_dim={} token_bytes={} offset={} | "
                "Kparent(dev={}, base_off={}, size={}) -> Kview(base_off={}, size={}) | "
                "Vparent(dev={}, base_off={}, size={}) -> Vview(base_off={}, size={})",
                L, slot, kv_dim, token_bytes, offset,
                (void*)k_parent.get_device_buffer(), k_parent.get_base_offset(), k_parent.get_size(),
                k_view->get_base_offset(), k_view->get_size(),
                (void*)v_parent.get_device_buffer(), v_parent.get_base_offset(), v_parent.get_size(),
                v_view->get_base_offset(), v_view->get_size()
            );

        Tensor t_dst_k(DataType::FP32, sTok);
        Tensor t_dst_v(DataType::FP32, sTok);
        t_dst_k.m_data = k_view;
        t_dst_v.m_data = v_view;

        // D2D 写入 cache 的这个 slot
        this->copy(&t_dst_k, k);
        this->copy(&t_dst_v, v);

        // ---- 可选：写完立刻抽样读回 8 个 float，确认“这一行真的被写了” ----
        // copy() 的 CL kernel 会显式传 base_offset（Scheme-B）:contentReference[oaicite:4]{index=4}，所以这个抽样能验证 offset 是否生效。
        Tensor host_k(DataType::FP32, Shape{8,1,1,1});
            host_k.m_data = powerserve::CPUBuffer::create_buffer<float>(Shape{8,1,1,1});

            Tensor k_first8(DataType::FP32, Shape{8,1,1,1});
            // 从 t_dst_k 再切一个 view：offset=0（在 token 内部取前8）
            auto k_first8_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(*k_view, Shape{8,1,1,1}, /*offset=*/0);
            k_first8.m_data = k_first8_view;

            this->copy(&host_k, &k_first8);

            auto &hb = host_k.get<powerserve::CPUBuffer>();
            float *p = (float*)hb.m_data;
            POWERSERVE_LOG_INFO("[KV][ADD_CACHE] L={} slot={} K_first8: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                                L, slot, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("add_cache expects OpenCLBuffer in KVCache: {}", e.what());
        return;
    }

    // ---- 更新 position：以 slot 为准，避免悄悄漂移 ----
    // v0 语义是 decode append 1 token（OpenCLKV::position）:contentReference[oaicite:5]{index=5}
    m_kv->position = slot + 1;
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
