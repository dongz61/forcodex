#include "backend/opencl/opencl_backend.hpp"
#include "backend/opencl/opencl_backend_helpers.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace powerserve::opencl {

using detail::ensure_contiguous_or_pack_f32;

static inline size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static inline bool is_adreno_device(cl_device_id dev) {
    auto get_str = [&](cl_device_info p) -> std::string {
        size_t n = 0;
        if (clGetDeviceInfo(dev, p, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
        std::string s(n, '\0');
        if (clGetDeviceInfo(dev, p, n, s.data(), nullptr) != CL_SUCCESS) return {};
        while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s;
    };
    const std::string name   = get_str(CL_DEVICE_NAME);
    const std::string vendor = get_str(CL_DEVICE_VENDOR);
    return (name.find("Adreno") != std::string::npos) ||
           (vendor.find("QUALCOMM") != std::string::npos) ||
           (vendor.find("Qualcomm") != std::string::npos);
}

void OpenCLBackend::get_embedding(const Tensor *dst,
                                  const Tensor *weight,
                                  const std::vector<int> &tokens) const {
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

    POWERSERVE_ASSERT(m_ggml_fallback && "m_ggml_fallback must be initialized in OpenCLBackend::initialize()");

    constexpr size_t kMinWSize = 1 * 1024 * 1024;
    if (m_ggml_fallback_wsize < kMinWSize) {
        m_ggml_fallback->setup_work_data(kMinWSize);
        m_ggml_fallback_wsize = kMinWSize;
    }

    Tensor host_tmp(DataType::FP32, dst->m_shape);
    host_tmp.m_data = CPUBuffer::create_buffer<float>(dst->m_shape);
    m_ggml_fallback->get_embedding(&host_tmp, weight, tokens);

    this->copy(dst, &host_tmp);
}

static inline powerserve::BufferPtr create_cpu_buffer_for_dtype(powerserve::DataType dt,
                                                                const powerserve::Shape &shape) {
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

    // ===== Quantized GGML buffers (match ggml nb[] layout) =====
    case powerserve::DataType::GGML_Q4_0:
    case powerserve::DataType::GGML_Q8_0: {
        const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(dt);

        powerserve::Stride stride{};
        stride[0] = (size_t) ggml_type_size(gt);
        stride[1] = (size_t) ggml_row_size(gt, (int64_t) shape[0]);
        stride[2] = stride[1] * (size_t) shape[1];
        stride[3] = stride[2] * (size_t) shape[2];

        const size_t bytes = stride[3] * (size_t) shape[3];
        void *ptr = malloc(bytes);
        POWERSERVE_ASSERT(ptr && "malloc failed for quant CPU buffer");

        return std::make_shared<CPUBuffer>(stride, ptr, /*allocated_by_malloc=*/true);
    }

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

    auto is_cpu_tensor = [](const Tensor *t) -> bool {
        return dynamic_cast<powerserve::CPUBuffer *>(t->m_data.get()) != nullptr;
    };

    const Tensor *a_host = src0;
    const Tensor *b_host = src1;

    Tensor host_a;
    Tensor host_b;

    if (!is_cpu_tensor(src0)) {
        host_a = Tensor(src0->m_dtype, src0->m_shape);
        host_a.m_data = create_cpu_buffer_for_dtype(src0->m_dtype, src0->m_shape);
        this->copy(&host_a, src0);
        a_host = &host_a;
    }

    if (!is_cpu_tensor(src1)) {
        host_b = Tensor(src1->m_dtype, src1->m_shape);
        host_b.m_data = create_cpu_buffer_for_dtype(src1->m_dtype, src1->m_shape);
        this->copy(&host_b, src1);   // supports quant bytes via ggml_compat_nbytes in opencl_tensor_ops.cpp
        b_host = &host_b;
    }

    Tensor host_c(dst->m_dtype, dst->m_shape);
    host_c.m_data = create_cpu_buffer_for_dtype(dst->m_dtype, dst->m_shape);

    POWERSERVE_ASSERT(m_ggml_fallback && "m_ggml_fallback must be initialized in OpenCLBackend::initialize()");

    const Tensor *w_host = a_host;
    const Tensor *x_host = b_host;

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

    const size_t n_threads = (size_t)m_hparams.n_threads;
    size_t required_wsize = sizeof(float) * (size_t)(K_w + 64) * n_threads;

    {
        const enum ggml_type vec_dot_type = m_ggml_fallback->get_vec_dot_type(x_host);
        const enum ggml_type w_type       = powerserve::ggml::convert_datatype_to_ggml(w_host->m_dtype);
        if (w_type != vec_dot_type) {
            const size_t extra = (size_t)ggml_row_size(vec_dot_type, (int64_t)w_host->n_elements());
            required_wsize = std::max(required_wsize, extra);
        }
    }

    if (required_wsize > m_ggml_fallback_wsize) {
        m_ggml_fallback->setup_work_data(required_wsize);
        m_ggml_fallback_wsize = required_wsize;
    }

    m_ggml_fallback->matmul(&host_c, w_host, x_host);

    this->copy(dst, &host_c);
}

OpenCLBackend::QuantSplitBuffers OpenCLBackend::get_or_create_split_q4_0(const Tensor* w) const {
    POWERSERVE_ASSERT(w != nullptr);
    POWERSERVE_ASSERT(w->m_dtype == DataType::GGML_Q4_0);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && "Q4_0 weight must be on OpenCLBuffer");
    POWERSERVE_ASSERT(is_contiguous(w, 4) && "Q4_0 weight must be ggml-contiguous on device");

    QuantSplitKey key;
    key.mem         = w_cl->get_device_buffer();
    key.base_offset = w_cl->get_base_offset();
    key.dtype       = w->m_dtype;
    key.shape       = w->m_shape;

    {
        std::lock_guard<std::mutex> lock(m_quant_split_mutex);
        auto it = m_quant_split_cache.find(key);
        if (it != m_quant_split_cache.end()) return it->second;
    }

    const int K  = (int)w->m_shape[0];
    const int N  = (int)w->m_shape[1];
    const int ne2 = (int)w->m_shape[2];
    const int ne3 = (int)w->m_shape[3];
    POWERSERVE_ASSERT((K % 32) == 0 && "Q4_0 expects K multiple of 32");

    const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(w->m_dtype);
    const size_t row_bytes   = (size_t)ggml_row_size(gt, K);
    const size_t block_bytes = (size_t)ggml_type_size(gt);   // 18 for q4_0
    const size_t q_bytes_per_block = 16;                     // QK4_0/2
    const size_t blocks_per_row = (size_t)(K / 32);
    const size_t rows_total = (size_t)N * (size_t)ne2 * (size_t)ne3;
    const size_t blocks_total = blocks_per_row * rows_total;

    const size_t total_bytes = row_bytes * rows_total;

    std::vector<uint8_t> interleaved(total_bytes);
    memory_pool->copy_device_to_host(interleaved.data(), w_cl->get_device_buffer(), total_bytes, w_cl->get_base_offset());

    std::vector<uint8_t> q_out(blocks_total * q_bytes_per_block);
    std::vector<cl_half> d_out(blocks_total);

    // Order matches ggml: contiguous rows, each row contains blocks_per_row blocks
    size_t out_block = 0;
    for (size_t r = 0; r < rows_total; ++r) {
        const uint8_t* row_ptr = interleaved.data() + r * row_bytes;
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_ptr + b * block_bytes;
            // [d:2 bytes][qs:16 bytes]
            cl_half d;
            std::memcpy(&d, blk, sizeof(cl_half));
            d_out[out_block] = d;
            std::memcpy(q_out.data() + out_block * q_bytes_per_block, blk + sizeof(cl_half), q_bytes_per_block);
            ++out_block;
        }
    }

    Shape q_shape{ q_out.size(), 1, 1, 1 };
    Shape d_shape{ blocks_total, 1, 1, 1 };

    QuantSplitBuffers res;
    res.q = OpenCLBuffer::create_buffer<uint8_t>(q_shape, memory_pool);
    res.d = OpenCLBuffer::create_buffer<cl_half>(d_shape, memory_pool);
    res.blocks_total = blocks_total;

    POWERSERVE_ASSERT(res.q && res.d);
    memory_pool->copy_host_to_device(res.q->get_device_buffer(), q_out.data(), q_out.size(), /*dst_offset=*/0);
    memory_pool->copy_host_to_device(res.d->get_device_buffer(), d_out.data(), d_out.size() * sizeof(cl_half), /*dst_offset=*/0);

    {
        std::lock_guard<std::mutex> lock(m_quant_split_mutex);
        m_quant_split_cache.emplace(key, res);
    }
    return res;
}

OpenCLBackend::QuantSplitBuffers OpenCLBackend::get_or_create_split_q8_0(const Tensor* w) const {
    POWERSERVE_ASSERT(w != nullptr);
    POWERSERVE_ASSERT(w->m_dtype == DataType::GGML_Q8_0);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && "Q8_0 weight must be on OpenCLBuffer");
    POWERSERVE_ASSERT(is_contiguous(w, 4) && "Q8_0 weight must be ggml-contiguous on device");

    QuantSplitKey key;
    key.mem         = w_cl->get_device_buffer();
    key.base_offset = w_cl->get_base_offset();
    key.dtype       = w->m_dtype;
    key.shape       = w->m_shape;

    {
        std::lock_guard<std::mutex> lock(m_quant_split_mutex);
        auto it = m_quant_split_cache.find(key);
        if (it != m_quant_split_cache.end()) return it->second;
    }

    const int K  = (int)w->m_shape[0];
    const int N  = (int)w->m_shape[1];
    const int ne2 = (int)w->m_shape[2];
    const int ne3 = (int)w->m_shape[3];
    POWERSERVE_ASSERT((K % 32) == 0 && "Q8_0 expects K multiple of 32");

    const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(w->m_dtype);
    const size_t row_bytes   = (size_t)ggml_row_size(gt, K);
    const size_t block_bytes = (size_t)ggml_type_size(gt);   // 34 for q8_0
    const size_t q_bytes_per_block = 32;                     // QK8_0
    const size_t blocks_per_row = (size_t)(K / 32);
    const size_t rows_total = (size_t)N * (size_t)ne2 * (size_t)ne3;
    const size_t blocks_total = blocks_per_row * rows_total;

    const size_t total_bytes = row_bytes * rows_total;

    std::vector<uint8_t> interleaved(total_bytes);
    memory_pool->copy_device_to_host(interleaved.data(), w_cl->get_device_buffer(), total_bytes, w_cl->get_base_offset());

    std::vector<uint8_t> q_out(blocks_total * q_bytes_per_block);
    std::vector<cl_half> d_out(blocks_total);

    size_t out_block = 0;
    for (size_t r = 0; r < rows_total; ++r) {
        const uint8_t* row_ptr = interleaved.data() + r * row_bytes;
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_ptr + b * block_bytes;
            cl_half d;
            std::memcpy(&d, blk, sizeof(cl_half));
            d_out[out_block] = d;
            std::memcpy(q_out.data() + out_block * q_bytes_per_block, blk + sizeof(cl_half), q_bytes_per_block);
            ++out_block;
        }
    }

    Shape q_shape{ q_out.size(), 1, 1, 1 };
    Shape d_shape{ blocks_total, 1, 1, 1 };

    QuantSplitBuffers res;
    res.q = OpenCLBuffer::create_buffer<uint8_t>(q_shape, memory_pool);
    res.d = OpenCLBuffer::create_buffer<cl_half>(d_shape, memory_pool);
    res.blocks_total = blocks_total;

    POWERSERVE_ASSERT(res.q && res.d);
    memory_pool->copy_host_to_device(res.q->get_device_buffer(), q_out.data(), q_out.size(), /*dst_offset=*/0);
    memory_pool->copy_host_to_device(res.d->get_device_buffer(), d_out.data(), d_out.size() * sizeof(cl_half), /*dst_offset=*/0);

    {
        std::lock_guard<std::mutex> lock(m_quant_split_mutex);
        m_quant_split_cache.emplace(key, res);
    }
    return res;
}

void OpenCLBackend::matmul_opencl_f16_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const {
    auto* ctx = context.get();
    POWERSERVE_ASSERT(ctx && kernel_manager);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    auto* x_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(x)->get<BaseBuffer>());
    auto* d_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(dst)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && x_cl && d_cl);

    // w: [K,N] (ggml), x: [K,M], dst: [N,M]
    const int K = (int)w->m_shape[0];
    const int N = (int)w->m_shape[1];
    const int M = (int)x->m_shape[1];

    // kernel expects:
    // A: [M_rows = N, K] row-major  => w is ok (ggml [K,N] contiguous == row-major [N,K])
    // B: [N_rows = M, K] row-major  => x is ok (ggml [K,M] contiguous == row-major [M,K])
    // C: column-major [M_rows=N, N_cols=M] => dst is ggml [N,M]
    const int Mm = N;
    const int Nn = M;

    cl_kernel k = kernel_manager->get_kernel("kernel_mul_mat_f16_f32");
    POWERSERVE_ASSERT(k && "kernel_mul_mat_f16_f32 not found (did you embed/compile mul_mat_f16_f32.cl?)");

    cl_mem A = w_cl->get_device_buffer();
    cl_mem B = x_cl->get_device_buffer();
    cl_mem C = d_cl->get_device_buffer();

    const cl_ulong A_off = (cl_ulong)w_cl->get_base_offset();
    const cl_ulong B_off = (cl_ulong)x_cl->get_base_offset();
    const cl_ulong C_off = (cl_ulong)d_cl->get_base_offset();

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &Mm));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &Nn));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &K));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &A));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &A_off));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &B));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &B_off));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &C));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &C_off));

    // kernel uses 2D local ids: (WG_M=16, WG_N=8)
    const size_t local[2] = { 16, 8 };
    const size_t global[2] = {
        ceil_div((size_t)Mm, (size_t)64) * local[0],
        ceil_div((size_t)Nn, (size_t)64) * local[1],
    };

    OCL_RETURN_IF_ERROR(ctx, clEnqueueNDRangeKernel(ctx->get_queue(), k, 2, nullptr, global, local, 0, nullptr, nullptr));
    OCL_RETURN_IF_ERROR(ctx, clFinish(ctx->get_queue()));
}

void OpenCLBackend::matmul_opencl_q4_0_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const {
    auto* ctx = context.get();
    POWERSERVE_ASSERT(ctx && kernel_manager);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    auto* x_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(x)->get<BaseBuffer>());
    auto* d_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(dst)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && x_cl && d_cl);

    const int K = (int)w->m_shape[0];
    const int N = (int)w->m_shape[1];
    const int M = (int)x->m_shape[1];

    cl_kernel k = kernel_manager->get_kernel("kernel_mul_mat_q4_0_f32_simple");
    POWERSERVE_ASSERT(k && "kernel_mul_mat_q4_0_f32_simple not found");

    cl_mem wmem  = w_cl->get_device_buffer();
    cl_mem xmem  = x_cl->get_device_buffer();
    cl_mem out   = d_cl->get_device_buffer();

    const cl_ulong off_w  = (cl_ulong)w_cl->get_base_offset();
    const cl_ulong off_x  = (cl_ulong)x_cl->get_base_offset();
    const cl_ulong off_d  = (cl_ulong)d_cl->get_base_offset();

    const cl_ulong nb_w1   = (cl_ulong)w_cl->get_stride()[1];
    const cl_ulong nb_x1   = (cl_ulong)x_cl->get_stride()[1];
    const cl_ulong nb_dst1 = (cl_ulong)d_cl->get_stride()[1];

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &wmem));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_w));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &xmem));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_x));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &out));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_d));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &K));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &N));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &M));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_w1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_x1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_dst1));

    const size_t local[2]  = { 16, 16 };
    const size_t global[2] = {
        ((size_t)N + local[0] - 1) / local[0] * local[0],
        ((size_t)M + local[1] - 1) / local[1] * local[1],
    };

    OCL_RETURN_IF_ERROR(ctx, clEnqueueNDRangeKernel(ctx->get_queue(), k, 2, nullptr, global, local, 0, nullptr, nullptr));
    OCL_RETURN_IF_ERROR(ctx, clFinish(ctx->get_queue()));
}

void OpenCLBackend::matmul_opencl_q8_0_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const {
    auto* ctx = context.get();
    POWERSERVE_ASSERT(ctx && kernel_manager);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    auto* x_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(x)->get<BaseBuffer>());
    auto* d_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(dst)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && x_cl && d_cl);

    const int K = (int)w->m_shape[0];
    const int N = (int)w->m_shape[1];
    const int M = (int)x->m_shape[1];

    cl_kernel k = kernel_manager->get_kernel("kernel_mul_mat_q8_0_f32_simple");
    POWERSERVE_ASSERT(k && "kernel_mul_mat_q8_0_f32_simple not found");

    cl_mem wmem  = w_cl->get_device_buffer();
    cl_mem xmem  = x_cl->get_device_buffer();
    cl_mem out   = d_cl->get_device_buffer();

    const cl_ulong off_w  = (cl_ulong)w_cl->get_base_offset();
    const cl_ulong off_x  = (cl_ulong)x_cl->get_base_offset();
    const cl_ulong off_d  = (cl_ulong)d_cl->get_base_offset();

    const cl_ulong nb_w1   = (cl_ulong)w_cl->get_stride()[1];
    const cl_ulong nb_x1   = (cl_ulong)x_cl->get_stride()[1];
    const cl_ulong nb_dst1 = (cl_ulong)d_cl->get_stride()[1];

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &wmem));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_w));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &xmem));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_x));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_mem), &out));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off_d));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &K));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &N));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(int), &M));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_w1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_x1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb_dst1));

    const size_t local[2]  = { 16, 16 };
    const size_t global[2] = {
        ((size_t)N + local[0] - 1) / local[0] * local[0],
        ((size_t)M + local[1] - 1) / local[1] * local[1],
    };

    OCL_RETURN_IF_ERROR(ctx, clEnqueueNDRangeKernel(ctx->get_queue(), k, 2, nullptr, global, local, 0, nullptr, nullptr));
    OCL_RETURN_IF_ERROR(ctx, clFinish(ctx->get_queue()));
}



void OpenCLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    POWERSERVE_ASSERT(dst && src0 && src1);
    POWERSERVE_ASSERT(context && kernel_manager);

    // We only support: (weight: FP16/Q4_0/Q8_0) x (activations: FP32) -> FP32
    if (dst->m_dtype != DataType::FP32 || src1->m_dtype != DataType::FP32) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: only supports dst=FP32 and src1=FP32 (got dst={}, src1={})",
                         (int)dst->m_dtype, (int)src1->m_dtype);
    }
    if (!(src0->m_dtype == DataType::FP16 || src0->m_dtype == DataType::GGML_Q4_0 || src0->m_dtype == DataType::GGML_Q8_0)) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: unsupported weight dtype {} (no ggml fallback)", (int)src0->m_dtype);
    }

    // Current implementation: 2D only (most linear layers)
    if (dst->m_shape[2] != 1 || dst->m_shape[3] != 1 ||
        src0->m_shape[2] != 1 || src0->m_shape[3] != 1 ||
        src1->m_shape[2] != 1 || src1->m_shape[3] != 1) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: only supports 2D tensors (shape[2]=shape[3]=1) for now");
    }

    // Shapes: w=[K,N], x=[K,M], dst=[N,M]
    const int K  = (int)src0->m_shape[0];
    const int N  = (int)src0->m_shape[1];
    const int Kx = (int)src1->m_shape[0];
    const int M  = (int)src1->m_shape[1];

    if (Kx != K) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: K mismatch w.K={} x.K={}", K, Kx);
    }
    if ((int)dst->m_shape[0] != N || (int)dst->m_shape[1] != M) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: dst shape mismatch, expected [N,M]=[{},{}], got [{},{}]",
                         N, M, (int)dst->m_shape[0], (int)dst->m_shape[1]);
    }

    auto *self = const_cast<OpenCLBackend *>(this);

    // ---- ensure weight is on OpenCL + ggml-contiguous ----
    Tensor tmp_w_upload;
    Tensor tmp_w_contig;
    const Tensor *w_dev = src0;

    // If weight is still on CPU (can happen depending on loader / backend init), upload it.
    if (!dynamic_cast<powerserve::opencl::OpenCLBuffer *>(src0->m_data.get())) {
        tmp_w_upload = Tensor(src0->m_dtype, src0->m_shape);
        tmp_w_upload.m_data = self->create_buffer(src0->m_shape, src0->m_dtype);
        self->copy(&tmp_w_upload, src0); // H2D, supports quant bytes via copy path
        w_dev = &tmp_w_upload;
    }

    // If weight is a view / non-ggml-contig on device, pack/copy to a contiguous buffer.
    if (!is_contiguous(w_dev, 4)) {
        tmp_w_contig = Tensor(w_dev->m_dtype, w_dev->m_shape);
        tmp_w_contig.m_data = self->create_buffer(w_dev->m_shape, w_dev->m_dtype);
        detail::cpy_tensor_cl(self, w_dev, &tmp_w_contig);
        w_dev = &tmp_w_contig;
    }

    // ---- Ensure src1 and dst are OpenCL buffers + contiguous when needed ----
    Tensor tmp_x_dev;
    Tensor tmp_dst_dev;

    const Tensor *x_use = src1;
    if (!is_contiguous(src1, 4)) {
        tmp_x_dev = Tensor(DataType::FP32, src1->m_shape);
        tmp_x_dev.m_data = self->create_buffer(src1->m_shape, DataType::FP32);
        detail::cpy_tensor_cl(self, src1, &tmp_x_dev);
        x_use = &tmp_x_dev;
    } else if (!dynamic_cast<powerserve::opencl::OpenCLBuffer *>(src1->m_data.get())) {
        // (rare) if activation is CPU but contiguous, upload
        tmp_x_dev = Tensor(DataType::FP32, src1->m_shape);
        tmp_x_dev.m_data = self->create_buffer(src1->m_shape, DataType::FP32);
        self->copy(&tmp_x_dev, src1);
        x_use = &tmp_x_dev;
    }

    const Tensor *dst_use = dst;
    bool need_scatter_back = false;
    if (!is_contiguous(dst, 4)) {
        tmp_dst_dev = Tensor(DataType::FP32, dst->m_shape);
        tmp_dst_dev.m_data = self->create_buffer(dst->m_shape, DataType::FP32);
        dst_use = &tmp_dst_dev;
        need_scatter_back = true;
    } else if (!dynamic_cast<powerserve::opencl::OpenCLBuffer *>(dst->m_data.get())) {
        // (rare) dst on CPU: compute into temp and copy back
        tmp_dst_dev = Tensor(DataType::FP32, dst->m_shape);
        tmp_dst_dev.m_data = self->create_buffer(dst->m_shape, DataType::FP32);
        dst_use = &tmp_dst_dev;
        need_scatter_back = true;
    }

    // Dispatch by weight dtype (no ggml fallback)
    switch (w_dev->m_dtype) {
        case DataType::FP16:
            matmul_opencl_f16_f32(dst_use, w_dev, x_use);
            break;
        case DataType::GGML_Q4_0:
            matmul_opencl_q4_0_f32(dst_use, w_dev, x_use);
            break;
        case DataType::GGML_Q8_0:
            matmul_opencl_q8_0_f32(dst_use, w_dev, x_use);
            break;
        default:
            POWERSERVE_ABORT("OpenCLBackend::matmul: unreachable dtype {}", (int)w_dev->m_dtype);
    }

    if (need_scatter_back) {
        detail::cpy_tensor_cl(self, dst_use, dst);
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

    Tensor host_x(DataType::FP32, x->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(x->m_shape);
    this->copy(&host_x, x);

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

    Tensor host_y(DataType::FP32, o->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(o->m_shape);
    m_ggml_fallback->rmsnorm(&host_y, &host_x, host_w_ptr, eps);

    this->copy(o, &host_y);
}

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

    if (out->m_dtype != DataType::FP32 || src->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope fallback only supports FP32");
        return;
    }
    if (out->m_shape != src->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope requires out.shape == src.shape");
        return;
    }

    Tensor host_x(DataType::FP32, src->m_shape);
    host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
    this->copy(&host_x, src);

    Tensor host_y(DataType::FP32, out->m_shape);
    host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

    m_ggml_fallback->rope(&host_y, &host_x, pos, rope_cfg);

    this->copy(out, &host_y);
}

void OpenCLBackend::softmax(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::softmax TODO");
}

static inline uint32_t floor_log2_u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << (r + 1)) <= x) ++r;
    return r;
}

static void softmax_ext_cpu_f32_ggml_semantics(
    float *dst,
    const float *src0,
    const float *src1,
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
        const uint32_t h = (uint32_t)((i1 / ne01) % ne02);

        const float slope =
            (max_bias > 0.0f)
            ? (h < n_head_log2
                ? std::pow(m0, (float)(h + 1))
                : std::pow(m1, (float)(2*(h - n_head_log2) + 1)))
            : 1.0f;

        const float *sp = src0 + (size_t)i1 * (size_t)nc;
        float *dp       = dst  + (size_t)i1 * (size_t)nc;

        const float *mp = src1 ? (src1 + (size_t)(i1 % ne01) * (size_t)ne00) : nullptr;

        for (int i = 0; i < nc; ++i) {
            float v = sp[i] * scale;
            if (mp) v += slope * mp[i];
            wp[i] = v;
        }

        float mx = -INFINITY;
        for (int i = 0; i < nc; ++i) mx = std::max(mx, wp[i]);

        float sum = 0.0f;
        for (int i = 0; i < nc; ++i) {
            float e = std::exp(wp[i] - mx);
            dp[i] = e;
            sum += e;
        }

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

    const int n_dims_check = 4;
    Tensor tmp_x_dev, tmp_mask_dev;
    const Tensor *x_dev    = ensure_contiguous_or_pack_f32(self, x,    n_dims_check, tmp_x_dev);
    const Tensor *m_dev    = ensure_contiguous_or_pack_f32(self, mask, n_dims_check, tmp_mask_dev);

    const int ne00 = (int)x_dev->m_shape[0];
    const int ne01 = (int)x_dev->m_shape[1];
    const int ne02 = (int)x_dev->m_shape[2];
    const int ne03 = (int)x_dev->m_shape[3];

    if (out->m_shape != x_dev->m_shape) {
        POWERSERVE_LOG_ERROR("softmax_ext: out shape != x shape");
        return;
    }

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

    softmax_ext_cpu_f32_ggml_semantics(
        o_buf, x_buf, m_buf,
        ne00, ne01, ne02, ne03,
        scale, max_bias
    );

    self->copy(out, &host_out);
}

} // namespace powerserve::opencl
