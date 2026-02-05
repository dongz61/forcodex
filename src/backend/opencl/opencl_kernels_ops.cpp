#include "backend/opencl/opencl_backend.hpp"
#include "backend/opencl/opencl_backend_helpers.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"
#include "ggml.h"

#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace powerserve::opencl {

using detail::ensure_contiguous_or_pack_f32;

static inline size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}

static inline uint32_t floor_log2_u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << (r + 1)) <= x) ++r;
    return r;
}

static inline bool device_supports_fp64(cl_device_id dev) {
    size_t n = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &n) != CL_SUCCESS || n == 0) {
        return false;
    }
    std::string ext(n, '\0');
    if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, n, ext.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    return ext.find("cl_khr_fp64") != std::string::npos || ext.find("cl_amd_fp64") != std::string::npos;
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

void OpenCLBackend::add_minimal(Tensor * dst, const Tensor * src0, const Tensor * src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }

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

    cl_kernel kernel = kernel_manager->get_kernel("kernel_add_contig_f32");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("kernel not found: kernel_add_contig_f32");
        return;
    }

    cl_int err = CL_SUCCESS;
    cl_uint idx = 0;
    const int n_i = static_cast<int>(n);

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

void OpenCLBackend::add_broadcast(Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    try {
        auto& src0_buffer = src0->get<OpenCLBuffer>();
        auto& src1_buffer = src1->get<OpenCLBuffer>();
        auto& dst_buffer = dst->get<OpenCLBuffer>();

        Shape src0_shape = src0->m_shape;
        Shape src1_shape = src1->m_shape;
        Shape dst_shape = dst->m_shape;

        Stride src0_stride = src0_buffer.get_stride();
        Stride src1_stride = src1_buffer.get_stride();
        Stride dst_stride = dst_buffer.get_stride();

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

            const bool src1_contig_dim0 = (nb10 == sizeof(float));
            const bool align_ok = true;
            bcast_row = src1_contig_dim0 && align_ok;
        }

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

        if (!kernel) {
            POWERSERVE_LOG_ERROR("Add kernel not found: {}", kernel_name);
            return;
        }

        cl_int err;
        cl_uint arg_index = 0;

        cl_ulong offset0 = (cl_ulong)src0_buffer.get_base_offset();
        cl_ulong offset1 = (cl_ulong)src1_buffer.get_base_offset();
        cl_ulong offsetd = (cl_ulong)dst_buffer.get_base_offset();

        auto* ctx = context.get();

        if (bcast_row) {
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
            OCL_RETURN_IF_ERROR(ctx, err);

            const int ne_vec4 = ne0 / 4;
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne_vec4);
            OCL_RETURN_IF_ERROR(ctx, err);
        } else {
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne00);  // 7
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne01);  // 8
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne02);  // 9
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne03);  // 10
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb00);  // 11
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb01);  // 12
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb02);  // 13
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb03);  // 14
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne10);  // 15
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne11);  // 16
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne12);  // 17
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne13);  // 18
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb10);  // 19
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb11);  // 20
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb12);  // 21
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb13);  // 22
            OCL_RETURN_IF_ERROR(ctx, err);

            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne0);  // 23
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne1);  // 24
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne2);  // 25
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne3);  // 26
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb0);  // 27
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb1);  // 28
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb2);  // 29
            OCL_RETURN_IF_ERROR(ctx, err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb3);  // 30
            OCL_RETURN_IF_ERROR(ctx, err);
        }

        if (bcast_row) {
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
            OCL_RETURN_IF_ERROR(ctx, err);

        } else {
            if (ne01 <= 0 || ne02 <= 0 || ne03 <= 0) return;

            size_t global_work_size[3] = {
                static_cast<size_t>(ne01),
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
            OCL_RETURN_IF_ERROR(ctx, err);
        }

        err = clFinish(context->get_queue());
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
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

    if (dst->m_dtype != DataType::FP32 ||
        src0->m_dtype != DataType::FP32 ||
        src1->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::add only supports FP32");
        return;
    }

    auto *self = const_cast<OpenCLBackend*>(this);

    if (dst->m_shape == src0->m_shape && dst->m_shape == src1->m_shape) {
        Tensor tmp0, tmp1;
        const Tensor *src0_c = ensure_contiguous_or_pack_f32(self, src0, 4, tmp0);
        const Tensor *src1_c = ensure_contiguous_or_pack_f32(self, src1, 4, tmp1);
        self->add_minimal(const_cast<Tensor *>(dst), src0_c, src1_c);
        return;
    }

    self->add_broadcast(const_cast<Tensor *>(dst), src0, src1);
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

void OpenCLBackend::silu_hadamard(const Tensor * out,
                                 const Tensor * hb,
                                 const Tensor * hb2) const {
    if (!initialized) {
        POWERSERVE_ABORT("OpenCL backend not initialized");
    }
    if (!out || !hb || !hb2) {
        POWERSERVE_ABORT("silu_hadamard got null tensor");
    }

    if (out->m_dtype != DataType::FP32 ||
        hb->m_dtype  != DataType::FP32 ||
        hb2->m_dtype != DataType::FP32) {
        POWERSERVE_ABORT("silu_hadamard only supports FP32");
    }

    if (out->m_shape != hb->m_shape || out->m_shape != hb2->m_shape) {
        POWERSERVE_ABORT("silu_hadamard requires same shape");
    }

    POWERSERVE_ASSERT(is_contiguous(out, 0));
    POWERSERVE_ASSERT(is_contiguous(hb, 0));
    POWERSERVE_ASSERT(is_contiguous(hb2, 0));

    const size_t n = out->n_elements();
    if (n == 0) return;

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

void OpenCLBackend::get_embedding(const Tensor *dst,
                                  const Tensor *weight,
                                  const std::vector<int> &tokens) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!dst || !weight) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding got null tensor");
        return;
    }
    if (tokens.empty()) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding got empty tokens");
        return;
    }
    if (dst->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding dst must be FP32");
        return;
    }
    if (dst->m_shape[1] != tokens.size()) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding dst batch {} != tokens {}", dst->m_shape[1], tokens.size());
        return;
    }

    auto dst_device = dynamic_cast<OpenCLBuffer *>(dst->m_data.get());
    if (!dst_device) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding dst must be OpenCLBuffer");
        return;
    }

    auto weight_device = dynamic_cast<OpenCLBuffer *>(weight->m_data.get());
    if (!weight_device) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding weight must be OpenCLBuffer");
        return;
    }

    auto *self = const_cast<OpenCLBackend *>(this);
    auto *ctx = self->context.get();
    if (!ctx || !self->kernel_manager) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding missing OpenCL context/kernel manager");
        return;
    }

    Shape tokens_shape{tokens.size(), 1, 1, 1};
    Tensor tokens_host(DataType::INT32, tokens_shape);
    tokens_host.m_data = powerserve::CPUBuffer::create_buffer<int32_t>(tokens_shape);
    auto &tokens_cpu = tokens_host.get<powerserve::CPUBuffer>();
    std::memcpy(tokens_cpu.m_data, tokens.data(), tokens.size() * sizeof(int32_t));

    Tensor tokens_dev(DataType::INT32, tokens_shape);
    tokens_dev.m_data = self->create_buffer(tokens_shape, DataType::INT32);
    if (!tokens_dev.m_data) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding failed to allocate tokens buffer");
        return;
    }
    self->copy(&tokens_dev, &tokens_host);

    auto *tokens_cl = dynamic_cast<OpenCLBuffer *>(&tokens_dev.get<BaseBuffer>());
    if (!tokens_cl) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding failed to create tokens OpenCLBuffer");
        return;
    }

    cl_mem w_cl = weight_device->get_device_buffer();
    cl_mem t_cl = tokens_cl->get_device_buffer();
    cl_mem o_cl = dst_device->get_device_buffer();
    if (!w_cl || !t_cl || !o_cl) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding invalid cl_mem buffers");
        return;
    }

    const cl_ulong off_w = static_cast<cl_ulong>(weight_device->get_base_offset());
    const cl_ulong off_t = static_cast<cl_ulong>(tokens_cl->get_base_offset());
    const cl_ulong off_o = static_cast<cl_ulong>(dst_device->get_base_offset());

    const int dim = static_cast<int>(dst->m_shape[0]);
    const int batch = static_cast<int>(tokens.size());
    if (dim <= 0 || batch <= 0) {
        return;
    }

    cl_kernel kernel = nullptr;
    if (weight->m_dtype == DataType::FP32) {
        kernel = self->kernel_manager->get_kernel("kernel_get_rows_f32");
    } else if (weight->m_dtype == DataType::GGML_Q4_0) {
        kernel = self->kernel_manager->get_kernel("kernel_get_rows_q4_0");
    } else if (weight->m_dtype == DataType::GGML_Q8_0) {
        kernel = self->kernel_manager->get_kernel("kernel_get_rows_q8_0");
    } else {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding unsupported weight dtype {}", (int)weight->m_dtype);
        return;
    }

    if (!kernel) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding kernel not found");
        return;
    }

    const auto w_stride = weight_device->get_stride();
    const auto dst_stride = dst_device->get_stride();

    cl_int err = CL_SUCCESS;
    cl_uint arg = 0;

    if (weight->m_dtype == DataType::FP32) {
        const cl_ulong nb_w0 = static_cast<cl_ulong>(w_stride[0]);
        const cl_ulong nb_w1 = static_cast<cl_ulong>(w_stride[1]);
        const cl_ulong nb_dst0 = static_cast<cl_ulong>(dst_stride[0]);
        const cl_ulong nb_dst1 = static_cast<cl_ulong>(dst_stride[1]);

        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &w_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_w);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_w0);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_w1);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &t_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_t);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &o_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_o);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_dst0);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_dst1);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(int), &dim);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(int), &batch);
        OCL_RETURN_IF_ERROR(ctx, err);
    } else {
        const cl_ulong nb_w1 = static_cast<cl_ulong>(w_stride[1]);
        const cl_ulong nb_dst1 = static_cast<cl_ulong>(dst_stride[1]);

        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &w_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_w);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_w1);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &t_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_t);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &o_cl);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_o);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb_dst1);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(int), &dim);
        OCL_RETURN_IF_ERROR(ctx, err);
        err = clSetKernelArg(kernel, arg++, sizeof(int), &batch);
        OCL_RETURN_IF_ERROR(ctx, err);
    }

    size_t global[2] = {static_cast<size_t>(dim), static_cast<size_t>(batch)};
    if (weight->m_dtype == DataType::GGML_Q4_0) {
        const size_t chunks = (static_cast<size_t>(dim) + 15) / 16;
        global[0] = chunks;
    }
    cl_command_queue q = ctx->get_queue();
    err = clEnqueueNDRangeKernel(q, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::get_embedding kernel launch failed: {}", ctx->get_error_string(err));
        return;
    }

    err = clFinish(q);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_WARN("OpenCLBackend::get_embedding clFinish failed: {}", ctx->get_error_string(err));
    }
}

void OpenCLBackend::matmul_cpu_ggml_fallback(
    const Tensor *dst,
    const Tensor *src0,
    const Tensor *src1
) const {
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

    cl_kernel k = kernel_manager->get_kernel("kernel_mul_mat_f16_f32_simple");
    POWERSERVE_ASSERT(k && "kernel_mul_mat_f16_f32_simple not found (did you embed/compile mul_mat_f16_f32.cl?)");

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

void OpenCLBackend::matmul_opencl_f32_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const {
    auto* ctx = context.get();
    POWERSERVE_ASSERT(ctx && kernel_manager);

    auto* w_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(w)->get<BaseBuffer>());
    auto* x_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(x)->get<BaseBuffer>());
    auto* d_cl = dynamic_cast<OpenCLBuffer*>(&const_cast<Tensor*>(dst)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_cl && x_cl && d_cl);

    const int K = (int)w->m_shape[0];
    const int N = (int)w->m_shape[1];
    const int M = (int)x->m_shape[1];

    cl_kernel k = kernel_manager->get_kernel("kernel_mul_mat_f32_f32_simple");
    POWERSERVE_ASSERT(k && "kernel_mul_mat_f32_f32_simple not found");

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

    // We only support: (weight: FP16/FP32/Q4_0/Q8_0) x (activations: FP32) -> FP32
    if (dst->m_dtype != DataType::FP32 || src1->m_dtype != DataType::FP32) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: only supports dst=FP32 and src1=FP32 (got dst={}, src1={})",
                         (int)dst->m_dtype, (int)src1->m_dtype);
    }
    if (!(src0->m_dtype == DataType::FP16 || src0->m_dtype == DataType::FP32 ||
          src0->m_dtype == DataType::GGML_Q4_0 || src0->m_dtype == DataType::GGML_Q8_0)) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: unsupported weight dtype {} (no ggml fallback)", (int)src0->m_dtype);
    }

    // // fallback
    // if (src0->m_dtype == DataType::FP32 || src0->m_dtype == DataType::GGML_Q8_0) {
    //     if (!m_ggml_fallback) {
    //         POWERSERVE_ABORT("OpenCLBackend::matmul: ggml fallback not initialized for FP32/Q8_0 alignment");
    //     }
    //     matmul_cpu_ggml_fallback(dst, src0, src1);
    //     return;
    // }

    // Shapes: w=[K,N], x=[K,M], dst=[N,M]
    const int K  = (int)src0->m_shape[0];
    const int N  = (int)src0->m_shape[1];
    const int Kx = (int)src1->m_shape[0];
    const int M  = (int)src1->m_shape[1];
    const size_t B2 = dst->m_shape[2];
    const size_t B3 = dst->m_shape[3];

    if (Kx != K) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: K mismatch w.K={} x.K={}", K, Kx);
    }
    if ((int)dst->m_shape[0] != N || (int)dst->m_shape[1] != M) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: dst shape mismatch, expected [N,M]=[{},{}], got [{},{}]",
                         N, M, (int)dst->m_shape[0], (int)dst->m_shape[1]);
    }
    if (src1->m_shape[2] != B2 || src1->m_shape[3] != B3) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: src1 batch dims mismatch dst (src1=[{},{}], dst=[{},{}])",
                         (int)src1->m_shape[2], (int)src1->m_shape[3], (int)B2, (int)B3);
    }
    if (!::powerserve::tensor_can_mul_mat(src0, src1)) {
        POWERSERVE_ABORT("OpenCLBackend::matmul: src0 batch dims not broadcastable to src1");
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

    auto *w_buf = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(w_dev)->get<BaseBuffer>());
    auto *x_buf = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(x_use)->get<BaseBuffer>());
    auto *d_buf = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(dst_use)->get<BaseBuffer>());
    POWERSERVE_ASSERT(w_buf && x_buf && d_buf);

    const Shape w_slice_shape{ w_dev->m_shape[0], w_dev->m_shape[1], 1, 1 };
    const Shape x_slice_shape{ x_use->m_shape[0], x_use->m_shape[1], 1, 1 };
    const Shape d_slice_shape{ dst_use->m_shape[0], dst_use->m_shape[1], 1, 1 };

    auto slice_bytes_for = [](const Tensor *base) -> size_t {
        if (base->m_dtype == DataType::GGML_Q4_0 || base->m_dtype == DataType::GGML_Q8_0) {
            const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(base->m_dtype);
            return (size_t)ggml_row_size(gt, (int64_t)base->m_shape[0]) * (size_t)base->m_shape[1];
        }
        return powerserve::get_type_size(base->m_dtype) * (size_t)base->m_shape[0] * (size_t)base->m_shape[1];
    };

    auto make_slice = [](const Tensor *base,
                         OpenCLBuffer *buf,
                         size_t offset_bytes,
                         size_t slice_bytes,
                         const Shape &shape) {
        Tensor slice(base->m_dtype, shape);
        auto view = std::make_shared<OpenCLBuffer>(
            buf->m_stride,
            buf->get_device_buffer(),
            slice_bytes,
            buf->memory_pool,
            /*owns_buffer=*/false,
            buf->is_pooled(),
            buf->get_base_offset() + offset_bytes
        );
        slice.m_data = std::move(view);
        return slice;
    };

    const size_t w_slice_bytes = slice_bytes_for(w_dev);
    const size_t x_slice_bytes = slice_bytes_for(x_use);
    const size_t d_slice_bytes = slice_bytes_for(dst_use);

    const size_t w_stride2 = w_slice_bytes;
    const size_t w_stride3 = w_slice_bytes * w_dev->m_shape[2];
    const size_t x_stride2 = x_slice_bytes;
    const size_t x_stride3 = x_slice_bytes * x_use->m_shape[2];
    const size_t d_stride2 = d_slice_bytes;
    const size_t d_stride3 = d_slice_bytes * dst_use->m_shape[2];

    const size_t r2 = B2 / src0->m_shape[2];
    const size_t r3 = B3 / src0->m_shape[3];

    for (size_t i3 = 0; i3 < B3; ++i3) {
        const size_t w_i3 = (src0->m_shape[3] == 1) ? 0 : (i3 / r3);
        const size_t x_i3 = i3;
        const size_t d_i3 = i3;

        for (size_t i2 = 0; i2 < B2; ++i2) {
            const size_t w_i2 = (src0->m_shape[2] == 1) ? 0 : (i2 / r2);
            const size_t x_i2 = i2;
            const size_t d_i2 = i2;

            const size_t w_off = w_i2 * w_stride2 + w_i3 * w_stride3;
            const size_t x_off = x_i2 * x_stride2 + x_i3 * x_stride3;
            const size_t d_off = d_i2 * d_stride2 + d_i3 * d_stride3;

            Tensor w_slice = make_slice(w_dev, w_buf, w_off, w_slice_bytes, w_slice_shape);
            Tensor x_slice = make_slice(x_use, x_buf, x_off, x_slice_bytes, x_slice_shape);
            Tensor d_slice = make_slice(dst_use, d_buf, d_off, d_slice_bytes, d_slice_shape);

            // Dispatch by weight dtype (no ggml fallback)
            switch (w_dev->m_dtype) {
                case DataType::FP16:
                    matmul_opencl_f16_f32(&d_slice, &w_slice, &x_slice);
                    break;
                case DataType::FP32:
                    matmul_opencl_f32_f32(&d_slice, &w_slice, &x_slice);
                    break;
                case DataType::GGML_Q4_0:
                    matmul_opencl_q4_0_f32(&d_slice, &w_slice, &x_slice);
                    break;
                case DataType::GGML_Q8_0:
                    matmul_opencl_q8_0_f32(&d_slice, &w_slice, &x_slice);
                    break;
                default:
                    POWERSERVE_ABORT("OpenCLBackend::matmul: unreachable dtype {}", (int)w_dev->m_dtype);
            }
        }
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
    if (!initialized) {
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

    auto *self = const_cast<OpenCLBackend *>(this);
    auto *ctx = self->context.get();
    if (!ctx || !self->kernel_manager) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm missing OpenCL context/kernel manager");
        return;
    }

    Tensor tmp_x_dev;
    const Tensor *x_dev = ensure_contiguous_or_pack_f32(self, x, /*n_dims_check=*/4, tmp_x_dev);

    Tensor tmp_w_upload;
    const Tensor *w_dev = weight;
    auto *w_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(w_dev)->get<BaseBuffer>());
    if (!w_cl) {
        tmp_w_upload = Tensor(weight->m_dtype, weight->m_shape);
        tmp_w_upload.m_data = self->create_buffer(weight->m_shape, weight->m_dtype);
        if (!tmp_w_upload.m_data) {
            POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm failed to allocate weight buffer");
            return;
        }
        self->copy(&tmp_w_upload, weight);
        w_dev = &tmp_w_upload;
    }

    Tensor tmp_w_dev;
    w_dev = ensure_contiguous_or_pack_f32(self, w_dev, /*n_dims_check=*/4, tmp_w_dev);

    Tensor tmp_out_dev;
    const Tensor *out_dev = o;
    if (!self->is_contiguous(o, /*n_dims_check=*/4)) {
        tmp_out_dev = Tensor(o->m_dtype, o->m_shape);
        tmp_out_dev.m_data = self->create_buffer(o->m_shape, o->m_dtype);
        if (!tmp_out_dev.m_data) {
            POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm failed to allocate temp output buffer");
            return;
        }
        out_dev = &tmp_out_dev;
    }

    auto *x_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(x_dev)->get<BaseBuffer>());
    w_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(w_dev)->get<BaseBuffer>());
    auto *o_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(out_dev)->get<BaseBuffer>());
    if (!x_cl || !w_cl || !o_cl) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm requires OpenCLBuffer inputs");
        return;
    }

    cl_kernel kernel = self->kernel_manager->get_kernel("kernel_rms_norm_mul");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm kernel_rms_norm_mul not found");
        return;
    }

    cl_mem mem_x = x_cl->get_device_buffer();
    cl_mem mem_w = w_cl->get_device_buffer();
    cl_mem mem_o = o_cl->get_device_buffer();
    if (!mem_x || !mem_w || !mem_o) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rmsnorm invalid cl_mem buffers");
        return;
    }

    const int ne00 = (int)x_dev->m_shape[0];
    const int ne01 = (int)x_dev->m_shape[1];
    const int ne02 = (int)x_dev->m_shape[2];
    const int ne03 = (int)x_dev->m_shape[3];

    const int ne10 = (int)w_dev->m_shape[0];
    const int ne11 = (int)w_dev->m_shape[1];
    const int ne12 = (int)w_dev->m_shape[2];
    const int ne13 = (int)w_dev->m_shape[3];

    const auto x_stride = x_cl->get_stride();
    const cl_ulong nb01 = (cl_ulong)x_stride[1];
    const cl_ulong nb02 = (cl_ulong)x_stride[2];
    const cl_ulong nb03 = (cl_ulong)x_stride[3];

    const auto w_stride = w_cl->get_stride();
    const cl_ulong nb11 = (cl_ulong)w_stride[1];
    const cl_ulong nb12 = (cl_ulong)w_stride[2];
    const cl_ulong nb13 = (cl_ulong)w_stride[3];

    const auto o_stride = o_cl->get_stride();
    const cl_ulong nb1 = (cl_ulong)o_stride[1];
    const cl_ulong nb2 = (cl_ulong)o_stride[2];
    const cl_ulong nb3 = (cl_ulong)o_stride[3];

    const cl_ulong off_x = (cl_ulong)x_cl->get_base_offset();
    const cl_ulong off_w = (cl_ulong)w_cl->get_base_offset();
    const cl_ulong off_o = (cl_ulong)o_cl->get_base_offset();

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_x));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_x));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_w));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_w));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_o));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_o));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne00));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne01));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne02));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne03));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne10));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne11));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne12));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne13));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb11));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb12));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb13));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb2));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb3));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &eps));

    const uint32_t chunk = std::max<uint32_t>(1u, (uint32_t)(ne00 / 4));
    const uint32_t local_cap = std::min<uint32_t>(256u, chunk);
    const size_t local_size = (size_t)(1u << floor_log2_u32(std::max(1u, local_cap)));
    const bool use_fp64 = device_supports_fp64(ctx->get_device());
    const size_t local_mem_bytes = local_size * (use_fp64 ? sizeof(double) : sizeof(float));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, local_mem_bytes, nullptr));

    const size_t local[3]  = { local_size, 1, 1 };
    const size_t global[3] = {
        local_size * (size_t)ne01,
        (size_t)ne02,
        (size_t)ne03
    };

    OCL_RETURN_IF_ERROR(ctx, clEnqueueNDRangeKernel(ctx->get_queue(), kernel, 3, nullptr, global, local, 0, nullptr, nullptr));
    OCL_RETURN_IF_ERROR(ctx, clFinish(ctx->get_queue()));

    if (out_dev != o) {
        detail::cpy_tensor_cl(self, out_dev, o);
    }
}

void OpenCLBackend::rope(
    Tensor *out,
    const Tensor *src,
    const std::vector<int> &pos,
    const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    auto fallback_to_cpu = [&]() {
        if (!m_ggml_fallback) {
            POWERSERVE_LOG_ERROR("m_ggml_fallback is null (initialize() not called?)");
            return;
        }

        if (out->m_dtype != DataType::FP32 || src->m_dtype != DataType::FP32) {
            POWERSERVE_LOG_ERROR("OpenCLBackend::rope fallback only supports FP32");
            return;
        }

        Tensor host_x(DataType::FP32, src->m_shape);
        host_x.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
        this->copy(&host_x, src);

        Tensor host_y(DataType::FP32, out->m_shape);
        host_y.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

        m_ggml_fallback->rope(&host_y, &host_x, pos, rope_cfg);

        this->copy(out, &host_y);
    };

    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    if (!out || !src) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope got null tensor");
        return;
    }
    if (pos.empty()) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope got empty pos");
        return;
    }

    if (out->m_shape != src->m_shape) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope requires out.shape == src.shape");
        return;
    }

    auto *self = const_cast<OpenCLBackend *>(this);
    auto *ctx = self->context.get();
    if (!ctx || !self->kernel_manager) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope missing OpenCL context/kernel manager, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    const bool is_f32 = (src->m_dtype == DataType::FP32);
    const bool is_f16 = (src->m_dtype == DataType::FP16);
    if ((!is_f32 && !is_f16) || out->m_dtype != src->m_dtype) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope unsupported dtype, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    if (pos.size() != src->m_shape[2]) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::rope pos.size() {} != src shape[2] {}", pos.size(), src->m_shape[2]);
        fallback_to_cpu();
        return;
    }

    auto *src_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(src)->get<BaseBuffer>());
    auto *out_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(out)->get<BaseBuffer>());
    if (!src_cl || !out_cl) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope expects OpenCLBuffer, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    Shape pos_shape{pos.size(), 1, 1, 1};
    Tensor pos_host(DataType::INT32, pos_shape);
    pos_host.m_data = powerserve::CPUBuffer::create_buffer<int32_t>(pos_shape);
    auto &pos_cpu = pos_host.get<powerserve::CPUBuffer>();
    std::memcpy(pos_cpu.m_data, pos.data(), pos.size() * sizeof(int32_t));

    Tensor pos_dev(DataType::INT32, pos_shape);
    pos_dev.m_data = self->create_buffer(pos_shape, DataType::INT32);
    if (!pos_dev.m_data) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope failed to allocate pos buffer, fallback to CPU");
        fallback_to_cpu();
        return;
    }
    self->copy(&pos_dev, &pos_host);

    auto *pos_cl = dynamic_cast<OpenCLBuffer *>(&pos_dev.get<BaseBuffer>());
    if (!pos_cl) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope failed to create pos OpenCLBuffer, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    cl_mem X_cl = src_cl->get_device_buffer();
    cl_mem P_cl = pos_cl->get_device_buffer();
    cl_mem Y_cl = out_cl->get_device_buffer();
    if (!X_cl || !P_cl || !Y_cl) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope invalid cl_mem buffers, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    cl_mem S_cl = X_cl;
    const cl_ulong off0 = static_cast<cl_ulong>(src_cl->get_base_offset());
    const cl_ulong off1 = static_cast<cl_ulong>(pos_cl->get_base_offset());
    const cl_ulong off2 = off0;
    const cl_ulong offd = static_cast<cl_ulong>(out_cl->get_base_offset());

    const int ne00 = static_cast<int>(src->m_shape[0]);
    const int ne01 = static_cast<int>(src->m_shape[1]);
    const int ne02 = static_cast<int>(src->m_shape[2]);
    const int ne03 = static_cast<int>(src->m_shape[3]);

    const int ne0 = static_cast<int>(out->m_shape[0]);
    const int ne1 = static_cast<int>(out->m_shape[1]);
    const int ne2 = static_cast<int>(out->m_shape[2]);
    const int ne3 = static_cast<int>(out->m_shape[3]);

    const auto src_stride = src_cl->get_stride();
    const auto dst_stride = out_cl->get_stride();

    const cl_ulong nb00 = static_cast<cl_ulong>(src_stride[0]);
    const cl_ulong nb01 = static_cast<cl_ulong>(src_stride[1]);
    const cl_ulong nb02 = static_cast<cl_ulong>(src_stride[2]);
    const cl_ulong nb03 = static_cast<cl_ulong>(src_stride[3]);

    const cl_ulong nb0 = static_cast<cl_ulong>(dst_stride[0]);
    const cl_ulong nb1 = static_cast<cl_ulong>(dst_stride[1]);
    const cl_ulong nb2 = static_cast<cl_ulong>(dst_stride[2]);
    const cl_ulong nb3 = static_cast<cl_ulong>(dst_stride[3]);

    if (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        return;
    }

    RopeParams params{};
    params.n_past = 0;
    params.n_dims = rope_cfg.n_dims;
    params.n_ctx_orig = rope_cfg.n_ctx_orig;
    params.freq_base = rope_cfg.freq_base;
    params.freq_scale = rope_cfg.freq_scale;
    params.ext_factor = rope_cfg.ext_factor;
    params.attn_factor = rope_cfg.attn_factor;
    params.beta_fast = rope_cfg.beta_fast;
    params.beta_slow = rope_cfg.beta_slow;
    params.mode = rope_cfg.rope_type;
    const bool mode_is_default = params.mode < 0;
    const bool is_neox = mode_is_default ? true : ((params.mode & GGML_ROPE_TYPE_NEOX) != 0);
    const bool is_mrope = (!mode_is_default) && ((params.mode & GGML_ROPE_TYPE_MROPE) != 0);
    const bool is_vision = (!mode_is_default) && (params.mode == GGML_ROPE_TYPE_VISION);
    const int is_imrope = (params.mode == GGML_ROPE_TYPE_IMROPE) ? 1 : 0;

    cl_kernel kernel = nullptr;
    if (is_neox) {
        kernel = self->kernel_manager->get_kernel(is_f16 ? "kernel_rope_neox_f16" : "kernel_rope_neox_f32");
    } else if (is_mrope && !is_vision) {
        kernel = self->kernel_manager->get_kernel(is_f16 ? "kernel_rope_multi_f16" : "kernel_rope_multi_f32");
    } else if (is_vision) {
        kernel = self->kernel_manager->get_kernel(is_f16 ? "kernel_rope_vision_f16" : "kernel_rope_vision_f32");
    } else {
        kernel = self->kernel_manager->get_kernel(is_f16 ? "kernel_rope_norm_f16" : "kernel_rope_norm_f32");
    }

    if (!kernel) {
        POWERSERVE_LOG_WARN("OpenCLBackend::rope kernel not found, fallback to CPU");
        fallback_to_cpu();
        return;
    }

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &X_cl));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off0));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &P_cl));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &S_cl));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off2));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &Y_cl));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offd));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne00));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne01));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne02));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne03));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb00));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne0));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne2));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &ne3));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb0));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb1));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb2));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb3));

    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &params.n_past));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &params.n_dims));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &params.n_ctx_orig));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.freq_base));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.freq_scale));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.ext_factor));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.attn_factor));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.beta_fast));
    OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(float), &params.beta_slow));

    if (is_mrope || is_vision) {
        OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int32_t) * 4, params.sections));
    }
    if (is_mrope && !is_vision) {
        OCL_RETURN_IF_ERROR(ctx, clSetKernelArg(kernel, arg++, sizeof(int), &is_imrope));
    }

    const int nth = std::min(64, ne00);
    const size_t local[3]  = {static_cast<size_t>(nth), 1, 1};
    const size_t global[3] = {
        static_cast<size_t>(ne01) * static_cast<size_t>(nth),
        static_cast<size_t>(ne02),
        static_cast<size_t>(ne03)
    };

    OCL_RETURN_IF_ERROR(ctx, clEnqueueNDRangeKernel(
        ctx->get_queue(),
        kernel,
        3,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr
    ));
    OCL_RETURN_IF_ERROR(ctx, clFinish(ctx->get_queue()));
}

void OpenCLBackend::softmax(const Tensor * /*out*/, const Tensor * /*x*/) const {
    POWERSERVE_ABORT("OpenCLBackend::softmax TODO");
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

    auto *context = self->context.get();
    if (!context || !self->kernel_manager) {
        POWERSERVE_LOG_ERROR("softmax_ext: OpenCL context or kernel manager not initialized");
        return;
    }

    auto *x_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(x_dev)->get<BaseBuffer>());
    auto *m_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(m_dev)->get<BaseBuffer>());
    auto *o_cl = dynamic_cast<OpenCLBuffer *>(&const_cast<Tensor *>(out)->get<BaseBuffer>());
    if (!x_cl || !m_cl || !o_cl) {
        POWERSERVE_LOG_ERROR("softmax_ext: x/mask/out must be OpenCLBuffer");
        return;
    }

    cl_kernel kernel = self->kernel_manager->get_kernel("kernel_soft_max");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("softmax_ext: kernel_soft_max not found");
        return;
    }

    const cl_mem mem_x = x_cl->get_device_buffer();
    const cl_mem mem_m = m_cl->get_device_buffer();
    const cl_mem mem_o = o_cl->get_device_buffer();
    if (!mem_x || !mem_m || !mem_o) {
        POWERSERVE_LOG_ERROR("softmax_ext: invalid cl_mem buffers");
        return;
    }

    const cl_ulong off_x = (cl_ulong)x_cl->get_base_offset();
    const cl_ulong off_m = (cl_ulong)m_cl->get_base_offset();
    const cl_ulong off_o = (cl_ulong)o_cl->get_base_offset();

    const auto x_stride = x_cl->get_stride();
    const cl_ulong nb01 = (cl_ulong)x_stride[1];
    const cl_ulong nb02 = (cl_ulong)x_stride[2];
    const cl_ulong nb03 = (cl_ulong)x_stride[3];

    const auto m_stride = m_cl->get_stride();
    const cl_ulong nb11 = (cl_ulong)m_stride[1];
    const cl_ulong nb12 = (cl_ulong)m_stride[2];
    const cl_ulong nb13 = (cl_ulong)m_stride[3];

    const auto o_stride = o_cl->get_stride();
    const cl_ulong nb1 = (cl_ulong)o_stride[1];
    const cl_ulong nb2 = (cl_ulong)o_stride[2];
    const cl_ulong nb3 = (cl_ulong)o_stride[3];

    const int ne12 = (int)m_dev->m_shape[2];
    const int ne13 = (int)m_dev->m_shape[3];

    const uint32_t n_head = (uint32_t)ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)floor_log2_u32(n_head);
    const float m0 = std::pow(2.0f, -(max_bias)        / (float)n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_x));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_x));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_m));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_m));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_x));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_x));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_o));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off_o));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(int), &ne00));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(int), &ne12));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(int), &ne13));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb11));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb12));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb13));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb1));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb2));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb3));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(float), &scale));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(float), &max_bias));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(float), &m0));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(float), &m1));
    const int n_head_log2_i = (int)n_head_log2;
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, sizeof(int), &n_head_log2_i));

    const uint32_t local_cap = std::min<uint32_t>((uint32_t)ne00, 256u);
    const size_t local_work_size = (size_t)(1u << floor_log2_u32(std::max(1u, local_cap)));
    const size_t local_mem_size = local_work_size * sizeof(float);
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(kernel, arg++, local_mem_size, nullptr));

    const size_t global[3] = {
        static_cast<size_t>(ne01) * local_work_size,
        static_cast<size_t>(ne02),
        static_cast<size_t>(ne03)
    };
    const size_t local[3] = { local_work_size, 1, 1 };

    OCL_RETURN_IF_ERROR(context, clEnqueueNDRangeKernel(
        context->get_queue(), kernel,
        3, nullptr, global, local,
        0, nullptr, nullptr
    ));
    OCL_RETURN_IF_ERROR(context, clFinish(context->get_queue()));
}

} // namespace powerserve::opencl
