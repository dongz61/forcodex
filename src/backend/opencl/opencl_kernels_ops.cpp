#include "backend/opencl/opencl_backend.hpp"
#include "backend/opencl/opencl_backend_helpers.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"

#include <CL/cl.h>
#include <cmath>
#include <exception>
#include <string>

namespace powerserve::opencl {

using detail::ensure_contiguous_or_pack_f32;

static inline size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
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

    {
        Tensor hb_cpu(DataType::FP32, hb->m_shape);
        hb_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(hb->m_shape);

        Tensor hb2_cpu(DataType::FP32, hb2->m_shape);
        hb2_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(hb2->m_shape);

        Tensor out_cpu(DataType::FP32, out->m_shape);
        out_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

        this->copy(&hb_cpu, hb);
        this->copy(&hb2_cpu, hb2);

        if (m_ggml_fallback) {
            m_ggml_fallback->silu_hadamard(&out_cpu, &hb_cpu, &hb2_cpu);
        } else {
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

} // namespace powerserve::opencl
