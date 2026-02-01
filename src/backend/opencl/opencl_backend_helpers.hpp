#pragma once

#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"

#include <CL/cl.h>

#define OCL_RETURN_IF_ERROR(ctx, call) \
    do { \
        cl_int _err = (call); \
        if (_err != CL_SUCCESS) { \
            POWERSERVE_LOG_ERROR("OpenCL error at {}:{} - {}: {}", \
                __FILE__, __LINE__, #call, (ctx)->get_error_string(_err)); \
            return; \
        } \
    } while (0)

namespace powerserve::opencl::detail {

static inline void cpy_tensor_cl(const OpenCLBackend* self,
                                 const Tensor* src,
                                 const Tensor* dst) {
    POWERSERVE_ASSERT(self && src && dst);

    auto* context = self->context.get();
    POWERSERVE_ASSERT(context != nullptr);

    auto* src_cl = dynamic_cast<OpenCLBuffer*>(
        &const_cast<Tensor*>(src)->get<BaseBuffer>());

    auto* dst_cl = dynamic_cast<OpenCLBuffer*>(
        &const_cast<Tensor*>(dst)->get<BaseBuffer>());

    if (!src_cl || !dst_cl) {
        POWERSERVE_LOG_ERROR("cpy_tensor_cl: src/dst not OpenCLBuffer");
        return;
    }

    if (src->m_shape != dst->m_shape) {
        POWERSERVE_ABORT(
            "cpy_tensor_cl: shape mismatch src=[{},{},{},{}] dst=[{},{},{},{}]",
            src->m_shape[0], src->m_shape[1], src->m_shape[2], src->m_shape[3],
            dst->m_shape[0], dst->m_shape[1], dst->m_shape[2], dst->m_shape[3]
        );
    }

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

    const int ne00 = (int)src->m_shape[0];
    const int ne01 = (int)src->m_shape[1];
    const int ne02 = (int)src->m_shape[2];
    const int ne03 = (int)src->m_shape[3];

    const int ne0  = (int)dst->m_shape[0];
    const int ne1  = (int)dst->m_shape[1];
    const int ne2  = (int)dst->m_shape[2];
    const int ne3  = (int)dst->m_shape[3];

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

    const cl_ulong off0 = (cl_ulong)src_cl->get_base_offset();
    const cl_ulong offd = (cl_ulong)dst_cl->get_base_offset();

    cl_uint arg = 0;
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_mem),   &src_mem));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &off0));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_mem),   &dst_mem));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &offd));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne00));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne01));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne02));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne03));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb00));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb01));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb02));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb03));

    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne0));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne1));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne2));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(int),      &ne3));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb0));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb1));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2));
    OCL_RETURN_IF_ERROR(context, clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb3));

    const size_t local[3]  = { 1, 1, 1 };
    const size_t global[3] = { (size_t)ne01, (size_t)ne02, (size_t)ne03 };

    OCL_RETURN_IF_ERROR(context, clEnqueueNDRangeKernel(self->context->get_queue(),
                                                        k, 3, nullptr, global, local,
                                                        0, nullptr, nullptr));
    OCL_RETURN_IF_ERROR(context, clFinish(self->context->get_queue()));
}

static inline const Tensor *ensure_contiguous_or_pack_f32(
    powerserve::opencl::OpenCLBackend *self,
    const Tensor *src,
    int n_dims_check,
    Tensor &tmp_dev
) {
    POWERSERVE_ASSERT(self && src);
    if (self->is_contiguous(src, n_dims_check)) {
        return src;
    }

    tmp_dev = Tensor(src->m_dtype, src->m_shape);
    tmp_dev.m_data = self->create_buffer(src->m_shape, src->m_dtype);
    if (!tmp_dev.m_data) {
        POWERSERVE_LOG_ERROR("ensure_contiguous: failed to allocate temp OpenCL buffer");
        return src;
    }

    cpy_tensor_cl(self, src, &tmp_dev);

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

    Tensor tmp_dev;
    const Tensor *src_packed = ensure_contiguous_or_pack_f32(self, src, /*n_dims_check=*/4, tmp_dev);

    Tensor host_contig(DataType::FP32, src->m_shape);
    host_contig.m_data = powerserve::CPUBuffer::create_buffer<float>(src->m_shape);
    self->copy(&host_contig, src_packed);

    self->copy(dst_contig_dev, &host_contig);
}

} // namespace powerserve::opencl::detail
