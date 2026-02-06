#include "backend/opencl/opencl_backend.hpp"
#include "backend/opencl/opencl_backend_helpers.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"
#include "ggml.h"

#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <execinfo.h>
#include <vector>

namespace powerserve::opencl {

using detail::cpy_tensor_cl;

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

static inline bool is_ggml_quant_dtype(powerserve::DataType dt) {
    using powerserve::DataType;
    return dt == DataType::GGML_Q4_0 || dt == DataType::GGML_Q8_0;
}

static inline size_t ggml_compat_nbytes(powerserve::DataType dt, const powerserve::Shape &s) {
    const size_t ne0 = (size_t)s[0];
    const size_t ne1 = (size_t)s[1];
    const size_t ne2 = (size_t)s[2];
    const size_t ne3 = (size_t)s[3];

    if (is_ggml_quant_dtype(dt)) {
        const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(dt);
        return (size_t)ggml_row_size(gt, (int64_t)ne0) * ne1 * ne2 * ne3;
    }

    const size_t elem = powerserve::get_type_size(dt);
    POWERSERVE_ASSERT(elem > 0);
    return elem * ne0 * ne1 * ne2 * ne3;
}

static inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        default: return 0;
    }
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

    if (src_cpu && dst_cl) {
        void* host = src_cpu->m_data;
        cl_mem dev = dst_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("H2D: invalid host/dev");
            return;
        }

        const bool src_contig = is_contiguous(src, 4);
        const bool dst_contig = is_contiguous(dst, 4);

        if (src_contig && dst_contig) {
            const size_t dst_off = dst_cl->get_base_offset();
            if (!memory_pool->copy_host_to_device(dev, host, src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("H2D: copy_host_to_device failed");
            }
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
            const size_t dst_off = dst_cl->get_base_offset();
            if (!memory_pool->copy_host_to_device(dev, host_src, src_bytes, dst_off)) {
                POWERSERVE_LOG_ERROR("H2D: copy_host_to_device failed");
            }
            return;
        }

        if (!is_cpy_kernel_supported(src->m_dtype) || !is_cpy_kernel_supported(dst->m_dtype)) {
            POWERSERVE_LOG_ERROR("H2D: non-contiguous copy requires cpy kernel, unsupported dtype src={} dst={}",
                                (int)src->m_dtype, (int)dst->m_dtype);
            return;
        }

        Tensor staging = make_contig_dev_tensor(const_cast<OpenCLBackend*>(this),
                                        dst->m_dtype,
                                        dst->m_shape);

        auto &staging_buf = staging.get<OpenCLBuffer>();
        cl_mem staging_mem = staging_buf.get_device_buffer();
        const size_t st_off = staging_buf.get_base_offset();
        if (!memory_pool->copy_host_to_device(staging_mem, host_src, src_bytes, st_off)) {
            POWERSERVE_LOG_ERROR("H2D: staging copy_host_to_device failed");
            dump_backtrace();
            std::abort();
        }

        cpy_tensor_cl(this, &staging, dst);
        return;
    }

    if (src_cl && dst_cpu) {
        void* host = dst_cpu->m_data;
        cl_mem dev = src_cl->get_device_buffer();
        if (!host || !dev) {
            POWERSERVE_LOG_ERROR("D2H: invalid host/dev");
            return;
        }

        const bool src_contig = is_contiguous(src, 4);
        const bool dst_contig = is_contiguous(dst, 4);

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

            cpy_tensor_cl(this, src, &staging);

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
            const size_t src_off = read_cl->get_base_offset();
            if (!memory_pool->copy_device_to_host(host, read_mem, src_bytes, src_off)) {
                POWERSERVE_LOG_ERROR("D2H: copy_device_to_host failed");
            }
            return;
        }

        std::vector<uint8_t> host_contig(src_bytes);
        const size_t src_off = read_cl->get_base_offset();
        if (!memory_pool->copy_device_to_host(host_contig.data(), read_mem, src_bytes, src_off)) {
            POWERSERVE_LOG_ERROR("D2H: staging copy_device_to_host failed");
            return;
        }

        if (!unpack_contig_to_cpu_strided(host_contig.data(), dst)) {
            POWERSERVE_LOG_ERROR("D2H: failed to unpack to CPU strided tensor");
        }
        return;
    }

    if (src_cl && dst_cl) {
        const size_t src_off = src_cl->get_base_offset();
        const size_t dst_off = dst_cl->get_base_offset();

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

    if (is_contiguous(x, 4)) {
        this->copy(out, x);
        return;
    }

    Tensor tmp = make_contig_dev_tensor(const_cast<OpenCLBackend*>(this),
                                        x->m_dtype,
                                        x->m_shape);
    cpy_tensor_cl(this, x, &tmp);

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

    if (src_off == 0 && dst_off == 0) {
        if (!memory_pool->copy_device_to_device(dst_dev, src_dev, x_bytes)) {
            POWERSERVE_ABORT("cont: copy_device_to_device failed");
        }
        return;
    }

    std::vector<uint8_t> host(x_bytes);
    if (!memory_pool->copy_device_to_host(host.data(), src_dev, x_bytes, src_off)) {
        POWERSERVE_ABORT("cont: copy_device_to_host failed");
    }
    if (!memory_pool->copy_host_to_device(dst_dev, host.data(), x_bytes, dst_off)) {
        POWERSERVE_ABORT("cont: copy_host_to_device failed");
    }
    clFinish(context->get_queue());
}

void OpenCLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    if (!initialized) POWERSERVE_ABORT("OpenCL backend not initialized");
    if (!out || !x)   POWERSERVE_ABORT("permute got null tensor");

    auto *dst = const_cast<Tensor*>(out);

    OpenCLBuffer *xbuf = nullptr;
    OpenCLBuffer *obuf = nullptr;
    try {
        xbuf = &const_cast<Tensor*>(x)->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::permute expects OpenCLBuffer input: {}", e.what());
        POWERSERVE_ABORT("permute: input not OpenCLBuffer");
    }

    if (!dst->m_data) {
        auto view = OpenCLBuffer::create_buffer_view<float>(*xbuf, dst->m_shape, /*offset=*/0);
        POWERSERVE_ASSERT(view && "permute: failed to create OpenCL view buffer");
        dst->m_data = std::static_pointer_cast<BaseBuffer>(view);
    }

    try {
        obuf = &dst->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::permute expects OpenCLBuffer output: {}", e.what());
        POWERSERVE_ABORT("permute: output not OpenCLBuffer");
    }

    Stride new_stride{};
    for (size_t i = 0; i < axes.size(); ++i) new_stride[i] = xbuf->m_stride[axes[i]];
    obuf->m_stride = new_stride;
}

void OpenCLBackend::transpose(const Tensor *out, const Tensor *x) const {
    POWERSERVE_ASSERT(out && x);
    POWERSERVE_ASSERT(out->m_data && x->m_data);

    auto *out_nc = const_cast<Tensor *>(out);
    auto *x_nc   = const_cast<Tensor *>(x);

    OpenCLBuffer *xbuf = nullptr;
    OpenCLBuffer *obuf = nullptr;
    try {
        xbuf = &x_nc->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::transpose expects OpenCLBuffer input: {}", e.what());
        POWERSERVE_ABORT("transpose: input not OpenCLBuffer");
    }

    if (!out_nc->m_data) {
        auto view = OpenCLBuffer::create_buffer_view<float>(*xbuf, out_nc->m_shape, /*offset=*/0);
        POWERSERVE_ASSERT(view && "transpose: failed to create OpenCL view buffer");
        out_nc->m_data = std::static_pointer_cast<BaseBuffer>(view);
    }

    try {
        obuf = &out_nc->get<OpenCLBuffer>();
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::transpose expects OpenCLBuffer output: {}", e.what());
        POWERSERVE_ABORT("transpose: output not OpenCLBuffer");
    }

    obuf->m_stride = xbuf->m_stride;
    std::swap(obuf->m_stride[0], obuf->m_stride[1]);
}

void OpenCLBackend::print(const Tensor* x, size_t size) const {
    POWERSERVE_ABORT("OpenCLBackend::print TODO");
}

} // namespace powerserve::opencl
