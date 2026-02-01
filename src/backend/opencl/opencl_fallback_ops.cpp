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

    auto is_opencl = [](const Tensor *t) -> bool {
#if defined(POWERSERVE_WITH_OPENCL)
        return t && t->m_data &&
               dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get()) != nullptr;
#else
        (void)t;
        return false;
#endif
    };

    const bool all_f32 = (dst->m_dtype  == DataType::FP32 &&
                          src0->m_dtype == DataType::FP32 &&
                          src1->m_dtype == DataType::FP32);

    const bool all_opencl = is_opencl(dst) && is_opencl(src0) && is_opencl(src1);

    if (all_f32 && all_opencl) {
        Tensor tmpA_dev, tmpB_dev;
        const int n_dims_check = 4;

        const Tensor *A = ensure_contiguous_or_pack_f32(self, src0, n_dims_check, tmpA_dev);
        const Tensor *B = ensure_contiguous_or_pack_f32(self, src1, n_dims_check, tmpB_dev);

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

    const Tensor *A_cpu = src0;
    const Tensor *B_cpu = src1;

    Tensor tmpB_cpu;
    if (is_opencl(src1)) {
        tmpB_cpu = Tensor(DataType::FP32, src1->m_shape);
        tmpB_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(src1->m_shape);
        self->copy(&tmpB_cpu, src1);
        B_cpu = &tmpB_cpu;
    }

    self->matmul_cpu_ggml_fallback(dst, A_cpu, B_cpu);
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
