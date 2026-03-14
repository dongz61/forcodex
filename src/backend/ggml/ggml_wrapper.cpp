// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ggml-quants.h"
#include "ggml.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace powerserve::ggml {

namespace {
using Clock = std::chrono::steady_clock;

struct TopKProfileGlobal {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> select_ns{0};
    std::atomic<uint64_t> softmax_ns{0};
    std::atomic<uint64_t> reduce_ns{0};
};

TopKProfileGlobal g_topk_profile;

ALWAYS_INLINE void maybe_log_topk_profile_every_1000() {
    const uint64_t calls = g_topk_profile.calls.load(std::memory_order_relaxed);
    if (calls == 0 || (calls % 1000) != 0) {
        return;
    }

    const uint64_t total_ns = g_topk_profile.total_ns.load(std::memory_order_relaxed);
    const uint64_t select_ns = g_topk_profile.select_ns.load(std::memory_order_relaxed);
    const uint64_t softmax_ns = g_topk_profile.softmax_ns.load(std::memory_order_relaxed);
    const uint64_t reduce_ns = g_topk_profile.reduce_ns.load(std::memory_order_relaxed);

    const double calls_d = static_cast<double>(calls);
    const double avg_total_ms = static_cast<double>(total_ns) / calls_d / 1e6;
    const double avg_select_ms = static_cast<double>(select_ns) / calls_d / 1e6;
    const double avg_softmax_ms = static_cast<double>(softmax_ns) / calls_d / 1e6;
    const double avg_reduce_ms = static_cast<double>(reduce_ns) / calls_d / 1e6;

    const double stage_sum = static_cast<double>(select_ns + softmax_ns + reduce_ns);
    const double pct_select = stage_sum > 0.0 ? (100.0 * static_cast<double>(select_ns) / stage_sum) : 0.0;
    const double pct_softmax = stage_sum > 0.0 ? (100.0 * static_cast<double>(softmax_ns) / stage_sum) : 0.0;
    const double pct_reduce = stage_sum > 0.0 ? (100.0 * static_cast<double>(reduce_ns) / stage_sum) : 0.0;

    POWERSERVE_LOG_INFO(
        "TOPK_ATTN profile calls={} avg_total={:.3f}ms avg_select={:.3f}ms avg_softmax={:.3f}ms avg_reduce={:.3f}ms "
        "stage_share(select/softmax/reduce)={:.1f}%/{:.1f}%/{:.1f}%",
        calls,
        avg_total_ms,
        avg_select_ms,
        avg_softmax_ms,
        avg_reduce_ms,
        pct_select,
        pct_softmax,
        pct_reduce
    );
}

ALWAYS_INLINE bool topk_force_scalar() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char *v = std::getenv("POWERSERVE_TOPK_FORCE_SCALAR");
    cached = (v && (
        std::strcmp(v, "1") == 0 ||
        std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "TRUE") == 0 ||
        std::strcmp(v, "on") == 0 ||
        std::strcmp(v, "ON") == 0
    )) ? 1 : 0;
    return cached == 1;
}

ALWAYS_INLINE float dot_f32_scalar_contig(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[static_cast<size_t>(i)] * b[static_cast<size_t>(i)];
    }
    return sum;
}

#if defined(__AVX2__)
ALWAYS_INLINE float dot_f32_avx2(const float *a, const float *b, int n) {
    int i = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        vsum = _mm256_fmadd_ps(va, vb, vsum);
#else
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, vsum);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; ++i) {
        sum += a[static_cast<size_t>(i)] * b[static_cast<size_t>(i)];
    }
    return sum;
}
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
ALWAYS_INLINE float dot_f32_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
#if defined(__aarch64__) && defined(__ARM_FEATURE_FMA)
        vsum = vfmaq_f32(vsum, va, vb);
#else
        vsum = vmlaq_f32(vsum, va, vb);
#endif
    }
#if defined(__aarch64__)
    float sum = vaddvq_f32(vsum);
#else
    float32x2_t sum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    sum2 = vpadd_f32(sum2, sum2);
    float sum = vget_lane_f32(sum2, 0);
#endif
    for (; i < n; ++i) {
        sum += a[static_cast<size_t>(i)] * b[static_cast<size_t>(i)];
    }
    return sum;
}
#endif

ALWAYS_INLINE float dot_f32_contig(const float *a, const float *b, int n) {
    if (topk_force_scalar()) {
        return dot_f32_scalar_contig(a, b, n);
    }
#if defined(__AVX2__)
    return dot_f32_avx2(a, b, n);
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return dot_f32_neon(a, b, n);
#else
    return dot_f32_scalar_contig(a, b, n);
#endif
}

struct TopKAttnLayout {
    const char *q_data = nullptr;
    const char *k_data = nullptr;
    const char *v_data = nullptr;
    float *out_data = nullptr;

    size_t q_s0 = 0;
    size_t q_s1 = 0;
    size_t q_s2 = 0;
    size_t k_s0 = 0;
    size_t k_s1 = 0;
    size_t k_s2 = 0;
    size_t v_s0 = 0;
    size_t v_s1 = 0;
    size_t v_s2 = 0;

    int head_size = 0;
    int n_heads = 0;
    int q_per_kv = 0;
    float scale = 1.0f;
};

ALWAYS_INLINE void load_query_local(
    const TopKAttnLayout &layout,
    size_t b,
    int qh,
    std::vector<float> &q_local
) {
    const char *q_base = layout.q_data + b * layout.q_s1 + static_cast<size_t>(qh) * layout.q_s2;
    for (int d = 0; d < layout.head_size; ++d) {
        q_local[static_cast<size_t>(d)] =
            *reinterpret_cast<const float *>(q_base + static_cast<size_t>(d) * layout.q_s0);
    }
}

ALWAYS_INLINE void select_topk(
    const TopKAttnLayout &layout,
    int kvh,
    int n_kv,
    int k_use,
    const std::vector<float> &q_local,
    std::vector<std::pair<float, int>> &best
) {
    auto heap_cmp = [](const std::pair<float, int> &a, const std::pair<float, int> &b) { return a.first > b.first; };
    best.clear();

    for (int t_idx = 0; t_idx < n_kv; ++t_idx) {
        const char *k_base = layout.k_data + static_cast<size_t>(t_idx) * layout.k_s1 + static_cast<size_t>(kvh) * layout.k_s2;
        float dot = 0.0f;
        if (layout.k_s0 == sizeof(float)) {
            const auto *k_ptr = reinterpret_cast<const float *>(k_base);
            dot = dot_f32_contig(q_local.data(), k_ptr, layout.head_size);
        } else {
            for (int d = 0; d < layout.head_size; ++d) {
                const float kv = *reinterpret_cast<const float *>(k_base + static_cast<size_t>(d) * layout.k_s0);
                dot += q_local[static_cast<size_t>(d)] * kv;
            }
        }
        const float score = dot * layout.scale;

        if (static_cast<int>(best.size()) < k_use) {
            best.emplace_back(score, t_idx);
            std::push_heap(best.begin(), best.end(), heap_cmp);
        } else if (score > best.front().first) {
            std::pop_heap(best.begin(), best.end(), heap_cmp);
            best.back() = {score, t_idx};
            std::push_heap(best.begin(), best.end(), heap_cmp);
        }
    }
}

ALWAYS_INLINE float softmax_topk(
    const std::vector<std::pair<float, int>> &best,
    std::vector<float> &probs
) {
    float smax = -std::numeric_limits<float>::infinity();
    for (const auto &p : best) {
        smax = std::max(smax, p.first);
    }

    probs.resize(best.size());
    float denom = 0.0f;
    for (size_t i = 0; i < best.size(); ++i) {
        probs[i] = std::exp(best[i].first - smax);
        denom += probs[i];
    }
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return 1.0f / denom;
}

ALWAYS_INLINE void reduce_topk_values(
    const TopKAttnLayout &layout,
    size_t b,
    int qh,
    int kvh,
    const std::vector<std::pair<float, int>> &best,
    const std::vector<float> &probs,
    float inv_denom
) {
    float *out_ptr =
        layout.out_data +
        b * static_cast<size_t>(layout.n_heads * layout.head_size) +
        static_cast<size_t>(qh * layout.head_size);

    for (int d = 0; d < layout.head_size; ++d) {
        float acc = 0.0f;
        for (size_t i = 0; i < best.size(); ++i) {
            const int t_idx = best[i].second;
            const char *v_ptr =
                layout.v_data +
                static_cast<size_t>(t_idx) * layout.v_s0 +
                static_cast<size_t>(d) * layout.v_s1 +
                static_cast<size_t>(kvh) * layout.v_s2;
            const float vv = *reinterpret_cast<const float *>(v_ptr);
            acc += (probs[i] * inv_denom) * vv;
        }
        out_ptr[static_cast<size_t>(d)] = acc;
    }
}

} // namespace

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        params.thread_pool = (void *)m_thread_pool.get();
        params.barrier_fn  = [](void *opaque) {
            auto thread_pool = (ThreadPool *)opaque;
            thread_pool->barrier();
        };
        params.current_chunk = (atomic_int *)&m_current_chunk;

        powerserve_compute_forward_mul_mat(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
}

void GGMLBackend::rmsnorm(const Tensor *out, const Tensor *x, const Tensor *weight, float eps) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(weight);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rms_norm(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), eps);
    });
}

void GGMLBackend::softmax(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_soft_max(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::rope(
    Tensor *out, const Tensor *src, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(src);
    auto src1_tensor = std::make_unique<ggml_tensor>();
    {
        src1_tensor->data  = (void *)pos.data();
        src1_tensor->type  = GGML_TYPE_I32;
        src1_tensor->ne[0] = pos.size();
        src1_tensor->ne[1] = src1_tensor->ne[2] = src1_tensor->ne[3] = 1;
        src1_tensor->nb[0]                                           = sizeof(int32_t);
        src1_tensor->nb[1] = src1_tensor->nb[2] = src1_tensor->nb[3] = pos.size() * sizeof(int32_t);
    }

    rope_compute_params rope_params = {
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow,
        .mode        = rope_cfg.rope_type,
    };

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rope(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), nullptr, &rope_params
        );
    });
}

void GGMLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_add(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
}

void GGMLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    stride[axes[0]] = x->get<CPUBuffer>().m_stride[0];
    stride[axes[1]] = x->get<CPUBuffer>().m_stride[1];
    stride[axes[2]] = x->get<CPUBuffer>().m_stride[2];
    stride[axes[3]] = x->get<CPUBuffer>().m_stride[3];

    out->get<CPUBuffer>().m_stride = stride;
}

void GGMLBackend::cont(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::copy(const Tensor *dst, const Tensor *src) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias)
    const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(mask);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_softmax_ext(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), scale, max_bias
        );
    });
}

void GGMLBackend::topk_attn(
    const Tensor *out,
    const Tensor *q,
    const Tensor *k,
    const Tensor *v,
    const std::vector<int> &pos,
    float scale,
    int topk,
    int n_heads,
    int n_kv_heads,
    int head_size
) const {
    const auto topk_t0 = Clock::now();

    POWERSERVE_ASSERT(out && q && k && v);
    POWERSERVE_ASSERT(out->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(q->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(k->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(v->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(topk > 0);
    POWERSERVE_ASSERT(n_heads > 0);
    POWERSERVE_ASSERT(n_kv_heads > 0);
    POWERSERVE_ASSERT(head_size > 0);
    POWERSERVE_ASSERT((n_heads % n_kv_heads) == 0);
    POWERSERVE_ASSERT(pos.size() == q->m_shape[1]);
    POWERSERVE_ASSERT(k->m_shape[0] == static_cast<size_t>(head_size));
    POWERSERVE_ASSERT(k->m_shape[1] == v->m_shape[0]);
    POWERSERVE_ASSERT(v->m_shape[1] == static_cast<size_t>(head_size));
    POWERSERVE_ASSERT(k->m_shape[2] == static_cast<size_t>(n_kv_heads));
    POWERSERVE_ASSERT(v->m_shape[2] == static_cast<size_t>(n_kv_heads));

    auto *out_data = reinterpret_cast<float *>(out->get<CPUBuffer>().m_data);
    std::fill(out_data, out_data + out->n_elements(), 0.0f);

    const size_t batch = q->m_shape[1];
    const int q_per_kv = n_heads / n_kv_heads;
    const auto &q_buf = q->get<CPUBuffer>();
    const auto &k_buf = k->get<CPUBuffer>();
    const auto &v_buf = v->get<CPUBuffer>();

    TopKAttnLayout layout{
        .q_data = reinterpret_cast<const char *>(q_buf.m_data),
        .k_data = reinterpret_cast<const char *>(k_buf.m_data),
        .v_data = reinterpret_cast<const char *>(v_buf.m_data),
        .out_data = out_data,
        .q_s0 = q_buf.m_stride[0],
        .q_s1 = q_buf.m_stride[1],
        .q_s2 = q_buf.m_stride[2],
        .k_s0 = k_buf.m_stride[0],
        .k_s1 = k_buf.m_stride[1],
        .k_s2 = k_buf.m_stride[2],
        .v_s0 = v_buf.m_stride[0],
        .v_s1 = v_buf.m_stride[1],
        .v_s2 = v_buf.m_stride[2],
        .head_size = head_size,
        .n_heads = n_heads,
        .q_per_kv = q_per_kv,
        .scale = scale,
    };

    const size_t total_queries = batch * static_cast<size_t>(n_heads);
    const size_t n_threads = m_thread_pool->size();
    std::atomic<uint64_t> call_select_ns{0};
    std::atomic<uint64_t> call_softmax_ns{0};
    std::atomic<uint64_t> call_reduce_ns{0};

    m_thread_pool->run([&](size_t thread_id) {
        std::vector<float> q_local(static_cast<size_t>(head_size));
        std::vector<std::pair<float, int>> best;
        best.reserve(static_cast<size_t>(topk));
        std::vector<float> probs;
        probs.reserve(static_cast<size_t>(topk));
        uint64_t local_select_ns = 0;
        uint64_t local_softmax_ns = 0;
        uint64_t local_reduce_ns = 0;

        const size_t q_begin = (total_queries * thread_id) / n_threads;
        const size_t q_end = (total_queries * (thread_id + 1)) / n_threads;

        for (size_t qi = q_begin; qi < q_end; ++qi) {
            const size_t b = qi / static_cast<size_t>(n_heads);
            const int qh = static_cast<int>(qi % static_cast<size_t>(n_heads));
            const int kvh = qh / q_per_kv;
            const int n_kv = std::max(0, pos[b] + 1);
            const int k_use = std::min(topk, n_kv);
            if (k_use <= 0) {
                continue;
            }

            load_query_local(layout, b, qh, q_local);

            const auto ts0 = Clock::now();
            select_topk(layout, kvh, n_kv, k_use, q_local, best);
            const auto ts1 = Clock::now();
            const float inv_denom = softmax_topk(best, probs);
            const auto ts2 = Clock::now();
            if (inv_denom <= 0.0f) {
                local_select_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(ts1 - ts0).count()
                );
                local_softmax_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count()
                );
                continue;
            }
            reduce_topk_values(layout, b, qh, kvh, best, probs, inv_denom);
            const auto ts3 = Clock::now();

            local_select_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts1 - ts0).count()
            );
            local_softmax_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count()
            );
            local_reduce_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts3 - ts2).count()
            );
        }

        call_select_ns.fetch_add(local_select_ns, std::memory_order_relaxed);
        call_softmax_ns.fetch_add(local_softmax_ns, std::memory_order_relaxed);
        call_reduce_ns.fetch_add(local_reduce_ns, std::memory_order_relaxed);
    });

    const uint64_t total_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - topk_t0).count()
    );
    const uint64_t select_ns = call_select_ns.load(std::memory_order_relaxed);
    const uint64_t softmax_ns = call_softmax_ns.load(std::memory_order_relaxed);
    const uint64_t reduce_ns = call_reduce_ns.load(std::memory_order_relaxed);

    g_topk_profile.calls.fetch_add(1, std::memory_order_relaxed);
    g_topk_profile.total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    g_topk_profile.select_ns.fetch_add(select_ns, std::memory_order_relaxed);
    g_topk_profile.softmax_ns.fetch_add(softmax_ns, std::memory_order_relaxed);
    g_topk_profile.reduce_ns.fetch_add(reduce_ns, std::memory_order_relaxed);
    maybe_log_topk_profile_every_1000();
}

void GGMLBackend::get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    auto embd_tb = static_cast<char *>(weight->get<CPUBuffer>().m_data);
    auto dst_tb  = static_cast<float *>(dst->get<CPUBuffer>().m_data);

    auto dim        = dst->m_shape[0];
    auto batch_size = tokens.size();
    POWERSERVE_ASSERT(batch_size == dst->m_shape[1]);
    auto weight_strip = weight->get<CPUBuffer>().m_stride;

    for (size_t i = 0; i < batch_size; i++) {
        auto token = tokens[i];
        auto src   = embd_tb + weight_strip[1] * token;
        POWERSERVE_ASSERT(src < embd_tb + weight_strip[2]);
        switch (weight->m_dtype) {
        case DataType::FP32: {
            memcpy(dst_tb + i * dim, src, dim * sizeof(float));
        } break;

        case DataType::GGML_Q4_0: {
            dequantize_row_q4_0((block_q4_0 *)src, dst_tb + i * dim, dim);
        } break;

        case DataType::GGML_Q8_0: {
            dequantize_row_q8_0((block_q8_0 *)src, dst_tb + i * dim, dim);
        } break;

        default:
            POWERSERVE_ASSERT(false);
        }
    }
}

bool GGMLBackend::is_contiguous(const Tensor *tensor, int n) const {
    POWERSERVE_ASSERT(n >= 0 && n <= 2);
    if (n == 0) {
        return ggml_is_contiguous_0(convert_to_ggml(tensor).get());
    } else if (n == 1) {
        return ggml_is_contiguous_1(convert_to_ggml(tensor).get());
    } else if (n == 2) {
        return ggml_is_contiguous_2(convert_to_ggml(tensor).get());
    }
    return false;
}

int GGMLBackend::get_n_tasks(std::shared_ptr<OpNode> op) {
    int n_tasks = 1;

    switch (op->op) {
    // custom ops
    case OpType::SILU_HADAMARD:
    case OpType::ADD_CACHE:
    case OpType::PRINT:
    case OpType::VIEW:
    case OpType::TRANSPOSE:
    case OpType::COPY: {
        n_tasks = 1;
    } break;

    // ggml wrapper ops
    case OpType::PERMUTE:
    case OpType::GET_MASK:
    case OpType::GET_EMBEDDING: {
        n_tasks = 1;
    } break;

    case OpType::ROPE:
    case OpType::RMS_NORM:
    case OpType::CONT:
    case OpType::MAT_MUL:
    case OpType::ADD: {
        n_tasks = num_threads;
    } break;

    case OpType::SOFTMAX_EXT:
    case OpType::SOFTMAX: {
        n_tasks = std::min((int64_t)num_threads, op->prev[0]->tensor()->nrows());
    } break;
    case OpType::TOPK_ATTN: {
        const auto &params = op->get_params<TopKAttnParams>();
        const int64_t batch = static_cast<int64_t>(op->next[0]->tensor()->m_shape[1]);
        const int64_t work_items = batch * std::max(1, params.n_heads);
        n_tasks = std::max<int64_t>(1, std::min<int64_t>(num_threads, work_items));
    } break;

#if defined(POWERSERVE_WITH_QNN)
    case OpType::QNN_FORWARD: {
        n_tasks = 1;
    } break;
    case OpType::QNN_FORWARD_VL: {
        n_tasks = 1;
    } break;
#endif

    default: {
        fmt::println("op not implemented: {}", int(op->op));
        POWERSERVE_ASSERT(false);
    }
    }

    return n_tasks;
}

ggml_type GGMLBackend::get_vec_dot_type(const Tensor *tensor) {
    auto t = convert_to_ggml(tensor);
    return powerserve_get_vec_dot_type(t.get());
}

} // namespace powerserve::ggml
