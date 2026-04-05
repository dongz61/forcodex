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
#include "backend/ggml/ggml_cluster_manager.hpp"
#include "backend/ggml/ggml_kv_pager.hpp"
#include "core/logger.hpp"
#include "model/module/ggml_cluster_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace powerserve::ggml {

namespace {
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
#if defined(__AVX2__)
    return dot_f32_avx2(a, b, n);
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return dot_f32_neon(a, b, n);
#else
    return dot_f32_scalar_contig(a, b, n);
#endif
}

struct ClusterQLayout {
    const char *q_data = nullptr;
    float *out_data = nullptr;
    size_t q_s0 = 0;
    size_t q_s1 = 0;
    size_t q_s2 = 0;
    int head_size = 0;
    int n_heads = 0;
    int q_per_kv = 0;
    float scale = 1.0f;
};

ALWAYS_INLINE void load_cluster_query_local(const ClusterQLayout &layout, int qh, std::vector<float> &q_local) {
    const char *q_base = layout.q_data + static_cast<size_t>(qh) * layout.q_s2;
    for (int d = 0; d < layout.head_size; ++d) {
        q_local[static_cast<size_t>(d)] =
            *reinterpret_cast<const float *>(q_base + static_cast<size_t>(d) * layout.q_s0);
    }
}

ALWAYS_INLINE void select_topk_cluster_ids(
    const std::vector<ClusterInfo> &clusters,
    int kvh,
    int head_size,
    int topk_clusters,
    float scale,
    const std::vector<float> &q_local,
    std::vector<int> &cluster_ids
) {
    auto heap_cmp = [](const std::pair<float, int> &a, const std::pair<float, int> &b) { return a.first > b.first; };
    std::vector<std::pair<float, int>> best;
    best.reserve(static_cast<size_t>(topk_clusters));

    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        const auto &center = clusters[cluster_id].center;
        const float *center_slice = center.data() + static_cast<size_t>(kvh * head_size);
        const float score = dot_f32_contig(q_local.data(), center_slice, head_size) * scale;
        if (static_cast<int>(best.size()) < topk_clusters) {
            best.emplace_back(score, static_cast<int>(cluster_id));
            std::push_heap(best.begin(), best.end(), heap_cmp);
        } else if (score > best.front().first) {
            std::pop_heap(best.begin(), best.end(), heap_cmp);
            best.back() = {score, static_cast<int>(cluster_id)};
            std::push_heap(best.begin(), best.end(), heap_cmp);
        }
    }

    cluster_ids.clear();
    cluster_ids.reserve(best.size());
    for (const auto &entry : best) {
        cluster_ids.push_back(entry.second);
    }
}

ALWAYS_INLINE void reduce_dense_values_from_cluster_views(
    float *out_ptr,
    const std::vector<ClusterView> &views,
    int head_size,
    int kvh,
    const std::vector<std::vector<float>> &probs_per_view,
    float inv_denom
) {
    for (int d = 0; d < head_size; ++d) {
        float acc = 0.0f;
        for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
            const auto &view = views[view_idx];
            const auto &probs = probs_per_view[view_idx];
            const size_t value_row = static_cast<size_t>(kvh * head_size + d) * view.token_count;
            for (size_t t = 0; t < view.token_count; ++t) {
                acc += (probs[t] * inv_denom) * view.v_ptr[value_row + t];
            }
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

void GGMLBackend::cluster_attn(
    const Tensor *out,
    const Tensor *q,
    const std::string &model_id,
    int layer_id,
    float scale,
    int topk_clusters,
    int n_heads,
    int n_kv_heads,
    int head_size
) const {
    POWERSERVE_ASSERT(out && q);
    POWERSERVE_ASSERT(out->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(q->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(q->m_shape[1] == 1, "CLUSTER_ATTN currently only supports batch_size == 1");
    POWERSERVE_ASSERT(topk_clusters > 0);
    POWERSERVE_ASSERT(n_heads > 0);
    POWERSERVE_ASSERT(n_kv_heads > 0);
    POWERSERVE_ASSERT(head_size > 0);
    POWERSERVE_ASSERT((n_heads % n_kv_heads) == 0);

    const auto runtime = get_cluster_runtime(model_id);
    POWERSERVE_ASSERT(runtime.manager != nullptr, "CLUSTER_ATTN missing cluster manager for model_id={}", model_id);
    const auto &clusters = runtime.manager->get_layer_clusters(static_cast<size_t>(layer_id));
    POWERSERVE_ASSERT(!clusters.empty(), "CLUSTER_ATTN requires non-empty clusters at layer={}", layer_id);

    auto *out_data = reinterpret_cast<float *>(out->get<CPUBuffer>().m_data);
    std::fill(out_data, out_data + out->n_elements(), 0.0f);

    const int q_per_kv = n_heads / n_kv_heads;
    const auto &q_buf = q->get<CPUBuffer>();
    ClusterQLayout layout{
        .q_data = reinterpret_cast<const char *>(q_buf.m_data),
        .out_data = out_data,
        .q_s0 = q_buf.m_stride[0],
        .q_s1 = q_buf.m_stride[1],
        .q_s2 = q_buf.m_stride[2],
        .head_size = head_size,
        .n_heads = n_heads,
        .q_per_kv = q_per_kv,
        .scale = scale,
    };

    std::vector<float> q_local(static_cast<size_t>(head_size));
    std::vector<int> cluster_ids;
    std::vector<char> selected(static_cast<size_t>(clusters.size()), 0);
    for (int qh = 0; qh < n_heads; ++qh) {
        const int kvh = qh / q_per_kv;
        load_cluster_query_local(layout, qh, q_local);
        select_topk_cluster_ids(
            clusters,
            kvh,
            head_size,
            std::min(topk_clusters, static_cast<int>(clusters.size())),
            scale,
            q_local,
            cluster_ids
        );
        for (int cluster_id : cluster_ids) {
            selected[static_cast<size_t>(cluster_id)] = 1;
        }
    }

    std::vector<int> selected_cluster_indices;
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        if (!selected[cluster_id]) {
            continue;
        }
        selected_cluster_indices.push_back(static_cast<int>(cluster_id));
    }
    POWERSERVE_ASSERT(!selected_cluster_indices.empty(), "CLUSTER_ATTN selected no clusters at layer={}", layer_id);

    std::vector<ClusterView> views;
    POWERSERVE_ASSERT(
        runtime.manager->query_cluster_views(static_cast<size_t>(layer_id), selected_cluster_indices, views),
        "CLUSTER_ATTN failed to query cluster views at layer={}",
        layer_id
    );
    POWERSERVE_ASSERT(!views.empty(), "CLUSTER_ATTN queried no cluster views at layer={}", layer_id);

    std::vector<std::vector<float>> probs_per_view;
    probs_per_view.resize(views.size());
    for (int qh = 0; qh < n_heads; ++qh) {
        const int kvh = qh / q_per_kv;
        load_cluster_query_local(layout, qh, q_local);

        float smax = -std::numeric_limits<float>::infinity();
        for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
            const auto &view = views[view_idx];
            auto &probs = probs_per_view[view_idx];
            probs.resize(view.token_count);
            for (size_t t = 0; t < view.token_count; ++t) {
                const float *k_slice =
                    view.k_ptr + t * static_cast<size_t>(n_kv_heads * head_size) + static_cast<size_t>(kvh * head_size);
                probs[t] = dot_f32_contig(q_local.data(), k_slice, head_size) * scale;
                smax = std::max(smax, probs[t]);
            }
        }

        float denom = 0.0f;
        for (auto &probs : probs_per_view) {
            for (float &prob : probs) {
                prob = std::exp(prob - smax);
                denom += prob;
            }
        }
        POWERSERVE_ASSERT(denom > 0.0f, "CLUSTER_ATTN softmax denom must be positive");

        float *out_ptr = out_data + static_cast<size_t>(qh * head_size);
        reduce_dense_values_from_cluster_views(out_ptr, views, head_size, kvh, probs_per_view, 1.0f / denom);
    }
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
    case OpType::CLUSTER_UPDATE:
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
    case OpType::CLUSTER_ATTN: {
        const auto &params = op->get_params<ClusterAttnParams>();
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
