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

#include "attention_path.hpp"

#include "backend/cpu_buffer.hpp"
#include "model/module/ggml_cluster_runtime.hpp"

#include <cstdlib>

namespace powerserve {

int get_ggml_cluster_topk() {
    const char *v = std::getenv("POWERSERVE_GGML_CLUSTER_TOPK");
    if (!v || v[0] == '\0') {
        return 0;
    }
    const int k = std::atoi(v);
    return k > 0 ? k : 0;
}

TensorNode *build_attention_scores(
    Graph &g,
    TensorNode *rope_q,
    int64_t layer_id,
    const TensorNode *k_cache,
    const TensorNode *v_cache,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    size_t head_size,
    size_t n_head,
    size_t n_head_kv,
    size_t n_ctx
) {
    const size_t batch_size = pos.size();
    const size_t n_kv = static_cast<size_t>(pos.back() + 1);
    const size_t kv_gqa = head_size * n_head_kv;
    const int cluster_topk = get_ggml_cluster_topk();
    const auto cluster_runtime = ggml::get_cluster_runtime(g.m_model_id);
    const bool use_cluster = (cluster_topk > 0) &&
                             (batch_size == 1) &&
                             (cluster_runtime.manager != nullptr) &&
                             (cluster_runtime.pager != nullptr) &&
                             cluster_runtime.ready;

    if (use_cluster) {
        POWERSERVE_ASSERT(k_cache->m_data && v_cache->m_data, "POWERSERVE_GGML_CLUSTER_TOPK requires allocated KV buffers");
        auto *k_base = k_cache->m_data.get();
        auto *v_base = v_cache->m_data.get();
        POWERSERVE_ASSERT(
            dynamic_cast<CPUBuffer *>(k_base) != nullptr && dynamic_cast<CPUBuffer *>(v_base) != nullptr,
            "POWERSERVE_GGML_CLUSTER_TOPK is currently only supported on GGML/CPU KV cache tensors"
        );
    }

    // (head_size, bs, n_heads, 1)
    auto q = g.permute(rope_q, {0, 2, 1, 3});

    // {head_size, n_kv, n_head_kv, 1}
    auto k = g.view(
        k_cache,
        {head_size, n_kv, n_head_kv, 1},
        {
            k_cache->element_size(),
            k_cache->row_size(kv_gqa),
            k_cache->row_size(head_size),
            k_cache->row_size(head_size) * n_head_kv,
        }
    );

    // {n_kv, head_size, n_head_kv, 1}
    auto v = g.view(
        v_cache,
        {n_kv, head_size, n_head_kv, 1},
        {v_cache->element_size(),
         v_cache->element_size() * n_ctx,
         v_cache->element_size() * n_ctx * head_size,
         v_cache->element_size() * n_ctx * head_size * n_head_kv}
    );

    if (use_cluster) {
        return g.cluster_attn(
            q,
            g.m_model_id,
            static_cast<int>(layer_id),
            1.0f / std::sqrt(float(head_size)),
            cluster_topk,
            static_cast<int>(n_head),
            static_cast<int>(n_head_kv),
            static_cast<int>(head_size)
        );
    }
    return build_attention_scores_dense(g, q, k, v, pos, mask, head_size, n_head, n_head_kv);
}

} // namespace powerserve
