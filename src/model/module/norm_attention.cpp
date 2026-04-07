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

#include "norm_attention.hpp"

#include "cluster_attention_path.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/module/ggml_cluster_runtime.hpp"

#include <cstdint>
#include <cmath>

namespace powerserve {

namespace {

auto build_attention_scores_dense(
    Graph &g,
    TensorNode *q,
    TensorNode *k,
    TensorNode *v,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    size_t head_size,
    size_t n_head
) -> TensorNode * {
    const size_t n_kv = static_cast<size_t>(pos.back() + 1);
    const size_t batch_size = pos.size();
    const float kq_scale = 1.0f / std::sqrt(float(head_size));

    auto kq = g.mat_mul(k, q); // {bs, n_kv, n_head_kv, 1}
    constexpr float f_max_alibi_bias = 0.0f;
    auto kq_mask = g.get_mask(mask, {n_kv, batch_size, 1, 1}, pos);
    kq = g.softmax_ext(kq, kq_mask, kq_scale, f_max_alibi_bias);

    auto kqv = g.mat_mul(v, kq); // {head_size, n_kv, n_head_kv, 1}
    auto kqv_merged = g.permute(kqv, {0, 2, 1, 3}); // {head_size, n_head_kv, bs, 1}
    return g.cont(kqv_merged, {head_size * n_head, batch_size, 1, 1});
}

} // namespace

TensorNode *NormAttention::build(
    Graph &g,
    TensorNode *x, // {embd_dim, bs, 1, 1}
    int64_t L,
    const TensorNode *k_cache, // {seq_len * kv_dim, 1, 1, 1}
    const TensorNode *v_cache,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    bool is_need_bias
) {
    auto batch_size = pos.size();
    auto head_size  = m_config.head_size;

    // POWERSERVE_ASSERT(head_size == (size_t)m_config.rope_config.n_dims);
    // 不要 assert 这个啦！Qwen3 跑不动的！
    
    auto n_head    = m_config.n_heads;
    auto n_head_kv = m_config.n_kv_heads;
    auto n_ctx     = m_config.seq_len;
    size_t kv_gqa  = head_size * n_head_kv;
    size_t cur_pos = pos[0];

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);     // (embd_dim, 1, 1, 1)
    auto att_norm_o = g.rms_norm(x, att_norm_w, m_config.norm_eps); // (embd_dim, bs, 1, 1)

    // QKV
    auto q_w = g.add_tensor(m_weights->lw[L].attn_q); // (embd_dim, embd_dim, 1, 1)
    auto q   = g.mat_mul(q_w, att_norm_o);            // (embd_dim, bs, 1, 1)
    if (is_need_bias) {
        auto q_b = g.add_tensor(m_weights->lw[L].attn_q_bias); // (embd_dim, 1, 1, 1)
        q        = g.add(q, q_b);
    }
    // embd_dim == n_heads * head_size
    // kv_dim == n_kv_heads * head_size
    auto k_w = g.add_tensor(m_weights->lw[L].attn_k); // (embd_dim, kv_dim, 1, 1)
    auto k   = g.mat_mul(k_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    if (is_need_bias) {
        auto k_b = g.add_tensor(m_weights->lw[L].attn_k_bias); // (kv_dim, 1, 1, 1)
        k        = g.add(k, k_b);
    }

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v); // (embd_dim, kv_dim, 1, 1)
    auto v   = g.mat_mul(v_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    if (is_need_bias) {
        auto v_b = g.add_tensor(m_weights->lw[L].attn_v_bias); // (kv_dim, 1, 1, 1)
        v        = g.add(v, v_b);
    }

    // (head_size, n_heads, bs, 1)
    auto q_view = g.view_tensor(q, {head_size, n_head, q->m_shape[1], q->m_shape[2]});
    // (head_size, n_kv_heads, bs, 1)
    auto k_view = g.view_tensor(k, {head_size, n_head_kv, k->m_shape[1], k->m_shape[2]});

    /* lsh 修改起始 */
    TensorNode *q_final = q_view;
    TensorNode *k_final = k_view; 

    if(is_need_bias==false){ // for Qwen3 only
        auto q_norm_w = g.add_tensor(m_weights->lw[L].attn_q_norm); // weight shape: (128, 1, 1, 1)
        q_final = g.rms_norm(q_view, q_norm_w, m_config.norm_eps);  // input shape: (128, 16, bs, 1)

        auto k_norm_w = g.add_tensor(m_weights->lw[L].attn_k_norm); // weight shape: (128, 1, 1, 1)
        k_final = g.rms_norm(k_view, k_norm_w, m_config.norm_eps);  // input shape: (128, 8, bs, 1)
    }
    /* lsh 修改结束 */
    auto rope_q = g.rope(q_final, pos, m_config.rope_config); // (head_size, n_heads, bs, 1)
    auto rope_k = g.rope(k_final, pos, m_config.rope_config); // (head_size, n_kv_heads, bs, 1)

    POWERSERVE_ASSERT(k_cache != nullptr && v_cache != nullptr, "dense attention requires KV cache tensors");

    // store kv
    {
        k                 = rope_k;
        v                 = g.transpose(v);
        auto k_cache_view = g.view(
            k_cache,
            {batch_size * kv_gqa, 1, 1, 1},
            {k_cache->element_size(),
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa},
            k_cache->row_size(kv_gqa) * cur_pos
        );
        g.copy(k_cache_view, k);

        auto v_cache_view = g.view(
            v_cache,
            {batch_size, kv_gqa, 1, 1},
            {
                v_cache->element_size(),
                n_ctx * v_cache->element_size(),
                n_ctx * v_cache->element_size() * kv_gqa,
                n_ctx * v_cache->element_size() * kv_gqa,
            },
            v_cache->element_size() * cur_pos
        );
        g.copy(v_cache_view, v);
    }

    // (head_size, bs, n_heads, 1)
    auto q_att = g.permute(rope_q, {0, 2, 1, 3});

    const size_t n_kv = static_cast<size_t>(pos.back() + 1);
    // {head_size, n_kv, n_head_kv, 1}
    auto k_cache_view = g.view(
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
    auto v_cache_view = g.view(
        v_cache,
        {n_kv, head_size, n_head_kv, 1},
        {
            v_cache->element_size(),
            v_cache->element_size() * n_ctx,
            v_cache->element_size() * n_ctx * head_size,
            v_cache->element_size() * n_ctx * head_size * n_head_kv,
        }
    );

    TensorNode *att_scores =
        build_attention_scores_dense(g, q_att, k_cache_view, v_cache_view, pos, mask, head_size, n_head);

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(attn_output_w, att_scores); // (embd_dim, bs, 1, 1)

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

TensorNode *NormAttention::build_cluster_decode(
    Graph &g,
    TensorNode *x,
    int64_t L,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    bool is_need_bias
) {
    POWERSERVE_UNUSED(mask);
    POWERSERVE_ASSERT(pos.size() == 1, "cluster decode path only supports batch_size == 1");

    auto head_size  = m_config.head_size;
    auto n_head     = m_config.n_heads;
    auto n_head_kv  = m_config.n_kv_heads;

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);
    auto att_norm_o = g.rms_norm(x, att_norm_w, m_config.norm_eps);

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(q_w, att_norm_o);
    if (is_need_bias) {
        auto q_b = g.add_tensor(m_weights->lw[L].attn_q_bias);
        q        = g.add(q, q_b);
    }

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(k_w, att_norm_o);
    if (is_need_bias) {
        auto k_b = g.add_tensor(m_weights->lw[L].attn_k_bias);
        k        = g.add(k, k_b);
    }

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(v_w, att_norm_o);
    if (is_need_bias) {
        auto v_b = g.add_tensor(m_weights->lw[L].attn_v_bias);
        v        = g.add(v, v_b);
    }

    auto q_view = g.view_tensor(q, {head_size, n_head, q->m_shape[1], q->m_shape[2]});
    auto k_view = g.view_tensor(k, {head_size, n_head_kv, k->m_shape[1], k->m_shape[2]});

    TensorNode *q_final = q_view;
    TensorNode *k_final = k_view;
    if (is_need_bias == false) {
        auto q_norm_w = g.add_tensor(m_weights->lw[L].attn_q_norm);
        q_final = g.rms_norm(q_view, q_norm_w, m_config.norm_eps);

        auto k_norm_w = g.add_tensor(m_weights->lw[L].attn_k_norm);
        k_final = g.rms_norm(k_view, k_norm_w, m_config.norm_eps);
    }

    auto rope_q = g.rope(q_final, pos, m_config.rope_config);
    auto rope_k = g.rope(k_final, pos, m_config.rope_config);
    auto att_scores =
        build_cluster_decode_attention_scores(g, rope_q, rope_k, v, L, pos, head_size, n_head, n_head_kv);

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(attn_output_w, att_scores);
    return g.add(x, attn_o);
}

} // namespace powerserve
