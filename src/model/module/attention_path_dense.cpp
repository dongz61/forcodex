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

#include <cmath>

namespace powerserve {

TensorNode *build_attention_scores_dense(
    Graph &g,
    TensorNode *q,
    TensorNode *k,
    TensorNode *v,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    size_t head_size,
    size_t n_head,
    size_t /*n_head_kv*/
) {
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

} // namespace powerserve
