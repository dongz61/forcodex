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

TensorNode *build_attention_scores_topk(
    Graph &g,
    TensorNode *q,
    TensorNode *k,
    TensorNode *v,
    const std::vector<int> &pos,
    size_t head_size,
    size_t n_head,
    size_t n_head_kv,
    int topk
) {
    const float kq_scale = 1.0f / std::sqrt(float(head_size));
    return g.topk_attn(
        q,
        k,
        v,
        pos,
        kq_scale,
        topk,
        static_cast<int>(n_head),
        static_cast<int>(n_head_kv),
        static_cast<int>(head_size)
    );
}

} // namespace powerserve

