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

#include "model/module/cluster_attention_path.hpp"

#include <cmath>
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

TensorNode *build_cluster_decode_attention_scores(
    Graph &g,
    TensorNode *rope_q,
    TensorNode *rope_k,
    TensorNode *v,
    int64_t layer_id,
    const std::vector<int> &pos,
    size_t head_size,
    size_t n_head,
    size_t n_head_kv
) {
    POWERSERVE_ASSERT(pos.size() == 1, "cluster decode path only supports batch_size == 1");
    auto q = g.permute(rope_q, {0, 2, 1, 3});
    g.cluster_update(rope_k, v, g.m_model_id, static_cast<int>(layer_id), pos.front());
    return g.cluster_attn(
        q,
        g.m_model_id,
        static_cast<int>(layer_id),
        1.0f / std::sqrt(float(head_size)),
        get_ggml_cluster_topk(),
        static_cast<int>(n_head),
        static_cast<int>(n_head_kv),
        static_cast<int>(head_size)
    );
}

} // namespace powerserve
