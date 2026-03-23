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

#pragma once

#include "graph/graph.hpp"
#include "model/module/attention_mask.hpp"

#include <vector>

namespace powerserve {

int get_ggml_cluster_topk();

TensorNode *build_attention_scores_dense(
    Graph &g,
    TensorNode *q,
    TensorNode *k,
    TensorNode *v,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    size_t head_size,
    size_t n_head,
    size_t n_head_kv
);

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
);

} // namespace powerserve
