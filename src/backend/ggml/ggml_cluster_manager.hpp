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

#include "backend/ggml/ggml_kv_cache.hpp"

#include <cstddef>
#include <vector>

namespace powerserve::ggml {

struct ClusterInfo {
    std::vector<int> token_positions;
    std::vector<float> center;
    float variance = 0.0f;
};

struct GGMLClusterManager {
public:
    explicit GGMLClusterManager(GGMLKV &kv);

    bool enabled() const;
    void clear();
    void build_layer_after_prefill(size_t layer_id);
    void build_all_layers_after_prefill();
    void update_layer_after_decode(size_t layer_id, int token_position, const std::vector<float> &new_k);

    auto get_layer_clusters(size_t layer_id) const -> const std::vector<ClusterInfo> &;

private:
    struct SplitSeeds {
        int first = -1;
        int second = -1;
    };

    auto current_token_count() const -> size_t;
    auto target_cluster_size() const -> size_t;
    auto max_cluster_variance() const -> float;
    auto make_cluster(size_t layer_id, const std::vector<int> &token_positions) const -> ClusterInfo;
    auto key_at(size_t layer_id, int token_position) const -> const float *;
    void recompute_cluster_stats(size_t layer_id, ClusterInfo &cluster) const;
    auto pick_best_cluster(size_t layer_id, const std::vector<float> &new_k) const -> size_t;
    auto maybe_split_cluster(size_t layer_id, ClusterInfo &cluster) const -> std::vector<ClusterInfo>;
    auto choose_split_seeds(size_t layer_id, const ClusterInfo &cluster) const -> SplitSeeds;

private:
    GGMLKV &m_kv;
    std::vector<std::vector<ClusterInfo>> m_layer_clusters;
};

} // namespace powerserve::ggml
