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
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace powerserve::ggml {

struct GGMLKVPager;

enum class ClusterDirtyMode : uint8_t {
    Clean = 0,
    AppendOnly,
    RewriteRequired,
};

struct ClusterView {
    const float *k_ptr = nullptr; // [token][kv_dim]
    const float *v_ptr = nullptr; // [kv_dim][token]
    size_t token_count = 0;
};

struct PendingClusterToken {
    int position = -1;
    std::vector<float> key;
    std::vector<float> value;
};

struct ClusterInfo {
    int cluster_id = -1;
    std::vector<int> token_positions;
    std::vector<float> center;
    float variance = 0.0f;

    int64_t disk_offset = -1;
    size_t persisted_token_count = 0;
    size_t capacity_tokens = 0;

    bool resident = false;
    ClusterDirtyMode dirty_mode = ClusterDirtyMode::Clean;
    std::vector<float> resident_key;   // [token][kv_dim]
    std::vector<float> resident_value; // [kv_dim][token]
};

struct GGMLClusterManager {
public:
    explicit GGMLClusterManager(GGMLKV &kv);

    void attach_pager(GGMLKVPager *pager);

    bool enabled() const;
    void clear();
    void build_layer_after_prefill(size_t layer_id);
    void build_all_layers_after_prefill();
    void update_layer_after_decode(
        size_t layer_id,
        int token_position,
        const std::vector<float> &new_k,
        const std::vector<float> &new_v
    );

    auto get_layer_clusters(size_t layer_id) const -> const std::vector<ClusterInfo> &;
    bool query_cluster_views(size_t layer_id, const std::vector<int> &cluster_indices, std::vector<ClusterView> &views);

private:
    struct SplitSeeds {
        int first = -1;
        int second = -1;
    };

    using PendingBucketMap = std::unordered_map<int, std::vector<PendingClusterToken>>;

private:
    auto current_token_count() const -> size_t;
    auto target_cluster_size() const -> size_t;
    auto max_cluster_variance() const -> float;
    auto cluster_cache_capacity() const -> size_t;
    auto pending_token_limit() const -> size_t;
    auto initial_cluster_capacity(size_t token_count) const -> size_t;
    bool variance_logging_enabled() const;
    auto variance_log_interval() const -> size_t;
    void maybe_log_layer_variance_stats(const char *stage, size_t layer_id, bool force) const;

    auto key_at(size_t layer_id, int token_position) const -> const float *;
    auto value_at(size_t layer_id, int token_position) const -> std::vector<float>;

    auto next_cluster_id(size_t layer_id) -> int;
    auto prefill_cluster_count(size_t token_count) const -> size_t;
    void initialize_prefill_kmeans_centers(
        size_t layer_id,
        size_t cluster_count,
        size_t token_count,
        std::vector<float> &centers
    ) const;
    auto assign_tokens_to_centers(
        size_t layer_id,
        const std::vector<float> &centers,
        size_t token_count
    ) const -> std::vector<size_t>;
    void recompute_centers_from_assignments(
        size_t layer_id,
        const std::vector<size_t> &assignments,
        size_t cluster_count,
        size_t token_count,
        std::vector<float> &centers
    ) const;
    auto make_cluster_from_positions(size_t layer_id, int cluster_id, const std::vector<int> &token_positions) const -> ClusterInfo;
    void build_cluster_storage_from_positions(
        size_t layer_id,
        const std::vector<int> &token_positions,
        std::vector<float> &out_k,
        std::vector<float> &out_v
    ) const;
    void recompute_cluster_stats_from_keys(const std::vector<float> &keys, ClusterInfo &cluster) const;

    auto pick_best_cluster(size_t layer_id, const std::vector<float> &new_k) const -> size_t;
    void update_cluster_center_append(ClusterInfo &cluster, const std::vector<float> &new_k);
    bool maybe_split_cluster(size_t layer_id, size_t cluster_index, int inserted_position);
    auto choose_split_seeds_from_resident(const ClusterInfo &cluster) const -> SplitSeeds;

    bool ensure_cluster_resident(size_t layer_id, size_t cluster_index, const std::vector<uint64_t> &protected_keys);
    void merge_pending_tokens_into_cluster(size_t layer_id, ClusterInfo &cluster);
    bool flush_pending_buckets_if_needed();
    bool flush_pending_bucket(size_t layer_id, ClusterInfo &cluster);
    bool write_back_cluster(size_t layer_id, ClusterInfo &cluster);

    auto make_cache_key(size_t layer_id, int cluster_id) const -> uint64_t;
    void touch_lru(uint64_t key);
    bool evict_one_cluster(const std::vector<uint64_t> &protected_keys);

private:
    GGMLKV &m_kv;
    GGMLKVPager *m_pager = nullptr;
    std::vector<std::vector<ClusterInfo>> m_layer_clusters;
    std::vector<int> m_next_cluster_ids;
    std::vector<size_t> m_decode_update_counts;
    std::vector<PendingBucketMap> m_pending_tokens;
    size_t m_pending_token_count = 0;

    std::list<uint64_t> m_lru_order;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> m_lru_lookup;
    size_t m_resident_cluster_count = 0;
};

} // namespace powerserve::ggml
