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

#include "backend/ggml/ggml_cluster_manager.hpp"

#include "backend/ggml/ggml_kv_pager.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/module/ggml_cluster_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace powerserve::ggml {

namespace {

bool env_flag_enabled(const char *name) {
    const char *v = std::getenv(name);
    if (!v) {
        return false;
    }
    return std::strcmp(v, "1") == 0 ||
           std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 ||
           std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
}

size_t env_size_t(const char *name, size_t fallback) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    const long long parsed = std::atoll(v);
    return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
}

float env_float(const char *name, float fallback) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    return std::strtof(v, nullptr);
}

float dot_product(const float *a, const float *b, size_t n) {
    float out = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

float squared_l2(const float *a, const float *b, size_t n) {
    float out = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = a[i] - b[i];
        out += diff * diff;
    }
    return out;
}

bool contains_key(const std::vector<uint64_t> &keys, uint64_t target) {
    return std::find(keys.begin(), keys.end(), target) != keys.end();
}

void record_layer_cluster_stats(size_t layer_id, const std::vector<ClusterInfo> &clusters) {
    if (!cluster_profile_enabled()) {
        return;
    }
    size_t total_tokens = 0;
    size_t max_tokens = 0;
    for (const auto &cluster : clusters) {
        const size_t token_count = cluster.token_positions.size();
        total_tokens += token_count;
        max_tokens = std::max(max_tokens, token_count);
    }
    cluster_profile_record_layer_cluster_stats(layer_id, clusters.size(), total_tokens, max_tokens);
}

} // namespace

GGMLClusterManager::GGMLClusterManager(GGMLKV &kv) :
    m_kv(kv),
    m_layer_clusters(kv.m_n_layers),
    m_next_cluster_ids(kv.m_n_layers, 0),
    m_decode_update_counts(kv.m_n_layers, 0),
    m_pending_tokens(kv.m_n_layers) {}

void GGMLClusterManager::attach_pager(GGMLKVPager *pager) {
    m_pager = pager;
}

bool GGMLClusterManager::enabled() const {
    return env_flag_enabled("POWERSERVE_GGML_CLUSTER");
}

void GGMLClusterManager::clear() {
    for (auto &clusters : m_layer_clusters) {
        clusters.clear();
    }
    for (auto &pending : m_pending_tokens) {
        pending.clear();
    }
    std::fill(m_next_cluster_ids.begin(), m_next_cluster_ids.end(), 0);
    std::fill(m_decode_update_counts.begin(), m_decode_update_counts.end(), 0);
    m_pending_token_count = 0;
    m_resident_cluster_count = 0;
    m_lru_order.clear();
    m_lru_lookup.clear();
}

void GGMLClusterManager::build_layer_after_prefill(size_t layer_id) {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    if (!enabled()) {
        return;
    }

    auto &clusters = m_layer_clusters[layer_id];
    clusters.clear();
    m_pending_tokens[layer_id].clear();
    m_next_cluster_ids[layer_id] = 0;

    const size_t n_tokens = current_token_count();
    if (n_tokens == 0) {
        return;
    }

    const size_t cluster_count = prefill_cluster_count(n_tokens);
    std::vector<float> centers;
    initialize_prefill_kmeans_centers(layer_id, cluster_count, n_tokens, centers);

    std::vector<size_t> assignments;
    constexpr size_t kmeans_iterations = 5;
    for (size_t iter = 0; iter < kmeans_iterations; ++iter) {
        assignments = assign_tokens_to_centers(layer_id, centers, n_tokens);
        recompute_centers_from_assignments(layer_id, assignments, cluster_count, n_tokens, centers);
    }
    assignments = assign_tokens_to_centers(layer_id, centers, n_tokens);

    std::vector<std::vector<int>> cluster_positions(cluster_count);
    for (size_t pos = 0; pos < n_tokens; ++pos) {
        POWERSERVE_ASSERT(assignments[pos] < cluster_positions.size());
        cluster_positions[assignments[pos]].push_back(static_cast<int>(pos));
    }

    clusters.reserve(cluster_count);
    for (const auto &token_positions : cluster_positions) {
        if (token_positions.empty()) {
            continue;
        }
        ClusterInfo cluster = make_cluster_from_positions(layer_id, next_cluster_id(layer_id), token_positions);
        std::vector<float> cluster_k;
        std::vector<float> cluster_v;
        build_cluster_storage_from_positions(layer_id, token_positions, cluster_k, cluster_v);

        cluster.persisted_token_count = cluster.token_positions.size();
        cluster.capacity_tokens = initial_cluster_capacity(cluster.persisted_token_count);
        cluster.dirty_mode = ClusterDirtyMode::Clean;
        cluster.resident = false;

        if (m_pager && m_pager->valid()) {
            cluster.disk_offset = m_pager->allocate_cluster_region(cluster.capacity_tokens);
            POWERSERVE_ASSERT(cluster.disk_offset >= 0, "failed to allocate cluster region");
            POWERSERVE_ASSERT(
                m_pager->write_cluster_full(
                    cluster.disk_offset,
                    cluster.capacity_tokens,
                    cluster.persisted_token_count,
                    cluster_k.data(),
                    cluster_v.data()
                ),
                "failed to persist cluster during prefill build"
            );
        }

        clusters.push_back(std::move(cluster));
    }
    maybe_log_layer_variance_stats("prefill", layer_id, true);
    record_layer_cluster_stats(layer_id, clusters);
}

void GGMLClusterManager::build_all_layers_after_prefill() {
    if (!enabled()) {
        return;
    }
    clear();
    for (size_t layer_id = 0; layer_id < m_layer_clusters.size(); ++layer_id) {
        build_layer_after_prefill(layer_id);
    }
}

void GGMLClusterManager::update_layer_after_decode(
    size_t layer_id,
    int token_position,
    const std::vector<float> &new_k,
    const std::vector<float> &new_v
) {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    POWERSERVE_ASSERT(new_k.size() == m_kv.m_kv_dim);
    POWERSERVE_ASSERT(new_v.size() == m_kv.m_kv_dim);
    if (!enabled()) {
        return;
    }

    auto &clusters = m_layer_clusters[layer_id];
    if (clusters.empty()) {
        ClusterInfo cluster;
        cluster.cluster_id = next_cluster_id(layer_id);
        cluster.token_positions.push_back(token_position);
        cluster.center = new_k;
        cluster.variance = 0.0f;
        cluster.capacity_tokens = initial_cluster_capacity(1);
        cluster.persisted_token_count = 0;
        clusters.push_back(std::move(cluster));
    }

    const size_t cluster_index = pick_best_cluster(layer_id, new_k);
    auto &cluster = clusters[cluster_index];
    cluster.token_positions.push_back(token_position);
    update_cluster_center_append(cluster, new_k);

    if (cluster.resident) {
        const size_t prev_tokens = cluster.token_positions.size() - 1;
        cluster.resident_key.insert(cluster.resident_key.end(), new_k.begin(), new_k.end());
        std::vector<float> new_resident_value(m_kv.m_kv_dim * cluster.token_positions.size(), 0.0f);
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            for (size_t t = 0; t < prev_tokens; ++t) {
                new_resident_value[d * cluster.token_positions.size() + t] =
                    cluster.resident_value[d * prev_tokens + t];
            }
        }
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            new_resident_value[d * cluster.token_positions.size() + prev_tokens] = new_v[d];
        }
        cluster.resident_value = std::move(new_resident_value);
        cluster.dirty_mode = cluster.dirty_mode == ClusterDirtyMode::RewriteRequired ?
                                 ClusterDirtyMode::RewriteRequired :
                                 ClusterDirtyMode::AppendOnly;
        touch_lru(make_cache_key(layer_id, cluster.cluster_id));
        maybe_split_cluster(layer_id, cluster_index, token_position);
    } else {
        auto &bucket = m_pending_tokens[layer_id][cluster.cluster_id];
        bucket.push_back(PendingClusterToken{
            .position = token_position,
            .key = new_k,
            .value = new_v,
        });
        m_pending_token_count += 1;
        flush_pending_buckets_if_needed();
    }
    m_decode_update_counts[layer_id] += 1;
    maybe_log_layer_variance_stats("decode", layer_id, false);
    record_layer_cluster_stats(layer_id, clusters);
}

auto GGMLClusterManager::get_layer_clusters(size_t layer_id) const -> const std::vector<ClusterInfo> & {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    return m_layer_clusters[layer_id];
}

bool GGMLClusterManager::query_cluster_views(
    size_t layer_id,
    const std::vector<int> &cluster_indices,
    std::vector<ClusterView> &views
) {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    views.clear();
    views.reserve(cluster_indices.size());

    std::vector<uint64_t> protected_keys;
    protected_keys.reserve(cluster_indices.size());
    for (int cluster_index : cluster_indices) {
        POWERSERVE_ASSERT(cluster_index >= 0);
        POWERSERVE_ASSERT(static_cast<size_t>(cluster_index) < m_layer_clusters[layer_id].size());
        auto &cluster = m_layer_clusters[layer_id][static_cast<size_t>(cluster_index)];
        protected_keys.push_back(make_cache_key(layer_id, cluster.cluster_id));
    }

    for (int cluster_index : cluster_indices) {
        auto &cluster = m_layer_clusters[layer_id][static_cast<size_t>(cluster_index)];
        const bool cache_hit = cluster.resident;
        const bool profile = cluster_profile_enabled();
        const int64_t fetch_start_ns = profile ? timestamp_ns() : 0;
        if (!ensure_cluster_resident(layer_id, static_cast<size_t>(cluster_index), protected_keys)) {
            if (profile) {
                cluster_profile_record_cluster_fetch(layer_id, cache_hit, timestamp_ns() - fetch_start_ns);
            }
            return false;
        }
        if (profile) {
            cluster_profile_record_cluster_fetch(layer_id, cache_hit, timestamp_ns() - fetch_start_ns);
        }
        views.push_back(ClusterView{
            .k_ptr = cluster.resident_key.data(),
            .v_ptr = cluster.resident_value.data(),
            .token_count = cluster.token_positions.size(),
        });
    }

    return true;
}

auto GGMLClusterManager::current_token_count() const -> size_t {
    return m_kv.kv_cache ? m_kv.kv_cache->position : 0;
}

auto GGMLClusterManager::target_cluster_size() const -> size_t {
    return env_size_t("POWERSERVE_GGML_CLUSTER_SIZE", 64);
}

auto GGMLClusterManager::max_cluster_variance() const -> float {
    return env_float("POWERSERVE_GGML_CLUSTER_MAX_VAR", 0.0f);
}

auto GGMLClusterManager::cluster_cache_capacity() const -> size_t {
    return env_size_t("POWERSERVE_GGML_CLUSTER_CACHE_SIZE", 8);
}

auto GGMLClusterManager::pending_token_limit() const -> size_t {
    return env_size_t("POWERSERVE_GGML_CLUSTER_BUFFER_SIZE", 256);
}

auto GGMLClusterManager::initial_cluster_capacity(size_t token_count) const -> size_t {
    return std::max(target_cluster_size() * 2, token_count);
}

bool GGMLClusterManager::variance_logging_enabled() const {
    return env_flag_enabled("POWERSERVE_GGML_CLUSTER_VAR_LOG");
}

auto GGMLClusterManager::variance_log_interval() const -> size_t {
    return env_size_t("POWERSERVE_GGML_CLUSTER_VAR_LOG_INTERVAL", 64);
}

void GGMLClusterManager::maybe_log_layer_variance_stats(const char *stage, size_t layer_id, bool force) const {
    if (!variance_logging_enabled()) {
        return;
    }
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    if (!force) {
        const size_t interval = variance_log_interval();
        if (interval == 0 || (m_decode_update_counts[layer_id] % interval) != 0) {
            return;
        }
    }

    const auto &clusters = m_layer_clusters[layer_id];
    if (clusters.empty()) {
        return;
    }

    std::vector<float> vars;
    vars.reserve(clusters.size());
    double sum = 0.0;
    for (const auto &cluster : clusters) {
        vars.push_back(cluster.variance);
        sum += cluster.variance;
    }
    std::sort(vars.begin(), vars.end());
    auto percentile = [&vars](double q) -> float {
        if (vars.empty()) {
            return 0.0f;
        }
        const size_t idx = static_cast<size_t>(q * static_cast<double>(vars.size() - 1));
        return vars[idx];
    };

    POWERSERVE_LOG_INFO(
        "[GGML cluster variance][{}] layer={} clusters={} mean={:.6f} p50={:.6f} p90={:.6f} p95={:.6f} max={:.6f}",
        stage,
        layer_id,
        clusters.size(),
        sum / static_cast<double>(clusters.size()),
        percentile(0.50),
        percentile(0.90),
        percentile(0.95),
        vars.back()
    );
}

auto GGMLClusterManager::key_at(size_t layer_id, int token_position) const -> const float * {
    POWERSERVE_ASSERT(token_position >= 0);
    const auto &buffer = m_kv.key_buffer_for_layer(layer_id);
    return buffer.data() + static_cast<size_t>(token_position) * m_kv.m_kv_dim;
}

auto GGMLClusterManager::value_at(size_t layer_id, int token_position) const -> std::vector<float> {
    std::vector<float> out(m_kv.m_kv_dim);
    const auto &buffer = m_kv.value_buffer_for_layer(layer_id);
    for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
        out[d] = buffer[d * m_kv.m_n_ctx + static_cast<size_t>(token_position)];
    }
    return out;
}

auto GGMLClusterManager::next_cluster_id(size_t layer_id) -> int {
    POWERSERVE_ASSERT(layer_id < m_next_cluster_ids.size());
    return m_next_cluster_ids[layer_id]++;
}

auto GGMLClusterManager::prefill_cluster_count(size_t token_count) const -> size_t {
    POWERSERVE_ASSERT(token_count > 0);
    const size_t cluster_size = target_cluster_size();
    const size_t requested = (token_count + cluster_size - 1) / cluster_size;
    return std::min(token_count, std::max<size_t>(1, requested));
}

void GGMLClusterManager::initialize_prefill_kmeans_centers(
    size_t layer_id,
    size_t cluster_count,
    size_t token_count,
    std::vector<float> &centers
) const {
    POWERSERVE_ASSERT(cluster_count > 0);
    POWERSERVE_ASSERT(token_count >= cluster_count);
    centers.assign(cluster_count * m_kv.m_kv_dim, 0.0f);

    const size_t cluster_size = target_cluster_size();
    for (size_t cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
        const size_t begin = cluster_index * cluster_size;
        const size_t end = std::min(begin + cluster_size, token_count);
        POWERSERVE_ASSERT(begin < end);

        float *center = centers.data() + cluster_index * m_kv.m_kv_dim;
        const float inv_count = 1.0f / static_cast<float>(end - begin);
        for (size_t pos = begin; pos < end; ++pos) {
            const float *key = key_at(layer_id, static_cast<int>(pos));
            for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
                center[d] += key[d];
            }
        }
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            center[d] *= inv_count;
        }
    }
}

auto GGMLClusterManager::assign_tokens_to_centers(
    size_t layer_id,
    const std::vector<float> &centers,
    size_t token_count
) const -> std::vector<size_t> {
    POWERSERVE_ASSERT((centers.size() % m_kv.m_kv_dim) == 0);
    const size_t cluster_count = centers.size() / m_kv.m_kv_dim;
    POWERSERVE_ASSERT(cluster_count > 0);

    std::vector<size_t> assignments(token_count, 0);
    for (size_t pos = 0; pos < token_count; ++pos) {
        const float *key = key_at(layer_id, static_cast<int>(pos));
        size_t best_cluster = 0;
        float best_distance = std::numeric_limits<float>::infinity();
        for (size_t cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
            const float *center = centers.data() + cluster_index * m_kv.m_kv_dim;
            const float distance = squared_l2(key, center, m_kv.m_kv_dim);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster_index;
            }
        }
        assignments[pos] = best_cluster;
    }
    return assignments;
}

void GGMLClusterManager::recompute_centers_from_assignments(
    size_t layer_id,
    const std::vector<size_t> &assignments,
    size_t cluster_count,
    size_t token_count,
    std::vector<float> &centers
) const {
    POWERSERVE_ASSERT(assignments.size() == token_count);
    std::vector<float> previous_centers = centers;
    centers.assign(cluster_count * m_kv.m_kv_dim, 0.0f);
    std::vector<size_t> counts(cluster_count, 0);

    for (size_t pos = 0; pos < token_count; ++pos) {
        const size_t cluster_index = assignments[pos];
        POWERSERVE_ASSERT(cluster_index < cluster_count);
        float *center = centers.data() + cluster_index * m_kv.m_kv_dim;
        const float *key = key_at(layer_id, static_cast<int>(pos));
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            center[d] += key[d];
        }
        counts[cluster_index] += 1;
    }

    for (size_t cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
        if (counts[cluster_index] == 0) {
            std::copy_n(
                previous_centers.data() + static_cast<std::ptrdiff_t>(cluster_index * m_kv.m_kv_dim),
                static_cast<std::ptrdiff_t>(m_kv.m_kv_dim),
                centers.data() + static_cast<std::ptrdiff_t>(cluster_index * m_kv.m_kv_dim)
            );
            continue;
        }
        float *center = centers.data() + cluster_index * m_kv.m_kv_dim;
        const float inv_count = 1.0f / static_cast<float>(counts[cluster_index]);
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            center[d] *= inv_count;
        }
    }
}

auto GGMLClusterManager::make_cluster_from_positions(
    size_t layer_id,
    int cluster_id,
    const std::vector<int> &token_positions
) const -> ClusterInfo {
    ClusterInfo cluster;
    cluster.cluster_id = cluster_id;
    cluster.token_positions = token_positions;
    cluster.center.resize(m_kv.m_kv_dim, 0.0f);

    std::vector<float> keys;
    std::vector<float> values;
    build_cluster_storage_from_positions(layer_id, token_positions, keys, values);
    recompute_cluster_stats_from_keys(keys, cluster);
    return cluster;
}

void GGMLClusterManager::build_cluster_storage_from_positions(
    size_t layer_id,
    const std::vector<int> &token_positions,
    std::vector<float> &out_k,
    std::vector<float> &out_v
) const {
    const size_t token_count = token_positions.size();
    out_k.resize(token_count * m_kv.m_kv_dim);
    out_v.assign(token_count * m_kv.m_kv_dim, 0.0f);

    const auto &key_src = m_kv.key_buffer_for_layer(layer_id);
    const auto &value_src = m_kv.value_buffer_for_layer(layer_id);
    for (size_t idx = 0; idx < token_count; ++idx) {
        const size_t pos = static_cast<size_t>(token_positions[idx]);
        std::copy_n(
            key_src.begin() + static_cast<std::ptrdiff_t>(pos * m_kv.m_kv_dim),
            static_cast<std::ptrdiff_t>(m_kv.m_kv_dim),
            out_k.begin() + static_cast<std::ptrdiff_t>(idx * m_kv.m_kv_dim)
        );
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            out_v[d * token_count + idx] = value_src[d * m_kv.m_n_ctx + pos];
        }
    }
}

void GGMLClusterManager::recompute_cluster_stats_from_keys(const std::vector<float> &keys, ClusterInfo &cluster) const {
    std::fill(cluster.center.begin(), cluster.center.end(), 0.0f);
    cluster.variance = 0.0f;
    const size_t token_count = cluster.token_positions.size();
    if (token_count == 0) {
        return;
    }

    for (size_t t = 0; t < token_count; ++t) {
        const float *key = keys.data() + t * m_kv.m_kv_dim;
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            cluster.center[d] += key[d];
        }
    }
    const float inv_count = 1.0f / static_cast<float>(token_count);
    for (float &value : cluster.center) {
        value *= inv_count;
    }

    float variance_sum = 0.0f;
    for (size_t t = 0; t < token_count; ++t) {
        variance_sum += squared_l2(keys.data() + t * m_kv.m_kv_dim, cluster.center.data(), m_kv.m_kv_dim);
    }
    cluster.variance = variance_sum * inv_count;
}

auto GGMLClusterManager::pick_best_cluster(size_t layer_id, const std::vector<float> &new_k) const -> size_t {
    size_t best_index = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    const auto &clusters = m_layer_clusters[layer_id];
    for (size_t i = 0; i < clusters.size(); ++i) {
        const float score = dot_product(clusters[i].center.data(), new_k.data(), m_kv.m_kv_dim);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    return best_index;
}

void GGMLClusterManager::update_cluster_center_append(ClusterInfo &cluster, const std::vector<float> &new_k) {
    if (cluster.center.empty()) {
        cluster.center = new_k;
        return;
    }
    const size_t new_count = cluster.token_positions.size();
    const float old_count = static_cast<float>(new_count - 1);
    const float inv = 1.0f / static_cast<float>(new_count);
    for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
        cluster.center[d] = (cluster.center[d] * old_count + new_k[d]) * inv;
    }
}

bool GGMLClusterManager::maybe_split_cluster(size_t layer_id, size_t cluster_index, int inserted_position) {
    auto &cluster = m_layer_clusters[layer_id][cluster_index];
    const float max_var = max_cluster_variance();
    if (max_var <= 0.0f || !cluster.resident || cluster.token_positions.size() < 2) {
        return false;
    }

    recompute_cluster_stats_from_keys(cluster.resident_key, cluster);
    if (cluster.variance <= max_var) {
        return false;
    }

    const SplitSeeds seeds = choose_split_seeds_from_resident(cluster);
    if (seeds.first < 0 || seeds.second < 0 || seeds.first == seeds.second) {
        return false;
    }

    std::vector<float> center_a(
        cluster.resident_key.begin() + static_cast<std::ptrdiff_t>(seeds.first * m_kv.m_kv_dim),
        cluster.resident_key.begin() + static_cast<std::ptrdiff_t>((seeds.first + 1) * m_kv.m_kv_dim)
    );
    std::vector<float> center_b(
        cluster.resident_key.begin() + static_cast<std::ptrdiff_t>(seeds.second * m_kv.m_kv_dim),
        cluster.resident_key.begin() + static_cast<std::ptrdiff_t>((seeds.second + 1) * m_kv.m_kv_dim)
    );

    std::vector<size_t> assign_a;
    std::vector<size_t> assign_b;
    for (int iter = 0; iter < 4; ++iter) {
        assign_a.clear();
        assign_b.clear();
        for (size_t idx = 0; idx < cluster.token_positions.size(); ++idx) {
            const float *key = cluster.resident_key.data() + idx * m_kv.m_kv_dim;
            const float dist_a = squared_l2(key, center_a.data(), m_kv.m_kv_dim);
            const float dist_b = squared_l2(key, center_b.data(), m_kv.m_kv_dim);
            if (dist_a <= dist_b) {
                assign_a.push_back(idx);
            } else {
                assign_b.push_back(idx);
            }
        }
        if (assign_a.empty() || assign_b.empty()) {
            return false;
        }

        ClusterInfo tmp_a;
        ClusterInfo tmp_b;
        tmp_a.center.resize(m_kv.m_kv_dim, 0.0f);
        tmp_b.center.resize(m_kv.m_kv_dim, 0.0f);
        tmp_a.token_positions.resize(assign_a.size());
        tmp_b.token_positions.resize(assign_b.size());
        std::vector<float> keys_a(assign_a.size() * m_kv.m_kv_dim);
        std::vector<float> keys_b(assign_b.size() * m_kv.m_kv_dim);
        for (size_t i = 0; i < assign_a.size(); ++i) {
            tmp_a.token_positions[i] = cluster.token_positions[assign_a[i]];
            std::copy_n(
                cluster.resident_key.begin() + static_cast<std::ptrdiff_t>(assign_a[i] * m_kv.m_kv_dim),
                static_cast<std::ptrdiff_t>(m_kv.m_kv_dim),
                keys_a.begin() + static_cast<std::ptrdiff_t>(i * m_kv.m_kv_dim)
            );
        }
        for (size_t i = 0; i < assign_b.size(); ++i) {
            tmp_b.token_positions[i] = cluster.token_positions[assign_b[i]];
            std::copy_n(
                cluster.resident_key.begin() + static_cast<std::ptrdiff_t>(assign_b[i] * m_kv.m_kv_dim),
                static_cast<std::ptrdiff_t>(m_kv.m_kv_dim),
                keys_b.begin() + static_cast<std::ptrdiff_t>(i * m_kv.m_kv_dim)
            );
        }
        recompute_cluster_stats_from_keys(keys_a, tmp_a);
        recompute_cluster_stats_from_keys(keys_b, tmp_b);
        center_a = tmp_a.center;
        center_b = tmp_b.center;
    }

    auto build_child = [&](const std::vector<size_t> &assignments, int cluster_id) {
        ClusterInfo child;
        child.cluster_id = cluster_id;
        child.token_positions.reserve(assignments.size());
        child.center.resize(m_kv.m_kv_dim, 0.0f);
        child.persisted_token_count = 0;
        child.capacity_tokens = initial_cluster_capacity(assignments.size());
        child.disk_offset = -1;
        child.resident = true;
        child.dirty_mode = ClusterDirtyMode::RewriteRequired;
        child.resident_key.resize(assignments.size() * m_kv.m_kv_dim);
        child.resident_value.resize(assignments.size() * m_kv.m_kv_dim);
        for (size_t i = 0; i < assignments.size(); ++i) {
            const size_t src_idx = assignments[i];
            child.token_positions.push_back(cluster.token_positions[src_idx]);
            std::copy_n(
                cluster.resident_key.begin() + static_cast<std::ptrdiff_t>(src_idx * m_kv.m_kv_dim),
                static_cast<std::ptrdiff_t>(m_kv.m_kv_dim),
                child.resident_key.begin() + static_cast<std::ptrdiff_t>(i * m_kv.m_kv_dim)
            );
            for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
                child.resident_value[d * assignments.size() + i] =
                    cluster.resident_value[d * cluster.token_positions.size() + src_idx];
            }
        }
        recompute_cluster_stats_from_keys(child.resident_key, child);
        return child;
    };

    bool inserted_in_a = false;
    for (size_t idx : assign_a) {
        if (cluster.token_positions[idx] == inserted_position) {
            inserted_in_a = true;
            break;
        }
    }

    const int old_cluster_id = cluster.cluster_id;
    ClusterInfo old_child = build_child(inserted_in_a ? assign_b : assign_a, old_cluster_id);
    ClusterInfo new_child = build_child(inserted_in_a ? assign_a : assign_b, next_cluster_id(layer_id));

    const uint64_t old_key = make_cache_key(layer_id, old_cluster_id);
    if (auto it = m_lru_lookup.find(old_key); it != m_lru_lookup.end()) {
        m_lru_order.erase(it->second);
        m_lru_lookup.erase(it);
        if (m_resident_cluster_count > 0) {
            m_resident_cluster_count -= 1;
        }
    }

    cluster = std::move(old_child);
    m_layer_clusters[layer_id].push_back(std::move(new_child));
    const int new_cluster_id = m_layer_clusters[layer_id].back().cluster_id;
    touch_lru(make_cache_key(layer_id, old_cluster_id));
    touch_lru(make_cache_key(layer_id, new_cluster_id));

    const std::vector<uint64_t> protected_keys = {
        make_cache_key(layer_id, old_cluster_id),
        make_cache_key(layer_id, new_cluster_id),
    };
    while (m_resident_cluster_count > cluster_cache_capacity()) {
        if (!evict_one_cluster(protected_keys)) {
            break;
        }
    }
    return true;
}

auto GGMLClusterManager::choose_split_seeds_from_resident(const ClusterInfo &cluster) const -> SplitSeeds {
    SplitSeeds out;
    if (cluster.token_positions.size() < 2) {
        return out;
    }
    out.first = 0;
    float farthest_distance = -1.0f;
    for (size_t idx = 0; idx < cluster.token_positions.size(); ++idx) {
        const float dist = squared_l2(
            cluster.resident_key.data() + idx * m_kv.m_kv_dim,
            cluster.center.data(),
            m_kv.m_kv_dim
        );
        if (dist > farthest_distance) {
            farthest_distance = dist;
            out.second = static_cast<int>(idx);
        }
    }
    if (out.first == out.second) {
        out.second = static_cast<int>(cluster.token_positions.size() - 1);
    }
    return out;
}

bool GGMLClusterManager::ensure_cluster_resident(
    size_t layer_id,
    size_t cluster_index,
    const std::vector<uint64_t> &protected_keys
) {
    auto &cluster = m_layer_clusters[layer_id][cluster_index];
    if (cluster.resident) {
        touch_lru(make_cache_key(layer_id, cluster.cluster_id));
        return true;
    }

    while (m_resident_cluster_count >= cluster_cache_capacity()) {
        if (!evict_one_cluster(protected_keys)) {
            break;
        }
    }

    cluster.resident = true;
    cluster.resident_key.clear();
    cluster.resident_value.clear();
    if (cluster.persisted_token_count > 0) {
        POWERSERVE_ASSERT(m_pager != nullptr && m_pager->valid(), "cluster pager is required for resident load");
        cluster.resident_key.assign(cluster.persisted_token_count * m_kv.m_kv_dim, 0.0f);
        cluster.resident_value.assign(cluster.persisted_token_count * m_kv.m_kv_dim, 0.0f);
        if (!m_pager->read_cluster(
                cluster.disk_offset,
                cluster.capacity_tokens,
                cluster.persisted_token_count,
                cluster.resident_key.data(),
                cluster.resident_value.data()
            )) {
            return false;
        }
    }

    merge_pending_tokens_into_cluster(layer_id, cluster);
    touch_lru(make_cache_key(layer_id, cluster.cluster_id));
    return true;
}

void GGMLClusterManager::merge_pending_tokens_into_cluster(size_t layer_id, ClusterInfo &cluster) {
    auto &bucket = m_pending_tokens[layer_id][cluster.cluster_id];
    if (bucket.empty()) {
        return;
    }
    const size_t existing_tokens = cluster.token_positions.size() - bucket.size();
    cluster.resident_key.resize(cluster.token_positions.size() * m_kv.m_kv_dim);
    std::vector<float> new_resident_value(cluster.token_positions.size() * m_kv.m_kv_dim, 0.0f);
    for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
        for (size_t t = 0; t < existing_tokens; ++t) {
            new_resident_value[d * cluster.token_positions.size() + t] =
                cluster.resident_value[d * existing_tokens + t];
        }
    }
    for (size_t i = 0; i < bucket.size(); ++i) {
        const auto &pending = bucket[i];
        std::copy(pending.key.begin(), pending.key.end(), cluster.resident_key.begin() +
                                                       static_cast<std::ptrdiff_t>((existing_tokens + i) * m_kv.m_kv_dim));
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            new_resident_value[d * cluster.token_positions.size() + existing_tokens + i] = pending.value[d];
        }
    }
    cluster.resident_value = std::move(new_resident_value);
    cluster.dirty_mode = cluster.dirty_mode == ClusterDirtyMode::RewriteRequired ?
                             ClusterDirtyMode::RewriteRequired :
                             ClusterDirtyMode::AppendOnly;
    m_pending_token_count -= bucket.size();
    bucket.clear();
}

bool GGMLClusterManager::flush_pending_buckets_if_needed() {
    if (m_pending_token_count <= pending_token_limit()) {
        return true;
    }
    for (size_t layer_id = 0; layer_id < m_layer_clusters.size(); ++layer_id) {
        for (auto &cluster : m_layer_clusters[layer_id]) {
            auto it = m_pending_tokens[layer_id].find(cluster.cluster_id);
            if (it == m_pending_tokens[layer_id].end() || it->second.empty()) {
                continue;
            }
            if (!flush_pending_bucket(layer_id, cluster)) {
                return false;
            }
        }
    }
    return true;
}

bool GGMLClusterManager::flush_pending_bucket(size_t layer_id, ClusterInfo &cluster) {
    auto it = m_pending_tokens[layer_id].find(cluster.cluster_id);
    if (it == m_pending_tokens[layer_id].end() || it->second.empty()) {
        return true;
    }
    POWERSERVE_ASSERT(m_pager != nullptr && m_pager->valid(), "cluster pager is required for pending flush");

    auto &bucket = it->second;
    const size_t total_tokens = cluster.token_positions.size();
    const size_t append_count = bucket.size();

    if (cluster.capacity_tokens < total_tokens) {
        const size_t new_capacity = initial_cluster_capacity(total_tokens);
        std::vector<float> full_k(total_tokens * m_kv.m_kv_dim, 0.0f);
        std::vector<float> full_v(total_tokens * m_kv.m_kv_dim, 0.0f);
        if (cluster.persisted_token_count > 0) {
            std::vector<float> persisted_k(cluster.persisted_token_count * m_kv.m_kv_dim, 0.0f);
            std::vector<float> persisted_v(cluster.persisted_token_count * m_kv.m_kv_dim, 0.0f);
            POWERSERVE_ASSERT(
                m_pager->read_cluster(
                    cluster.disk_offset,
                    cluster.capacity_tokens,
                    cluster.persisted_token_count,
                    persisted_k.data(),
                    persisted_v.data()
                ),
                "failed to read cluster for relocation"
            );
            std::copy(persisted_k.begin(), persisted_k.end(), full_k.begin());
            for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
                for (size_t t = 0; t < cluster.persisted_token_count; ++t) {
                    full_v[d * total_tokens + t] = persisted_v[d * cluster.persisted_token_count + t];
                }
            }
        }
        for (size_t i = 0; i < append_count; ++i) {
            std::copy(
                bucket[i].key.begin(),
                bucket[i].key.end(),
                full_k.begin() + static_cast<std::ptrdiff_t>((cluster.persisted_token_count + i) * m_kv.m_kv_dim)
            );
            for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
                full_v[d * total_tokens + cluster.persisted_token_count + i] = bucket[i].value[d];
            }
        }
        const int64_t new_offset = m_pager->allocate_cluster_region(new_capacity);
        POWERSERVE_ASSERT(new_offset >= 0, "failed to allocate relocated cluster region");
        POWERSERVE_ASSERT(
            m_pager->write_cluster_full(new_offset, new_capacity, total_tokens, full_k.data(), full_v.data()),
            "failed to rewrite relocated cluster"
        );
        cluster.disk_offset = new_offset;
        cluster.capacity_tokens = new_capacity;
        cluster.persisted_token_count = total_tokens;
        cluster.dirty_mode = ClusterDirtyMode::Clean;
        bucket.clear();
        m_pending_token_count -= append_count;
        return true;
    }

    std::vector<float> append_k(append_count * m_kv.m_kv_dim, 0.0f);
    std::vector<float> append_v(append_count * m_kv.m_kv_dim, 0.0f);
    for (size_t i = 0; i < append_count; ++i) {
        std::copy(
            bucket[i].key.begin(),
            bucket[i].key.end(),
            append_k.begin() + static_cast<std::ptrdiff_t>(i * m_kv.m_kv_dim)
        );
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            append_v[d * append_count + i] = bucket[i].value[d];
        }
    }
    if (cluster.disk_offset < 0) {
        cluster.capacity_tokens = initial_cluster_capacity(total_tokens);
        cluster.disk_offset = m_pager->allocate_cluster_region(cluster.capacity_tokens);
        POWERSERVE_ASSERT(cluster.disk_offset >= 0, "failed to allocate cluster region for append flush");
    }
    POWERSERVE_ASSERT(
        m_pager->append_cluster_tokens(
            cluster.disk_offset,
            cluster.capacity_tokens,
            cluster.persisted_token_count,
            append_count,
            append_k.data(),
            append_v.data()
        ),
        "failed to append cluster tokens"
    );
    cluster.persisted_token_count += append_count;
    cluster.dirty_mode = ClusterDirtyMode::Clean;
    bucket.clear();
    m_pending_token_count -= append_count;
    return true;
}

bool GGMLClusterManager::write_back_cluster(size_t layer_id, ClusterInfo &cluster) {
    POWERSERVE_UNUSED(layer_id);
    if (!cluster.resident || cluster.dirty_mode == ClusterDirtyMode::Clean) {
        return true;
    }
    POWERSERVE_ASSERT(m_pager != nullptr && m_pager->valid(), "cluster pager is required for writeback");

    const size_t token_count = cluster.token_positions.size();
    if (cluster.disk_offset < 0 || cluster.capacity_tokens < token_count || cluster.dirty_mode == ClusterDirtyMode::RewriteRequired) {
        cluster.capacity_tokens = std::max(cluster.capacity_tokens, initial_cluster_capacity(token_count));
        cluster.disk_offset = m_pager->allocate_cluster_region(cluster.capacity_tokens);
        POWERSERVE_ASSERT(cluster.disk_offset >= 0, "failed to allocate cluster region for full writeback");
        if (!m_pager->write_cluster_full(
                cluster.disk_offset,
                cluster.capacity_tokens,
                token_count,
                cluster.resident_key.data(),
                cluster.resident_value.data()
            )) {
            return false;
        }
        cluster.persisted_token_count = token_count;
        cluster.dirty_mode = ClusterDirtyMode::Clean;
        return true;
    }

    const size_t append_count = token_count - cluster.persisted_token_count;
    if (append_count > 0) {
        const float *append_k =
            cluster.resident_key.data() + static_cast<std::ptrdiff_t>(cluster.persisted_token_count * m_kv.m_kv_dim);
        std::vector<float> append_v(append_count * m_kv.m_kv_dim, 0.0f);
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            for (size_t i = 0; i < append_count; ++i) {
                append_v[d * append_count + i] =
                    cluster.resident_value[d * token_count + cluster.persisted_token_count + i];
            }
        }
        if (!m_pager->append_cluster_tokens(
                cluster.disk_offset,
                cluster.capacity_tokens,
                cluster.persisted_token_count,
                append_count,
                append_k,
                append_v.data()
            )) {
            return false;
        }
    }
    cluster.persisted_token_count = token_count;
    cluster.dirty_mode = ClusterDirtyMode::Clean;
    return true;
}

auto GGMLClusterManager::make_cache_key(size_t layer_id, int cluster_id) const -> uint64_t {
    return (static_cast<uint64_t>(layer_id) << 32) | static_cast<uint32_t>(cluster_id);
}

void GGMLClusterManager::touch_lru(uint64_t key) {
    auto it = m_lru_lookup.find(key);
    if (it != m_lru_lookup.end()) {
        m_lru_order.erase(it->second);
    } else {
        m_resident_cluster_count += 1;
    }
    m_lru_order.push_front(key);
    m_lru_lookup[key] = m_lru_order.begin();
}

bool GGMLClusterManager::evict_one_cluster(const std::vector<uint64_t> &protected_keys) {
    for (auto it = m_lru_order.rbegin(); it != m_lru_order.rend(); ++it) {
        const uint64_t key = *it;
        if (contains_key(protected_keys, key)) {
            continue;
        }
        const size_t layer_id = static_cast<size_t>(key >> 32);
        const int cluster_id = static_cast<int>(key & 0xffffffffu);
        auto &clusters = m_layer_clusters[layer_id];
        auto cluster_it = std::find_if(clusters.begin(), clusters.end(), [cluster_id](const ClusterInfo &cluster) {
            return cluster.cluster_id == cluster_id;
        });
        if (cluster_it == clusters.end() || !cluster_it->resident) {
            continue;
        }
        if (!write_back_cluster(layer_id, *cluster_it)) {
            return false;
        }
        cluster_it->resident = false;
        cluster_it->resident_key.clear();
        cluster_it->resident_key.shrink_to_fit();
        cluster_it->resident_value.clear();
        cluster_it->resident_value.shrink_to_fit();

        auto lookup_it = m_lru_lookup.find(key);
        if (lookup_it != m_lru_lookup.end()) {
            m_lru_order.erase(lookup_it->second);
            m_lru_lookup.erase(lookup_it);
        }
        if (m_resident_cluster_count > 0) {
            m_resident_cluster_count -= 1;
        }
        return true;
    }
    return false;
}

} // namespace powerserve::ggml
