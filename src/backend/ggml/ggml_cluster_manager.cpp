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

#include "core/logger.hpp"

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

} // namespace

GGMLClusterManager::GGMLClusterManager(GGMLKV &kv) : m_kv(kv), m_layer_clusters(kv.m_n_layers) {}

bool GGMLClusterManager::enabled() const {
    return env_flag_enabled("POWERSERVE_GGML_CLUSTER");
}

void GGMLClusterManager::clear() {
    for (auto &clusters : m_layer_clusters) {
        clusters.clear();
    }
}

void GGMLClusterManager::build_layer_after_prefill(size_t layer_id) {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    if (!enabled()) {
        return;
    }

    auto &clusters = m_layer_clusters[layer_id];
    clusters.clear();

    const size_t n_tokens = current_token_count();
    if (n_tokens == 0) {
        return;
    }

    const size_t cluster_size = target_cluster_size();
    clusters.reserve((n_tokens + cluster_size - 1) / cluster_size);
    for (size_t begin = 0; begin < n_tokens; begin += cluster_size) {
        const size_t end = std::min(begin + cluster_size, n_tokens);
        std::vector<int> token_positions;
        token_positions.reserve(end - begin);
        for (size_t pos = begin; pos < end; ++pos) {
            token_positions.push_back(static_cast<int>(pos));
        }
        clusters.push_back(make_cluster(layer_id, token_positions));
    }
}

void GGMLClusterManager::build_all_layers_after_prefill() {
    if (!enabled()) {
        return;
    }
    for (size_t layer_id = 0; layer_id < m_layer_clusters.size(); ++layer_id) {
        build_layer_after_prefill(layer_id);
    }
}

void GGMLClusterManager::update_layer_after_decode(size_t layer_id, int token_position, const std::vector<float> &new_k) {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    POWERSERVE_ASSERT(new_k.size() == m_kv.m_kv_dim);
    if (!enabled()) {
        return;
    }

    auto &clusters = m_layer_clusters[layer_id];
    if (clusters.empty()) {
        std::vector<int> token_positions = {token_position};
        clusters.push_back(make_cluster(layer_id, token_positions));
        return;
    }

    const size_t cluster_index = pick_best_cluster(layer_id, new_k);
    auto &cluster = clusters[cluster_index];
    cluster.token_positions.push_back(token_position);
    recompute_cluster_stats(layer_id, cluster);

    auto replacement = maybe_split_cluster(layer_id, cluster);
    if (replacement.size() <= 1) {
        return;
    }

    cluster = std::move(replacement.front());
    for (size_t i = 1; i < replacement.size(); ++i) {
        clusters.push_back(std::move(replacement[i]));
    }
}

auto GGMLClusterManager::get_layer_clusters(size_t layer_id) const -> const std::vector<ClusterInfo> & {
    POWERSERVE_ASSERT(layer_id < m_layer_clusters.size());
    return m_layer_clusters[layer_id];
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

auto GGMLClusterManager::make_cluster(size_t layer_id, const std::vector<int> &token_positions) const -> ClusterInfo {
    ClusterInfo cluster;
    cluster.token_positions = token_positions;
    cluster.center.resize(m_kv.m_kv_dim, 0.0f);
    recompute_cluster_stats(layer_id, cluster);
    return cluster;
}

auto GGMLClusterManager::key_at(size_t layer_id, int token_position) const -> const float * {
    POWERSERVE_ASSERT(token_position >= 0);
    const auto &buffer = m_kv.key_buffer_for_layer(layer_id);
    return buffer.data() + static_cast<size_t>(token_position) * m_kv.m_kv_dim;
}

void GGMLClusterManager::recompute_cluster_stats(size_t layer_id, ClusterInfo &cluster) const {
    std::fill(cluster.center.begin(), cluster.center.end(), 0.0f);
    cluster.variance = 0.0f;
    if (cluster.token_positions.empty()) {
        return;
    }

    for (int token_position : cluster.token_positions) {
        const float *key = key_at(layer_id, token_position);
        for (size_t d = 0; d < m_kv.m_kv_dim; ++d) {
            cluster.center[d] += key[d];
        }
    }
    const float inv_count = 1.0f / static_cast<float>(cluster.token_positions.size());
    for (float &value : cluster.center) {
        value *= inv_count;
    }

    float variance_sum = 0.0f;
    for (int token_position : cluster.token_positions) {
        variance_sum += squared_l2(key_at(layer_id, token_position), cluster.center.data(), m_kv.m_kv_dim);
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

auto GGMLClusterManager::maybe_split_cluster(size_t layer_id, ClusterInfo &cluster) const -> std::vector<ClusterInfo> {
    const float max_var = max_cluster_variance();
    if (max_var <= 0.0f || cluster.variance <= max_var || cluster.token_positions.size() < 2) {
        return {cluster};
    }

    const SplitSeeds seeds = choose_split_seeds(layer_id, cluster);
    if (seeds.first < 0 || seeds.second < 0 || seeds.first == seeds.second) {
        return {cluster};
    }

    std::vector<float> center_a(key_at(layer_id, seeds.first), key_at(layer_id, seeds.first) + m_kv.m_kv_dim);
    std::vector<float> center_b(key_at(layer_id, seeds.second), key_at(layer_id, seeds.second) + m_kv.m_kv_dim);
    std::vector<int> assign_a;
    std::vector<int> assign_b;

    for (int iter = 0; iter < 4; ++iter) {
        assign_a.clear();
        assign_b.clear();
        for (int token_position : cluster.token_positions) {
            const float *key = key_at(layer_id, token_position);
            const float dist_a = squared_l2(key, center_a.data(), m_kv.m_kv_dim);
            const float dist_b = squared_l2(key, center_b.data(), m_kv.m_kv_dim);
            if (dist_a <= dist_b) {
                assign_a.push_back(token_position);
            } else {
                assign_b.push_back(token_position);
            }
        }

        if (assign_a.empty() || assign_b.empty()) {
            return {cluster};
        }

        ClusterInfo cluster_a = make_cluster(layer_id, assign_a);
        ClusterInfo cluster_b = make_cluster(layer_id, assign_b);
        center_a = cluster_a.center;
        center_b = cluster_b.center;
    }

    return {make_cluster(layer_id, assign_a), make_cluster(layer_id, assign_b)};
}

auto GGMLClusterManager::choose_split_seeds(size_t layer_id, const ClusterInfo &cluster) const -> SplitSeeds {
    SplitSeeds out;
    if (cluster.token_positions.size() < 2) {
        return out;
    }

    out.first = cluster.token_positions.front();
    float farthest_distance = -1.0f;
    for (int token_position : cluster.token_positions) {
        const float dist = squared_l2(key_at(layer_id, token_position), cluster.center.data(), m_kv.m_kv_dim);
        if (dist > farthest_distance) {
            farthest_distance = dist;
            out.second = token_position;
        }
    }

    if (out.first == out.second) {
        out.second = cluster.token_positions.back();
    }
    return out;
}

} // namespace powerserve::ggml
