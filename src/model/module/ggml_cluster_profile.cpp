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

#include "model/module/ggml_cluster_profile.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

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

int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(v);
    return parsed > 0 ? parsed : fallback;
}

double ns_to_ms(int64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

double avg_ms(int64_t ns, size_t count) {
    return count == 0 ? 0.0 : ns_to_ms(ns) / static_cast<double>(count);
}

struct LayerProfile {
    int64_t layer_total_ns = 0;
    size_t layer_total_count = 0;

    int64_t ffn_ns = 0;
    size_t ffn_count = 0;

    int64_t cluster_update_copy_ns = 0;
    int64_t cluster_update_apply_ns = 0;
    size_t cluster_update_count = 0;

    int64_t select_ns = 0;
    size_t select_count = 0;

    int64_t fetch_hit_ns = 0;
    size_t fetch_hit_count = 0;
    int64_t fetch_miss_ns = 0;
    size_t fetch_miss_count = 0;

    int64_t attn_compute_ns = 0;
    size_t attn_compute_count = 0;

    size_t selected_cluster_sum = 0;
    size_t selected_token_sum = 0;
    size_t selected_count = 0;

    size_t cluster_count = 0;
    size_t cluster_token_total = 0;
    size_t cluster_token_max = 0;
};

struct ClusterProfile {
    bool initialized = false;
    size_t n_layers = 0;
    size_t report_count = 0;

    size_t prefill_tokens = 0;
    int64_t prefill_ns = 0;
    int64_t global_cluster_build_ns = 0;

    std::vector<int64_t> decode_token_ns;
    size_t reported_decode_tokens = 0;

    std::vector<LayerProfile> layers;
};

std::mutex g_mutex;
ClusterProfile g_profile;
bool g_report_file_initialized = false;

constexpr const char *k_cluster_profile_report_path = "/data/local/tmp/ziqian/powerserve/report.txt";
constexpr const char *k_dense_profile_report_path = "/data/local/tmp/ziqian/powerserve/dense_report.txt";

LayerProfile *layer_locked(size_t layer_id) {
    if (!g_profile.initialized || layer_id >= g_profile.layers.size()) {
        return nullptr;
    }
    return &g_profile.layers[layer_id];
}

double percentile_ms(std::vector<int64_t> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t idx = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
    return ns_to_ms(values[idx]);
}

bool write_report_lines_locked(const char *path, const std::vector<std::string> &lines, const char *mode) {
    FILE *fp = std::fopen(path, mode);
    if (!fp) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            POWERSERVE_LOG_WARN(
                "[GGML cluster profile] failed to open {} for writing: {}",
                path,
                get_system_error()
            );
        }
        return false;
    }
    for (const auto &line : lines) {
        std::fputs(line.c_str(), fp);
        std::fputc('\n', fp);
    }
    std::fclose(fp);
    return true;
}

struct DenseLayerProfile {
    int64_t layer_total_ns = 0;
    size_t layer_total_count = 0;

    int64_t attn_ns = 0;
    size_t attn_count = 0;

    int64_t ffn_ns = 0;
    size_t ffn_count = 0;
};

struct DenseProfile {
    bool initialized = false;
    size_t n_layers = 0;
    size_t report_count = 0;

    size_t prefill_tokens = 0;
    int64_t prefill_ns = 0;

    std::vector<int64_t> decode_token_ns;
    size_t reported_decode_tokens = 0;

    std::vector<DenseLayerProfile> layers;
};

DenseProfile g_dense_profile;
bool g_dense_report_file_initialized = false;

DenseLayerProfile *dense_layer_locked(size_t layer_id) {
    if (!g_dense_profile.initialized || layer_id >= g_dense_profile.layers.size()) {
        return nullptr;
    }
    return &g_dense_profile.layers[layer_id];
}

} // namespace

bool cluster_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_flag_enabled("POWERSERVE_GGML_CLUSTER_PROFILE") ? 1 : 0;
    }
    return cached == 1;
}

int cluster_profile_report_interval() {
    static int cached = env_int("POWERSERVE_GGML_CLUSTER_PROFILE_INTERVAL", 16);
    return cached;
}

bool dense_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_flag_enabled("POWERSERVE_GGML_DENSE_PROFILE") ? 1 : 0;
    }
    return cached == 1;
}

int dense_profile_report_interval() {
    static int cached = env_int("POWERSERVE_GGML_DENSE_PROFILE_INTERVAL", 16);
    return cached;
}

void cluster_profile_reset(size_t n_layers) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_profile = ClusterProfile{};
    g_profile.initialized = true;
    g_profile.n_layers = n_layers;
    g_profile.layers.resize(n_layers);

    const char *mode = g_report_file_initialized ? "a" : "w";
    if (write_report_lines_locked(k_cluster_profile_report_path, {
            "[GGML cluster profile]",
            fmt::format("report_file={}", k_cluster_profile_report_path),
            fmt::format("n_layers={}", n_layers),
            ""
        }, mode)) {
        g_report_file_initialized = true;
        POWERSERVE_LOG_INFO("[GGML cluster profile] writing report to {}", k_cluster_profile_report_path);
    }
}

void dense_profile_reset(size_t n_layers) {
    if (!dense_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_dense_profile = DenseProfile{};
    g_dense_profile.initialized = true;
    g_dense_profile.n_layers = n_layers;
    g_dense_profile.layers.resize(n_layers);

    const char *mode = g_dense_report_file_initialized ? "a" : "w";
    if (write_report_lines_locked(k_dense_profile_report_path, {
            "[GGML dense profile]",
            fmt::format("report_file={}", k_dense_profile_report_path),
            fmt::format("n_layers={}", n_layers),
            ""
        }, mode)) {
        g_dense_report_file_initialized = true;
        POWERSERVE_LOG_INFO("[GGML dense profile] writing report to {}", k_dense_profile_report_path);
    }
}

void cluster_profile_record_prefill(size_t tokens, int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_profile.prefill_tokens += tokens;
    g_profile.prefill_ns += ns;
}

void dense_profile_record_prefill(size_t tokens, int64_t ns) {
    if (!dense_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_dense_profile.prefill_tokens += tokens;
    g_dense_profile.prefill_ns += ns;
}

void cluster_profile_record_global_cluster_build(int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_profile.global_cluster_build_ns += ns;
}

void cluster_profile_record_decode_token(int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_profile.decode_token_ns.push_back(ns);
}

void dense_profile_record_decode_token(int64_t ns) {
    if (!dense_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_dense_profile.decode_token_ns.push_back(ns);
}

void cluster_profile_record_layer_op(size_t layer_id, const char *name, int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    if (std::strcmp(name, "layer") == 0) {
        layer->layer_total_ns += ns;
        layer->layer_total_count += 1;
    } else if (std::strcmp(name, "ffn") == 0) {
        layer->ffn_ns += ns;
        layer->ffn_count += 1;
    }
}

void dense_profile_record_layer_op(size_t layer_id, const char *name, int64_t ns) {
    if (!dense_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = dense_layer_locked(layer_id);
    if (!layer) {
        return;
    }
    if (std::strcmp(name, "layer") == 0) {
        layer->layer_total_ns += ns;
        layer->layer_total_count += 1;
    } else if (std::strcmp(name, "attn") == 0) {
        layer->attn_ns += ns;
        layer->attn_count += 1;
    } else if (std::strcmp(name, "ffn") == 0) {
        layer->ffn_ns += ns;
        layer->ffn_count += 1;
    }
}

void cluster_profile_record_cluster_update(size_t layer_id, int64_t copy_ns, int64_t update_ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    layer->cluster_update_copy_ns += copy_ns;
    layer->cluster_update_apply_ns += update_ns;
    layer->cluster_update_count += 1;
}

void cluster_profile_record_cluster_select(size_t layer_id, int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    layer->select_ns += ns;
    layer->select_count += 1;
}

void cluster_profile_record_cluster_fetch(size_t layer_id, bool cache_hit, int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    if (cache_hit) {
        layer->fetch_hit_ns += ns;
        layer->fetch_hit_count += 1;
    } else {
        layer->fetch_miss_ns += ns;
        layer->fetch_miss_count += 1;
    }
}

void cluster_profile_record_cluster_attn_compute(size_t layer_id, int64_t ns) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    layer->attn_compute_ns += ns;
    layer->attn_compute_count += 1;
}

void cluster_profile_record_cluster_selection(size_t layer_id, size_t selected_clusters, size_t selected_tokens) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    layer->selected_cluster_sum += selected_clusters;
    layer->selected_token_sum += selected_tokens;
    layer->selected_count += 1;
}

void cluster_profile_record_layer_cluster_stats(size_t layer_id, size_t cluster_count, size_t total_tokens, size_t max_tokens) {
    if (!cluster_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto *layer = layer_locked(layer_id);
    if (!layer) {
        return;
    }
    layer->cluster_count = cluster_count;
    layer->cluster_token_total = total_tokens;
    layer->cluster_token_max = max_tokens;
}

void cluster_profile_maybe_report(const std::string &model_id, bool force) {
    if (!cluster_profile_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const size_t decode_count = g_profile.decode_token_ns.size();
    const int interval = cluster_profile_report_interval();
    if (!force && (decode_count == g_profile.reported_decode_tokens || (decode_count % static_cast<size_t>(interval)) != 0)) {
        return;
    }
    g_profile.reported_decode_tokens = decode_count;

    const double prefill_tps = g_profile.prefill_ns > 0 ?
        static_cast<double>(g_profile.prefill_tokens) * 1000000000.0 / static_cast<double>(g_profile.prefill_ns) :
        0.0;
    const int64_t decode_total_ns = std::accumulate(g_profile.decode_token_ns.begin(), g_profile.decode_token_ns.end(), int64_t{0});
    const int64_t measured_total_ns = g_profile.prefill_ns + g_profile.global_cluster_build_ns + decode_total_ns;
    const size_t measured_tokens = g_profile.prefill_tokens + decode_count;
    const double measured_tps = measured_total_ns > 0 ?
        static_cast<double>(measured_tokens) * 1000000000.0 / static_cast<double>(measured_total_ns) :
        0.0;
    const double decode_tps = decode_total_ns > 0 ?
        static_cast<double>(decode_count) * 1000000000.0 / static_cast<double>(decode_total_ns) :
        0.0;

    std::vector<std::string> report_lines;
    g_profile.report_count += 1;
    report_lines.push_back(fmt::format("[GGML cluster profile report #{}]", g_profile.report_count));
    report_lines.push_back(fmt::format(
        "[GGML cluster profile] model={} measured_total_ms={:.3f} measured_tps={:.3f} prefill_tokens={} prefill_ms={:.3f} prefill_tps={:.3f} cluster_build_ms={:.3f} decode_tokens={} decode_total_ms={:.3f} decode_avg_ms={:.3f} p50_ms={:.3f} p90_ms={:.3f} max_ms={:.3f} decode_tps={:.3f}",
        model_id,
        ns_to_ms(measured_total_ns),
        measured_tps,
        g_profile.prefill_tokens,
        ns_to_ms(g_profile.prefill_ns),
        prefill_tps,
        ns_to_ms(g_profile.global_cluster_build_ns),
        decode_count,
        ns_to_ms(decode_total_ns),
        avg_ms(decode_total_ns, decode_count),
        percentile_ms(g_profile.decode_token_ns, 0.50),
        percentile_ms(g_profile.decode_token_ns, 0.90),
        g_profile.decode_token_ns.empty() ? 0.0 : ns_to_ms(*std::max_element(g_profile.decode_token_ns.begin(), g_profile.decode_token_ns.end())),
        decode_tps
    ));

    for (size_t layer_id = 0; layer_id < g_profile.layers.size(); ++layer_id) {
        const auto &l = g_profile.layers[layer_id];
        if (l.layer_total_count == 0 && l.cluster_update_count == 0 && l.attn_compute_count == 0) {
            continue;
        }
        const size_t layer_invocations = std::max<size_t>(1, std::max(l.cluster_update_count, l.attn_compute_count));
        const size_t fetch_total = l.fetch_hit_count + l.fetch_miss_count;
        const double hit_rate = fetch_total > 0 ? 100.0 * static_cast<double>(l.fetch_hit_count) / static_cast<double>(fetch_total) : 0.0;
        const double avg_cluster_size = l.cluster_count > 0 ?
            static_cast<double>(l.cluster_token_total) / static_cast<double>(l.cluster_count) :
            0.0;
        report_lines.push_back(fmt::format(
            "[GGML cluster profile][L{}] layer_ms={:.3f} ffn_ms={:.3f} update_copy_ms={:.3f} update_apply_ms={:.3f} select_ms={:.3f} attn_compute_ms={:.3f} fetch_hit_ms={:.3f} fetch_miss_ms={:.3f} hit_rate={:.1f}% selected_clusters={:.2f} selected_tokens={:.2f} clusters={} avg_cluster_size={:.2f} max_cluster_size={}",
            layer_id,
            avg_ms(l.layer_total_ns, layer_invocations),
            avg_ms(l.ffn_ns, layer_invocations),
            avg_ms(l.cluster_update_copy_ns, l.cluster_update_count),
            avg_ms(l.cluster_update_apply_ns, l.cluster_update_count),
            avg_ms(l.select_ns, l.select_count),
            avg_ms(l.attn_compute_ns, l.attn_compute_count),
            avg_ms(l.fetch_hit_ns, l.fetch_hit_count),
            avg_ms(l.fetch_miss_ns, l.fetch_miss_count),
            hit_rate,
            l.selected_count > 0 ? static_cast<double>(l.selected_cluster_sum) / static_cast<double>(l.selected_count) : 0.0,
            l.selected_count > 0 ? static_cast<double>(l.selected_token_sum) / static_cast<double>(l.selected_count) : 0.0,
            l.cluster_count,
            avg_cluster_size,
            l.cluster_token_max
        ));
    }
    report_lines.push_back("");
    write_report_lines_locked(k_cluster_profile_report_path, report_lines, "a");
}

void dense_profile_maybe_report(const std::string &model_id, bool force) {
    if (!dense_profile_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const size_t decode_count = g_dense_profile.decode_token_ns.size();
    const int interval = dense_profile_report_interval();
    if (!force && (decode_count == g_dense_profile.reported_decode_tokens || (decode_count % static_cast<size_t>(interval)) != 0)) {
        return;
    }
    g_dense_profile.reported_decode_tokens = decode_count;

    const double prefill_tps = g_dense_profile.prefill_ns > 0 ?
        static_cast<double>(g_dense_profile.prefill_tokens) * 1000000000.0 / static_cast<double>(g_dense_profile.prefill_ns) :
        0.0;
    const int64_t decode_total_ns = std::accumulate(g_dense_profile.decode_token_ns.begin(), g_dense_profile.decode_token_ns.end(), int64_t{0});
    const int64_t measured_total_ns = g_dense_profile.prefill_ns + decode_total_ns;
    const size_t measured_tokens = g_dense_profile.prefill_tokens + decode_count;
    const double measured_tps = measured_total_ns > 0 ?
        static_cast<double>(measured_tokens) * 1000000000.0 / static_cast<double>(measured_total_ns) :
        0.0;
    const double decode_tps = decode_total_ns > 0 ?
        static_cast<double>(decode_count) * 1000000000.0 / static_cast<double>(decode_total_ns) :
        0.0;

    std::vector<std::string> report_lines;
    g_dense_profile.report_count += 1;
    report_lines.push_back(fmt::format("[GGML dense profile report #{}]", g_dense_profile.report_count));
    report_lines.push_back(fmt::format(
        "[GGML dense profile] model={} measured_total_ms={:.3f} measured_tps={:.3f} prefill_tokens={} prefill_ms={:.3f} prefill_tps={:.3f} decode_tokens={} decode_total_ms={:.3f} decode_avg_ms={:.3f} p50_ms={:.3f} p90_ms={:.3f} max_ms={:.3f} decode_tps={:.3f}",
        model_id,
        ns_to_ms(measured_total_ns),
        measured_tps,
        g_dense_profile.prefill_tokens,
        ns_to_ms(g_dense_profile.prefill_ns),
        prefill_tps,
        decode_count,
        ns_to_ms(decode_total_ns),
        avg_ms(decode_total_ns, decode_count),
        percentile_ms(g_dense_profile.decode_token_ns, 0.50),
        percentile_ms(g_dense_profile.decode_token_ns, 0.90),
        g_dense_profile.decode_token_ns.empty() ? 0.0 : ns_to_ms(*std::max_element(g_dense_profile.decode_token_ns.begin(), g_dense_profile.decode_token_ns.end())),
        decode_tps
    ));

    for (size_t layer_id = 0; layer_id < g_dense_profile.layers.size(); ++layer_id) {
        const auto &l = g_dense_profile.layers[layer_id];
        if (l.layer_total_count == 0) {
            continue;
        }
        const size_t layer_invocations = std::max<size_t>(1, decode_count);
        report_lines.push_back(fmt::format(
            "[GGML dense profile][L{}] layer_ms={:.3f} attn_ms={:.3f} ffn_ms={:.3f}",
            layer_id,
            avg_ms(l.layer_total_ns, layer_invocations),
            avg_ms(l.attn_ns, layer_invocations),
            avg_ms(l.ffn_ns, layer_invocations)
        ));
    }
    report_lines.push_back("");
    write_report_lines_locked(k_dense_profile_report_path, report_lines, "a");
}

} // namespace powerserve::ggml
