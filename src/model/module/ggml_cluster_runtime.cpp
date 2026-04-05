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

#include "model/module/ggml_cluster_runtime.hpp"

#include <mutex>
#include <unordered_map>

namespace powerserve::ggml {

namespace {

std::mutex g_cluster_runtime_mutex;
std::unordered_map<std::string, GGMLClusterRuntimeView> g_cluster_runtimes;

} // namespace

void register_cluster_runtime(
    const std::string &model_id,
    GGMLClusterManager *manager,
    GGMLKVPager *pager
) {
    std::lock_guard<std::mutex> lock(g_cluster_runtime_mutex);
    g_cluster_runtimes[model_id] = GGMLClusterRuntimeView{
        .manager = manager,
        .pager = pager,
        .ready = false,
    };
}

void set_cluster_runtime_ready(const std::string &model_id, bool ready) {
    std::lock_guard<std::mutex> lock(g_cluster_runtime_mutex);
    auto &runtime = g_cluster_runtimes[model_id];
    runtime.ready = ready;
}

auto get_cluster_runtime(const std::string &model_id) -> GGMLClusterRuntimeView {
    std::lock_guard<std::mutex> lock(g_cluster_runtime_mutex);
    const auto it = g_cluster_runtimes.find(model_id);
    if (it == g_cluster_runtimes.end()) {
        return {};
    }
    return it->second;
}

} // namespace powerserve::ggml
