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
#include "backend/ggml/ggml_kv_pager.hpp"

#include <memory>
#include <string>
#include <vector>

namespace powerserve::ggml_runtime {

struct KVRuntimeState {
    bool pager_active = false;
    bool pager_do_sync = false;
    std::vector<bool> layer_computed_this_step;
};

int get_ggml_segment_layers();
bool is_kv_pager_enabled();
bool is_kv_pager_sync_enabled();
std::string get_kv_pager_file_path(const std::string &weights_path, const std::string &model_id);

KVRuntimeState prepare_kv_runtime(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    const std::string &weights_path,
    const std::string &model_id,
    const std::vector<int> &pos,
    size_t n_layers
);

void prepare_kv_segment(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t n_layers,
    size_t tokens_before_step,
    size_t tokens_after_step
);

void wait_kv_segment_ready(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    const ggml::GGMLKV &ggml_kv,
    const KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t tokens_before_step
);

void finish_kv_segment(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t n_layers,
    size_t tokens_before_step,
    size_t tokens_after_step
);

void sync_kv_runtime_if_needed(std::unique_ptr<ggml::GGMLKVPager> &kv_pager, const KVRuntimeState &state);

} // namespace powerserve::ggml_runtime
