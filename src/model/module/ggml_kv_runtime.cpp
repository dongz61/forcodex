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

#include "ggml_kv_runtime.hpp"

#include "core/logger.hpp"

#include <cstdlib>
#include <cstring>

namespace powerserve::ggml_runtime {

namespace {

bool prefetch_layer_with_optional_binding(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    const KVRuntimeState &state,
    size_t layer_id,
    size_t n_layers,
    size_t tokens_before_step,
    size_t tokens_after_step,
    bool allow_blocking_evict
) {
    if (ggml_kv.layer_to_slot(layer_id) < 0) {
        int free_slot = ggml_kv.find_free_slot();
        if (free_slot < 0) {
            if (!allow_blocking_evict) {
                return false;
            }

            const size_t keep = ggml_kv.slot_window_size();
            size_t victim_layer = n_layers;

            if (layer_id >= keep && ggml_kv.layer_to_slot(layer_id - keep) >= 0) {
                victim_layer = layer_id - keep;
            } else {
                for (size_t s = 0; s < ggml_kv.m_slot_to_layer.size(); ++s) {
                    const int mapped_layer = ggml_kv.m_slot_to_layer[s];
                    if (mapped_layer >= 0) {
                        victim_layer = static_cast<size_t>(mapped_layer);
                        break;
                    }
                }
            }

            POWERSERVE_ASSERT(victim_layer < n_layers, "no resident victim layer found while no free slot exists");

            const size_t evict_tokens =
                state.layer_computed_this_step[victim_layer] ? tokens_after_step : tokens_before_step;
            if (!kv_pager->evict_layer_async(victim_layer, evict_tokens, false)) {
                POWERSERVE_LOG_ERROR("KV pager evict(victim) failed at layer {}", victim_layer);
                POWERSERVE_ABORT("KV pager evict victim failure");
            }
            if (!kv_pager->wait_layer_evicted(victim_layer)) {
                POWERSERVE_LOG_ERROR("KV pager evict(victim) wait failed at layer {}", victim_layer);
                POWERSERVE_ABORT("KV pager evict victim wait failure");
            }
            const int victim_slot = ggml_kv.layer_to_slot(victim_layer);
            POWERSERVE_ASSERT(victim_slot >= 0);
            ggml_kv.unbind_layer(victim_layer);
            free_slot = victim_slot;
        }
        ggml_kv.bind_layer_to_slot(layer_id, static_cast<size_t>(free_slot));
    }

    if (!kv_pager->prefetch_layer_async(layer_id, tokens_before_step)) {
        POWERSERVE_LOG_ERROR("KV pager prefetch failed at layer {}", layer_id);
        POWERSERVE_ABORT("KV pager prefetch failure");
    }
    return true;
}

} // namespace

int get_ggml_segment_layers() {
    const char *v = std::getenv("POWERSERVE_GGML_SEGMENT_LAYERS");
    if (!v) {
        return 0;
    }
    const int n = std::atoi(v);
    return n > 0 ? n : 0;
}

bool is_kv_pager_enabled() {
    const char *v = std::getenv("POWERSERVE_KV_PAGER");
    if (!v) {
        return false;
    }
    return std::strcmp(v, "1") == 0 ||
           std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 ||
           std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
}

bool is_kv_pager_sync_enabled() {
    const char *v = std::getenv("POWERSERVE_KV_PAGER_SYNC_EACH_STEP");
    if (!v) {
        return false;
    }
    return std::strcmp(v, "1") == 0 ||
           std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 ||
           std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
}

std::string get_kv_pager_file_path(const std::string &weights_path, const std::string &model_id) {
    const char *env = std::getenv("POWERSERVE_KV_PAGER_FILE");
    if (env && env[0] != '\0') {
        return env;
    }
    return weights_path + "." + model_id + ".kvpager.bin";
}

KVRuntimeState prepare_kv_runtime(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    const std::string &weights_path,
    const std::string &model_id,
    const std::vector<int> &pos,
    size_t n_layers
) {
    KVRuntimeState state{};
    const bool use_kv_pager = is_kv_pager_enabled() || ggml_kv.slot_mode_enabled();
    state.pager_do_sync = is_kv_pager_sync_enabled();

    if (use_kv_pager) {
        if (!kv_pager) {
            kv_pager = std::make_unique<ggml::GGMLKVPager>(ggml_kv, get_kv_pager_file_path(weights_path, model_id));
        }
        if (!kv_pager->valid()) {
            POWERSERVE_LOG_WARN("KV pager is not valid. pager is disabled for this forward.");
        } else if (!pos.empty()) {
            const size_t kv_cursor = ggml_kv.kv_cache ? ggml_kv.kv_cache->position : 0;
            const size_t req_begin = static_cast<size_t>(pos.front());
            const size_t kv_reset_floor = ggml_kv.kv_size;

            const bool request_rewind = req_begin < kv_cursor;
            const bool request_at_reset_floor = req_begin == kv_reset_floor && kv_cursor == kv_reset_floor;
            const bool need_pager_reset = req_begin == 0 || request_rewind || request_at_reset_floor;

            if (need_pager_reset) {
                kv_pager->reset_runtime_state();
                if (ggml_kv.slot_mode_enabled()) {
                    ggml_kv.clear_all_mappings();
                }
            }
        }
    }

    state.pager_active = use_kv_pager && kv_pager && kv_pager->valid();
    if (ggml_kv.slot_mode_enabled() && !state.pager_active) {
        POWERSERVE_ABORT("slot mode requires KV pager to be active");
    }
    state.layer_computed_this_step.assign(n_layers, false);
    return state;
}

void prepare_kv_segment(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t n_layers,
    size_t tokens_before_step,
    size_t tokens_after_step
) {
    if (!state.pager_active) {
        return;
    }

    for (size_t L = begin; L < end; ++L) {
        prefetch_layer_with_optional_binding(
            kv_pager,
            ggml_kv,
            state,
            L,
            n_layers,
            tokens_before_step,
            tokens_after_step,
            true
        );
    }

    const size_t segment_layers = end - begin;
    const size_t lookahead_begin = end;
    const size_t lookahead_end = std::min(end + segment_layers, n_layers);
    for (size_t L = lookahead_begin; L < lookahead_end; ++L) {
        // Lookahead prefetch must not block current segment execution by forcing evictions.
        prefetch_layer_with_optional_binding(
            kv_pager,
            ggml_kv,
            state,
            L,
            n_layers,
            tokens_before_step,
            tokens_after_step,
            false
        );
    }
}

void wait_kv_segment_ready(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    const ggml::GGMLKV &ggml_kv,
    const KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t tokens_before_step
) {
    if (!state.pager_active) {
        return;
    }

    for (size_t L = begin; L < end; ++L) {
        POWERSERVE_ASSERT(ggml_kv.layer_to_slot(L) >= 0, "layer {} is not mapped before wait", L);
        if (!kv_pager->wait_layer_ready(L, tokens_before_step)) {
            POWERSERVE_LOG_ERROR("KV pager acquire(wait) failed at layer {}", L);
            POWERSERVE_ABORT("KV pager acquire wait failure");
        }
    }
}

void finish_kv_segment(
    std::unique_ptr<ggml::GGMLKVPager> &kv_pager,
    ggml::GGMLKV &ggml_kv,
    KVRuntimeState &state,
    size_t begin,
    size_t end,
    size_t n_layers,
    size_t tokens_before_step,
    size_t tokens_after_step
) {
    if (!state.pager_active) {
        return;
    }

    for (size_t L = begin; L < end; ++L) {
        state.layer_computed_this_step[L] = true;
        kv_pager->mark_dirty_layer(L);
        const size_t keep = ggml_kv.slot_window_size();
        if (L >= keep) {
            const size_t victim_layer = L - keep;
            if (ggml_kv.layer_to_slot(victim_layer) < 0) {
                continue;
            }
            if (!kv_pager->evict_layer_async(victim_layer, tokens_after_step, false)) {
                POWERSERVE_LOG_ERROR("KV pager evict failed at layer {}", victim_layer);
                POWERSERVE_ABORT("KV pager evict failure");
            }
        }
    }

    // Proactively free slots and prefetch the next segment right after current compute finishes.
    // This avoids delaying eviction until the next segment's prepare path.
    const size_t segment_layers = end - begin;
    const size_t lookahead_begin = end;
    const size_t lookahead_end = std::min(end + segment_layers, n_layers);
    for (size_t L = lookahead_begin; L < lookahead_end; ++L) {
        prefetch_layer_with_optional_binding(
            kv_pager,
            ggml_kv,
            state,
            L,
            n_layers,
            tokens_before_step,
            tokens_after_step,
            true
        );
    }
}

void sync_kv_runtime_if_needed(std::unique_ptr<ggml::GGMLKVPager> &kv_pager, const KVRuntimeState &state) {
    if (!state.pager_active || !state.pager_do_sync) {
        return;
    }
    if (!kv_pager->wait_all_evictions()) {
        POWERSERVE_LOG_WARN("KV pager wait_all_evictions failed before sync");
    }
    if (!kv_pager->sync()) {
        POWERSERVE_LOG_WARN("KV pager sync failed");
    }
}

} // namespace powerserve::ggml_runtime
