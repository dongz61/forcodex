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
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace powerserve::ggml {

struct CompactClusterKV {
    Tensor key;
    Tensor value;
    std::vector<int> positions;
};

struct GGMLKVPager {
public:
    enum class LayerState : uint8_t {
        Unloaded = 0,
        Loading,
        ResidentClean,
        ResidentDirty,
        Writing,
    };

public:
    GGMLKVPager(GGMLKV &kv, const std::string &file_path);
    ~GGMLKVPager();

    GGMLKVPager(const GGMLKVPager &) = delete;
    GGMLKVPager &operator=(const GGMLKVPager &) = delete;

public:
    bool valid() const {
        return m_fd >= 0;
    }

    void reset_runtime_state();
    bool prefetch_layer_async(size_t layer_id, size_t need_tokens);
    bool wait_layer_ready(size_t layer_id, size_t need_tokens);
    bool evict_layer_async(size_t layer_id, size_t valid_tokens, bool do_sync);
    bool wait_layer_evicted(size_t layer_id);
    bool wait_all_async();
    bool wait_all_evictions();

    bool acquire_layer(size_t layer_id, size_t need_tokens);
    void mark_dirty_layer(size_t layer_id);
    bool evict_layer(size_t layer_id, size_t valid_tokens, bool do_sync);
    bool sync();
    auto materialize_compact_kv(size_t layer_id, const std::vector<int> &token_positions) const -> CompactClusterKV;

private:
    bool acquire_layer_sync(size_t layer_id, size_t need_tokens);
    bool evict_layer_sync(size_t layer_id, size_t valid_tokens, bool do_sync, bool write_data);

    bool write_header();
    bool preallocate_file(size_t bytes);
    bool pwrite_all(const void *src, size_t bytes, int64_t offset);
    bool pread_all(void *dst, size_t bytes, int64_t offset);

    int64_t offset_k(size_t layer_id) const;
    int64_t offset_v(size_t layer_id) const;

private:
    GGMLKV &m_kv;
    int m_fd = -1;
    std::string m_file_path;

    size_t m_n_layers = 0;     
    size_t m_n_ctx = 0;
    size_t m_kv_dim = 0;
    size_t m_layer_bytes = 0;
    size_t m_header_bytes = 4096;

    std::vector<LayerState> m_layer_states;
    std::vector<size_t> m_persisted_tokens;
    std::vector<std::future<bool>> m_load_futures;
    std::vector<std::future<bool>> m_evict_futures;
    std::vector<bool> m_load_inflight;
    std::vector<bool> m_evict_inflight;
    std::mutex m_async_mutex;
};

} // namespace powerserve::ggml
