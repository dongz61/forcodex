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
#include <string>
#include <vector>

namespace powerserve::ggml {

struct CompactClusterKV {
    Tensor key;
    Tensor value;
    std::vector<int> positions;
};

struct ClusterDiskLayout {
    int64_t offset = -1;
    size_t token_count = 0;
    size_t capacity_tokens = 0;
};

struct GGMLKVPager {
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
    bool wait_all_async();
    bool sync();
    auto materialize_compact_kv(size_t layer_id, const std::vector<int> &token_positions) const -> CompactClusterKV;

    bool read_cluster(
        int64_t disk_offset,
        size_t capacity_tokens,
        size_t token_count,
        float *key_dst,
        float *value_dst
    ) const;
    bool write_cluster_full(
        int64_t disk_offset,
        size_t capacity_tokens,
        size_t token_count,
        const float *key_src,
        const float *value_src
    );
    bool append_cluster_tokens(
        int64_t disk_offset,
        size_t capacity_tokens,
        size_t start_token,
        size_t append_count,
        const float *key_src,
        const float *value_src
    );
    auto allocate_cluster_region(size_t capacity_tokens) -> int64_t;

private:
    bool write_header();
    bool preallocate_file(size_t bytes);
    bool pwrite_all(const void *src, size_t bytes, int64_t offset);
    bool pread_all(void *dst, size_t bytes, int64_t offset) const;

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
    int64_t m_next_cluster_offset = 0;
};

} // namespace powerserve::ggml
