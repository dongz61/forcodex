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

#include "backend/ggml/ggml_kv_pager.hpp"

#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace powerserve::ggml {

namespace {
struct KVPagerFileHeader {
    char magic[8];
    uint32_t version;
    uint32_t dtype;
    uint32_t n_layers;
    uint32_t n_ctx;
    uint32_t kv_dim;
    uint32_t elem_size;
    uint32_t reserved[9];
};
} // namespace

GGMLKVPager::GGMLKVPager(GGMLKV &kv, const std::string &file_path) :
    m_kv(kv),
    m_file_path(file_path),
    m_n_layers(kv.m_n_layers),
    m_n_ctx(kv.m_n_ctx),
    m_kv_dim(kv.m_kv_dim) {
    m_layer_bytes = m_n_ctx * m_kv_dim * sizeof(float);
    m_next_cluster_offset = static_cast<int64_t>(m_header_bytes);

    m_fd = open(m_file_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (m_fd < 0) {
        POWERSERVE_LOG_ERROR("KV pager open failed: path={} err={}", m_file_path, std::strerror(errno));
        return;
    }

    if (!preallocate_file(m_header_bytes)) {
        POWERSERVE_LOG_ERROR("KV pager preallocate failed: path={} err={}", m_file_path, std::strerror(errno));
        close(m_fd);
        m_fd = -1;
        return;
    }

    if (!write_header()) {
        POWERSERVE_LOG_ERROR("KV pager header write failed: path={} err={}", m_file_path, std::strerror(errno));
        close(m_fd);
        m_fd = -1;
        return;
    }
}

GGMLKVPager::~GGMLKVPager() {
    if (!wait_all_async()) {
        POWERSERVE_LOG_WARN("KV pager async wait failed on destroy");
    }
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
    if (!m_file_path.empty()) {
        if (unlink(m_file_path.c_str()) != 0 && errno != ENOENT) {
            POWERSERVE_LOG_WARN("KV pager cleanup failed: path={} err={}", m_file_path, std::strerror(errno));
        }
    }
}

void GGMLKVPager::reset_runtime_state() {}

bool GGMLKVPager::wait_all_async() {
    return true;
}

bool GGMLKVPager::sync() {
    if (!valid()) {
        return false;
    }
    return fdatasync(m_fd) == 0;
}

auto GGMLKVPager::materialize_compact_kv(size_t layer_id, const std::vector<int> &token_positions) const -> CompactClusterKV {
    POWERSERVE_ASSERT(layer_id < m_n_layers);

    CompactClusterKV out{
        .key = Tensor(DataType::FP32, {token_positions.size(), m_kv_dim, 1, 1}),
        .value = Tensor(DataType::FP32, {token_positions.size(), m_kv_dim, 1, 1}),
        .positions = token_positions,
    };
    std::sort(out.positions.begin(), out.positions.end());
    out.positions.erase(std::unique(out.positions.begin(), out.positions.end()), out.positions.end());

    out.key.m_shape[0] = out.positions.size();
    out.value.m_shape[0] = out.positions.size();
    out.key.m_data = CPUBuffer::create_buffer<float>(out.key.m_shape);
    out.value.m_data = CPUBuffer::create_buffer<float>(out.value.m_shape);

    float *key_dst = static_cast<float *>(out.key.get<CPUBuffer>().m_data);
    float *value_dst = static_cast<float *>(out.value.get<CPUBuffer>().m_data);
    const auto &key_src = m_kv.key_buffer_for_layer(layer_id);
    const auto &value_src = m_kv.value_buffer_for_layer(layer_id);
    const size_t compact_tokens = out.positions.size();

    for (size_t compact_idx = 0; compact_idx < compact_tokens; ++compact_idx) {
        const int token_position = out.positions[compact_idx];
        POWERSERVE_ASSERT(token_position >= 0);
        POWERSERVE_ASSERT(static_cast<size_t>(token_position) < m_n_ctx);

        const size_t src_k_offset = static_cast<size_t>(token_position) * m_kv_dim;
        const size_t dst_k_offset = compact_idx * m_kv_dim;
        std::copy_n(
            key_src.begin() + static_cast<std::ptrdiff_t>(src_k_offset),
            static_cast<std::ptrdiff_t>(m_kv_dim),
            key_dst + static_cast<std::ptrdiff_t>(dst_k_offset)
        );

        for (size_t dim = 0; dim < m_kv_dim; ++dim) {
            value_dst[dim * compact_tokens + compact_idx] =
                value_src[dim * m_n_ctx + static_cast<size_t>(token_position)];
        }
    }

    return out;
}

bool GGMLKVPager::read_cluster(
    int64_t disk_offset,
    size_t capacity_tokens,
    size_t token_count,
    float *key_dst,
    float *value_dst
) const {
    if (!valid() || disk_offset < 0) {
        return false;
    }
    const size_t key_bytes = token_count * m_kv_dim * sizeof(float);
    if (key_bytes > 0 && !pread_all(key_dst, key_bytes, disk_offset)) {
        return false;
    }
    const int64_t value_offset =
        disk_offset + static_cast<int64_t>(capacity_tokens * m_kv_dim * sizeof(float));
    const size_t value_seg_bytes = token_count * sizeof(float);
    for (size_t d = 0; d < m_kv_dim; ++d) {
        if (value_seg_bytes == 0) {
            continue;
        }
        if (!pread_all(
                value_dst + d * token_count,
                value_seg_bytes,
                value_offset + static_cast<int64_t>((d * capacity_tokens) * sizeof(float))
            )) {
            return false;
        }
    }
    return true;
}

bool GGMLKVPager::write_cluster_full(
    int64_t disk_offset,
    size_t capacity_tokens,
    size_t token_count,
    const float *key_src,
    const float *value_src
) {
    if (!valid() || disk_offset < 0) {
        return false;
    }
    const size_t key_bytes = token_count * m_kv_dim * sizeof(float);
    if (key_bytes > 0 && !pwrite_all(key_src, key_bytes, disk_offset)) {
        return false;
    }
    const int64_t value_offset =
        disk_offset + static_cast<int64_t>(capacity_tokens * m_kv_dim * sizeof(float));
    const size_t value_seg_bytes = token_count * sizeof(float);
    for (size_t d = 0; d < m_kv_dim; ++d) {
        if (value_seg_bytes == 0) {
            continue;
        }
        if (!pwrite_all(
                value_src + d * token_count,
                value_seg_bytes,
                value_offset + static_cast<int64_t>((d * capacity_tokens) * sizeof(float))
            )) {
            return false;
        }
    }
    return true;
}

bool GGMLKVPager::append_cluster_tokens(
    int64_t disk_offset,
    size_t capacity_tokens,
    size_t start_token,
    size_t append_count,
    const float *key_src,
    const float *value_src
) {
    if (!valid() || disk_offset < 0) {
        return false;
    }
    const size_t key_token_bytes = m_kv_dim * sizeof(float);
    const size_t key_bytes = append_count * key_token_bytes;
    if (key_bytes > 0 &&
        !pwrite_all(
            key_src,
            key_bytes,
            disk_offset + static_cast<int64_t>(start_token * key_token_bytes)
        )) {
        return false;
    }

    const int64_t value_offset =
        disk_offset + static_cast<int64_t>(capacity_tokens * m_kv_dim * sizeof(float));
    const size_t value_seg_bytes = append_count * sizeof(float);
    for (size_t d = 0; d < m_kv_dim; ++d) {
        if (value_seg_bytes == 0) {
            continue;
        }
        if (!pwrite_all(
                value_src + d * append_count,
                value_seg_bytes,
                value_offset + static_cast<int64_t>((d * capacity_tokens + start_token) * sizeof(float))
            )) {
            return false;
        }
    }
    return true;
}

auto GGMLKVPager::allocate_cluster_region(size_t capacity_tokens) -> int64_t {
    if (!valid()) {
        return -1;
    }
    const size_t cluster_bytes = 2 * capacity_tokens * m_kv_dim * sizeof(float);
    const int64_t out = m_next_cluster_offset;
    m_next_cluster_offset += static_cast<int64_t>(cluster_bytes);
    if (!preallocate_file(static_cast<size_t>(m_next_cluster_offset))) {
        return -1;
    }
    return out;
}

bool GGMLKVPager::write_header() {
    KVPagerFileHeader hdr{};
    std::memcpy(hdr.magic, "PSKVPGR", 7);
    hdr.version = 1;
    hdr.dtype = static_cast<uint32_t>(DataType::FP32);
    hdr.n_layers = static_cast<uint32_t>(m_n_layers);
    hdr.n_ctx = static_cast<uint32_t>(m_n_ctx);
    hdr.kv_dim = static_cast<uint32_t>(m_kv_dim);
    hdr.elem_size = static_cast<uint32_t>(sizeof(float));

    return pwrite_all(&hdr, sizeof(hdr), 0);
}

bool GGMLKVPager::preallocate_file(size_t bytes) {
    const int rc = posix_fallocate(m_fd, 0, static_cast<off_t>(bytes));
    if (rc == 0) {
        return true;
    }
    return ftruncate(m_fd, static_cast<off_t>(bytes)) == 0;
}

bool GGMLKVPager::pwrite_all(const void *src, size_t bytes, int64_t offset) {
    size_t done = 0;
    const auto *buf = static_cast<const char *>(src);
    while (done < bytes) {
        const ssize_t n = pwrite(m_fd, buf + done, bytes - done, offset + static_cast<int64_t>(done));
        if (n <= 0) {
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool GGMLKVPager::pread_all(void *dst, size_t bytes, int64_t offset) const {
    size_t done = 0;
    auto *buf = static_cast<char *>(dst);
    while (done < bytes) {
        const ssize_t n = pread(m_fd, buf + done, bytes - done, offset + static_cast<int64_t>(done));
        if (n <= 0) {
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

int64_t GGMLKVPager::offset_k(size_t layer_id) const {
    return static_cast<int64_t>(m_header_bytes + (2 * layer_id) * m_layer_bytes);
}

int64_t GGMLKVPager::offset_v(size_t layer_id) const {
    return static_cast<int64_t>(m_header_bytes + (2 * layer_id + 1) * m_layer_bytes);
}

} // namespace powerserve::ggml
