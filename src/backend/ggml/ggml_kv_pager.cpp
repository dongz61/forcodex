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
    m_layer_states.assign(m_n_layers, LayerState::ResidentClean);
    m_persisted_tokens.assign(m_n_layers, 0);

    m_fd = open(m_file_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (m_fd < 0) {
        POWERSERVE_LOG_ERROR("KV pager open failed: path={} err={}", m_file_path, std::strerror(errno));
        return;
    }

    const size_t data_bytes = m_n_layers * 2 * m_layer_bytes;
    if (!preallocate_file(m_header_bytes + data_bytes)) {
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

    POWERSERVE_LOG_INFO(
        "KV pager ready: path={} layers={} n_ctx={} kv_dim={} layer_bytes={}",
        m_file_path,
        m_n_layers,
        m_n_ctx,
        m_kv_dim,
        m_layer_bytes
    );
}

GGMLKVPager::~GGMLKVPager() {
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
}

void GGMLKVPager::reset_runtime_state() {
    std::fill(m_layer_states.begin(), m_layer_states.end(), LayerState::ResidentClean);
    std::fill(m_persisted_tokens.begin(), m_persisted_tokens.end(), 0);
}

bool GGMLKVPager::acquire_layer(size_t layer_id, size_t need_tokens) {
    if (!valid() || layer_id >= m_n_layers) {
        return false;
    }
    if (m_layer_states[layer_id] != LayerState::Unloaded) {
        return true;
    }

    const size_t capped_need = std::min(need_tokens, m_n_ctx);
    const size_t load_tokens = std::min(capped_need, m_persisted_tokens[layer_id]);
    const size_t load_bytes = load_tokens * m_kv_dim * sizeof(float);

    auto &k = m_kv.chunk.key_buffer[layer_id];
    auto &v = m_kv.chunk.value_buffer[layer_id];
    if (load_bytes > 0) {
        if (!pread_all(k.data(), load_bytes, offset_k(layer_id))) {
            POWERSERVE_LOG_ERROR("KV pager acquire K failed: layer={} err={}", layer_id, std::strerror(errno));
            return false;
        }
        if (!pread_all(v.data(), load_bytes, offset_v(layer_id))) {
            POWERSERVE_LOG_ERROR("KV pager acquire V failed: layer={} err={}", layer_id, std::strerror(errno));
            return false;
        }
    }

    const size_t need_bytes = capped_need * m_kv_dim * sizeof(float);
    if (need_bytes > load_bytes) {
        std::memset(static_cast<char *>(static_cast<void *>(k.data())) + load_bytes, 0, need_bytes - load_bytes);
        std::memset(static_cast<char *>(static_cast<void *>(v.data())) + load_bytes, 0, need_bytes - load_bytes);
    }

    m_layer_states[layer_id] = LayerState::ResidentClean;
    return true;
}

void GGMLKVPager::mark_dirty_layer(size_t layer_id) {
    if (!valid() || layer_id >= m_n_layers) {
        return;
    }
    if (m_layer_states[layer_id] != LayerState::Unloaded) {
        m_layer_states[layer_id] = LayerState::ResidentDirty;
    }
}

bool GGMLKVPager::evict_layer(size_t layer_id, size_t valid_tokens, bool do_sync) {
    if (!valid() || layer_id >= m_n_layers) {
        return false;
    }
    if (m_layer_states[layer_id] == LayerState::Unloaded) {
        return true;
    }

    const size_t capped_tokens = std::min(valid_tokens, m_n_ctx);
    if (m_layer_states[layer_id] == LayerState::ResidentDirty) {
        const size_t write_bytes = capped_tokens * m_kv_dim * sizeof(float);
        auto &k = m_kv.chunk.key_buffer[layer_id];
        auto &v = m_kv.chunk.value_buffer[layer_id];

        if (write_bytes > 0) {
            if (!pwrite_all(k.data(), write_bytes, offset_k(layer_id))) {
                POWERSERVE_LOG_ERROR("KV pager evict K failed: layer={} err={}", layer_id, std::strerror(errno));
                return false;
            }
            if (!pwrite_all(v.data(), write_bytes, offset_v(layer_id))) {
                POWERSERVE_LOG_ERROR("KV pager evict V failed: layer={} err={}", layer_id, std::strerror(errno));
                return false;
            }
        }
        m_persisted_tokens[layer_id] = std::max(m_persisted_tokens[layer_id], capped_tokens);
    }

    std::fill(m_kv.chunk.key_buffer[layer_id].begin(), m_kv.chunk.key_buffer[layer_id].end(), 0.0f);
    std::fill(m_kv.chunk.value_buffer[layer_id].begin(), m_kv.chunk.value_buffer[layer_id].end(), 0.0f);

    m_layer_states[layer_id] = LayerState::Unloaded;
    if (do_sync) {
        return sync();
    }
    return true;
}

bool GGMLKVPager::sync() {
    if (!valid()) {
        return false;
    }
    return fdatasync(m_fd) == 0;
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

bool GGMLKVPager::pread_all(void *dst, size_t bytes, int64_t offset) {
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

