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

#include "backend/ggml/ggml_kv_cache.hpp"

#include "backend/cpu_buffer.hpp"

#include <sys/mman.h>

namespace powerserve::ggml {

MappedFloatBuffer::~MappedFloatBuffer() {
    release();
}

MappedFloatBuffer::MappedFloatBuffer(MappedFloatBuffer &&other) noexcept {
    move_from(std::move(other));
}

auto MappedFloatBuffer::operator=(MappedFloatBuffer &&other) noexcept -> MappedFloatBuffer & {
    if (this != &other) {
        release();
        move_from(std::move(other));
    }
    return *this;
}

bool MappedFloatBuffer::allocate(size_t n_floats) {
    release();
    if (n_floats == 0) {
        return true;
    }

    m_size = n_floats;
    m_bytes = n_floats * sizeof(float);
    void *mapped = mmap(nullptr, m_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) {
        m_data = nullptr;
        m_size = 0;
        m_bytes = 0;
        return false;
    }

    m_data = static_cast<float *>(mapped);
    zero_fill();
    return true;
}

void MappedFloatBuffer::release() {
    if (m_data != nullptr) {
        const int rc = munmap(m_data, m_bytes);
        POWERSERVE_ASSERT(rc == 0, "munmap failed for mapped float buffer");
    }
    m_data = nullptr;
    m_size = 0;
    m_bytes = 0;
}

void MappedFloatBuffer::move_from(MappedFloatBuffer &&other) noexcept {
    m_data = other.m_data;
    m_size = other.m_size;
    m_bytes = other.m_bytes;
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_bytes = 0;
}

GGMLKV::GGMLKV(const ModelConfig::LLMConfig &config) :
    m_kv_dim(config.kv_dim),
    m_n_kv_heads(config.n_kv_heads),
    m_n_ctx(config.seq_len),
    m_n_layers(config.n_layers),
    m_head_size(config.head_size),
    m_batch_size(1), // FIXME:
    m_config(config) {
    prepare_model_chunk();

    kv_cache = std::make_unique<KVCache<GGMLKVInterface>>(m_n_layers, m_n_kv_heads, m_n_ctx, *this, chunk);
}

void GGMLKV::prepare_model_chunk() {
    auto &key_buffer   = chunk.key_buffer;
    auto &value_buffer = chunk.value_buffer;
    auto &k            = chunk.current_k;
    auto &v            = chunk.current_v;

    key_buffer.resize(m_n_layers);
    value_buffer.resize(m_n_layers);
    chunk.key_tensors.clear();
    chunk.value_tensors.clear();
    chunk.key_tensors.reserve(m_n_layers);
    chunk.value_tensors.reserve(m_n_layers);
    size_t layer_size = m_kv_dim * m_n_ctx;
    for (size_t layer_id = 0; layer_id < m_n_layers; layer_id++) {
        POWERSERVE_ASSERT(key_buffer[layer_id].allocate(layer_size), "failed to mmap key buffer for layer {}", layer_id);
        POWERSERVE_ASSERT(
            value_buffer[layer_id].allocate(layer_size),
            "failed to mmap value buffer for layer {}",
            layer_id
        );

        chunk.key_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
        chunk.value_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
    }
    bind_full_kv_tensors();
    m_full_kv_allocated = true;

    k.resize(m_n_layers);
    v.resize(m_n_layers);
    for (size_t L = 0; L < m_n_layers; L++) {
        k[L].resize(m_batch_size * m_kv_dim);
        v[L].resize(m_batch_size * m_kv_dim);
    }

    auto &attn_bias = chunk.attn_bias;
    attn_bias.resize(m_batch_size * m_n_ctx);
}

void GGMLKV::bind_full_kv_tensors() {
    Stride stride = {
        sizeof(float),
        sizeof(float) * m_n_ctx,
        sizeof(float) * m_kv_dim * m_n_ctx,
        sizeof(float) * m_kv_dim * m_n_ctx
    };
    for (size_t layer_id = 0; layer_id < m_n_layers; ++layer_id) {
        chunk.key_tensors[layer_id].m_data   = std::make_shared<CPUBuffer>(stride, chunk.key_buffer[layer_id].data());
        chunk.value_tensors[layer_id].m_data = std::make_shared<CPUBuffer>(stride, chunk.value_buffer[layer_id].data());
    }
}

void GGMLKV::ensure_full_kv_storage() {
    if (m_full_kv_allocated) {
        return;
    }
    const size_t layer_size = m_kv_dim * m_n_ctx;
    for (size_t layer_id = 0; layer_id < m_n_layers; ++layer_id) {
        POWERSERVE_ASSERT(
            chunk.key_buffer[layer_id].allocate(layer_size),
            "failed to mmap key buffer for layer {}",
            layer_id
        );
        POWERSERVE_ASSERT(
            chunk.value_buffer[layer_id].allocate(layer_size),
            "failed to mmap value buffer for layer {}",
            layer_id
        );
    }
    bind_full_kv_tensors();
    m_full_kv_allocated = true;
}

void GGMLKV::release_full_kv_storage() {
    if (!m_full_kv_allocated) {
        return;
    }
    for (size_t layer_id = 0; layer_id < m_n_layers; ++layer_id) {
        chunk.key_tensors[layer_id].m_data.reset();
        chunk.value_tensors[layer_id].m_data.reset();
        chunk.key_buffer[layer_id].release();
        chunk.value_buffer[layer_id].release();
    }
    m_full_kv_allocated = false;
}

} // namespace powerserve::ggml
