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

namespace powerserve::ggml {

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
        key_buffer[layer_id].resize(layer_size);
        value_buffer[layer_id].resize(layer_size);

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
        chunk.key_buffer[layer_id].resize(layer_size);
        chunk.value_buffer[layer_id].resize(layer_size);
    }
    bind_full_kv_tensors();
    m_full_kv_allocated = true;
}

void GGMLKV::release_full_kv_storage() {
    if (!m_full_kv_allocated) {
        return;
    }
    for (size_t layer_id = 0; layer_id < m_n_layers; ++layer_id) {
        chunk.key_buffer[layer_id].clear();
        chunk.key_buffer[layer_id].shrink_to_fit();
        chunk.value_buffer[layer_id].clear();
        chunk.value_buffer[layer_id].shrink_to_fit();
        chunk.key_tensors[layer_id].m_data.reset();
        chunk.value_tensors[layer_id].m_data.reset();
    }
    m_full_kv_allocated = false;
}

} // namespace powerserve::ggml
