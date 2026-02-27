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

#include "core/config.hpp"
#include "core/kv_cache.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace powerserve::ggml {

struct GGMLKV {
public:
    using KVBuffer = std::vector<std::vector<float>>;

    size_t m_kv_dim     = 0;
    size_t m_n_kv_heads = 0;
    size_t m_n_ctx      = 0;
    size_t m_n_layers   = 0;
    size_t m_n_slots    = 0;
    size_t m_head_size  = 0;
    size_t m_batch_size = 0;
    bool m_slot_mode    = false;
    size_t kv_size      = 0; // system_prompt size
    const ModelConfig::LLMConfig &m_config;

    struct GGMLChunk {
        KVBuffer key_buffer;          // [n_layers][seq_len * kv_dim]) kv_dim == n_kv_heads * head_size
        KVBuffer value_buffer;        // [n_layers][seq_len * kv_dim])
        KVBuffer current_k;           // [n_layers][batch_size * kv_dim])
        KVBuffer current_v;           // [n_layers][batch_size * kv_dim])
        std::vector<float> attn_bias; // [batch_size * n_ctx]

        std::vector<Tensor> key_tensors;   // n_layers
        std::vector<Tensor> value_tensors; // n_layers
    };

    GGMLChunk chunk;
    std::vector<int> m_layer_to_slot; // [n_layers] -> slot id or -1
    std::vector<int> m_slot_to_layer; // [n_slots]  -> layer id or -1

    struct GGMLKVInterface {
        GGMLKV &parent;
        GGMLChunk &chunk;

        GGMLKVInterface(GGMLKV &parent, GGMLChunk &chunk) : parent(parent), chunk(chunk) {}

        // Note: get entry from temporary kv
        ALWAYS_INLINE auto get_key(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_k[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto get_value(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_v[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        };

        // Note: get entry from KV cache
        ALWAYS_INLINE auto key_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = parent.key_buffer_for_layer(cache_pos.layer_id);
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto value_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = parent.value_buffer_for_layer(cache_pos.layer_id);
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE void set_mask(size_t cache_index, bool mask) {
            for (size_t i = 0; i < parent.m_batch_size; i++) {
                auto attn_bias         = chunk.attn_bias.data() + i * parent.m_n_ctx;
                attn_bias[cache_index] = mask ? -INFINITY : 0;
            }
        }
    };

public:
    std::unique_ptr<KVCache<GGMLKVInterface>> kv_cache;

public:
    GGMLKV(const ModelConfig::LLMConfig &config);

    ~GGMLKV() = default;

public:
    void reset_batch_size(const size_t &batch_size) {
        if (m_batch_size == batch_size)
            return;
        // fmt::println("Resize batch size: {} -> {}", m_batch_size, batch_size);
        m_batch_size = batch_size;

        auto &k = chunk.current_k;
        auto &v = chunk.current_v;
        for (size_t L = 0; L < m_n_layers; L++) {
            k[L].resize(m_batch_size * m_kv_dim);
            v[L].resize(m_batch_size * m_kv_dim);
        }
        chunk.attn_bias.resize(m_batch_size * m_n_ctx);
    }

    void reset_kv_cache() {
        kv_cache->truncate_tokens(kv_size);
    }

    void save_kv(int size) {
        kv_cache->save_tokens(size);
    }

    void advance(int size) {
        kv_cache->advance_tokens(size);
    }

    auto get_cache(size_t L) -> std::pair<Tensor &, Tensor &> {
        const size_t slot = logical_layer_to_slot_or_die(L);
        return {chunk.key_tensors[slot], chunk.value_tensors[slot]};
    }

    bool slot_mode_enabled() const {
        return m_slot_mode;
    }

    size_t slot_window_size() const {
        return m_n_slots;
    }

    int layer_to_slot(size_t layer_id) const {
        POWERSERVE_ASSERT(layer_id < m_layer_to_slot.size());
        return m_layer_to_slot[layer_id];
    }

    int find_free_slot() const {
        for (size_t s = 0; s < m_slot_to_layer.size(); ++s) {
            if (m_slot_to_layer[s] < 0) {
                return static_cast<int>(s);
            }
        }
        return -1;
    }

    void bind_layer_to_slot(size_t layer_id, size_t slot_id) {
        POWERSERVE_ASSERT(layer_id < m_n_layers);
        POWERSERVE_ASSERT(slot_id < m_n_slots);

        const int prev_layer = m_slot_to_layer[slot_id];
        if (prev_layer >= 0) {
            m_layer_to_slot[static_cast<size_t>(prev_layer)] = -1;
        }

        const int prev_slot = m_layer_to_slot[layer_id];
        if (prev_slot >= 0) {
            m_slot_to_layer[static_cast<size_t>(prev_slot)] = -1;
        }

        m_layer_to_slot[layer_id] = static_cast<int>(slot_id);
        m_slot_to_layer[slot_id] = static_cast<int>(layer_id);
    }

    void unbind_layer(size_t layer_id) {
        POWERSERVE_ASSERT(layer_id < m_n_layers);
        const int slot = m_layer_to_slot[layer_id];
        if (slot >= 0) {
            m_slot_to_layer[static_cast<size_t>(slot)] = -1;
            m_layer_to_slot[layer_id] = -1;
        }
    }

    auto key_buffer_for_layer(size_t layer_id) -> std::vector<float> & {
        const size_t slot = logical_layer_to_slot_or_die(layer_id);
        return chunk.key_buffer[slot];
    }

    auto value_buffer_for_layer(size_t layer_id) -> std::vector<float> & {
        const size_t slot = logical_layer_to_slot_or_die(layer_id);
        return chunk.value_buffer[slot];
    }

    auto key_buffer_for_layer(size_t layer_id) const -> const std::vector<float> & {
        const size_t slot = logical_layer_to_slot_or_die(layer_id);
        return chunk.key_buffer[slot];
    }

    auto value_buffer_for_layer(size_t layer_id) const -> const std::vector<float> & {
        const size_t slot = logical_layer_to_slot_or_die(layer_id);
        return chunk.value_buffer[slot];
    }

    void clear_all_mappings() {
        std::fill(m_layer_to_slot.begin(), m_layer_to_slot.end(), -1);
        std::fill(m_slot_to_layer.begin(), m_slot_to_layer.end(), -1);
    }

private:
    size_t logical_layer_to_slot_or_die(size_t layer_id) const {
        POWERSERVE_ASSERT(layer_id < m_n_layers);
        if (!m_slot_mode) {
            return layer_id;
        }
        const int slot = m_layer_to_slot[layer_id];
        POWERSERVE_ASSERT(slot >= 0, "layer {} is not mapped to any KV slot", layer_id);
        return static_cast<size_t>(slot);
    }

    void prepare_model_chunk();
};

} // namespace powerserve::ggml
