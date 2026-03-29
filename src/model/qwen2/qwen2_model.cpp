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

#include "qwen2_model.hpp"

#include "backend/cpu_buffer.hpp"
#include "backend/ggml/ggml_cluster_manager.hpp"
#include "backend/ggml/ggml_kv_pager.hpp"
#include "core/logger.hpp"
#include "core/perfetto_trace.hpp"
#include "core/timer.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/module/ggml_cluster_runtime.hpp"
#include "model/module/ggml_kv_runtime.hpp"
#include "model/qwen2/qwen2_weight.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace powerserve {

static bool is_ggml_layer_profile_enabled() {
    const char *v = std::getenv("POWERSERVE_GGML_LAYER_PROFILE");
    if (!v) {
        return false;
    }
    return std::strcmp(v, "1") == 0 ||
           std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 ||
           std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
}

Qwen2Model::Qwen2Model(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        POWERSERVE_ASSERT(gguf_ctx != nullptr);
        POWERSERVE_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output_norm.weight") == nullptr ? true : false;
    m_weights = std::make_shared<Qwen2Weight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        POWERSERVE_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

Qwen2Model::~Qwen2Model() {
    gguf_free(gguf_ctx);
}

void Qwen2Model::ensure_cluster_manager() {
    if (!m_platform || !m_platform->ggml_backends[m_config->model_id]) {
        return;
    }
    auto *ggml_kv = m_platform->ggml_backends[m_config->model_id]->m_kv.get();
    if (!m_cluster_manager) {
        m_cluster_manager = std::make_unique<ggml::GGMLClusterManager>(*ggml_kv);
    }
    if (!m_kv_pager) {
        m_kv_pager = std::make_unique<ggml::GGMLKVPager>(
            *ggml_kv,
            ggml_runtime::get_kv_pager_file_path(m_filename, m_config->model_id)
        );
    }
    ggml::register_cluster_runtime(m_config->model_id, m_cluster_manager.get(), m_kv_pager.get());
    ggml::set_cluster_runtime_ready(m_config->model_id, false);
}

void Qwen2Model::on_prefill_finished() {
    if (!m_platform) {
        return;
    }
    bool has_qnn_backend = false;
#if defined(POWERSERVE_WITH_QNN)
    has_qnn_backend = (m_platform->qnn_backend != nullptr);
#endif
    if (lazy_load || m_platform->using_opencl(m_config->model_id) || has_qnn_backend) {
        return;
    }
    ensure_cluster_manager();
    if (!m_cluster_manager || !m_cluster_manager->enabled()) {
        return;
    }
    if (m_platform->ggml_backends[m_config->model_id]->m_kv->slot_mode_enabled()) {
        POWERSERVE_LOG_WARN("GGML cluster manager is currently disabled when slot mode is active");
        return;
    }
    m_cluster_manager->build_all_layers_after_prefill();
    ggml::set_cluster_runtime_ready(m_config->model_id, true);
}

void Qwen2Model::update_decode_clusters_for_layers(size_t begin, size_t end, int token_position) {
    if (token_position < 0 || !m_platform) {
        return;
    }
    ensure_cluster_manager();
    if (!m_cluster_manager || !m_cluster_manager->enabled()) {
        return;
    }

    auto *ggml_kv = m_platform->ggml_backends[m_config->model_id]->m_kv.get();
    if (ggml_kv->slot_mode_enabled()) {
        return;
    }
    for (size_t layer_id = begin; layer_id < end; ++layer_id) {
        const auto &key_buffer = ggml_kv->key_buffer_for_layer(layer_id);
        const size_t offset = static_cast<size_t>(token_position) * ggml_kv->m_kv_dim;
        POWERSERVE_ASSERT(offset + ggml_kv->m_kv_dim <= key_buffer.size());
        std::vector<float> new_k(
            key_buffer.begin() + static_cast<std::ptrdiff_t>(offset),
            key_buffer.begin() + static_cast<std::ptrdiff_t>(offset + ggml_kv->m_kv_dim)
        );
        m_cluster_manager->update_layer_after_decode(layer_id, token_position, new_k);
    }
}

auto Qwen2Model::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> LogitsVector {
    struct LayerProfileAccum {
        int64_t forwards = 0;
        double sum_mean_layer_ms = 0.0;
        double sum_mad_layer_ms = 0.0;
        double sum_span_layer_ms = 0.0;
    };

    static LayerProfileAccum prefill_accum;
    static LayerProfileAccum decode_accum;
    static constexpr int64_t kDecodePrintInterval = 16;

    const size_t batch_size = tokens.size();
    auto &llm_config = m_config->llm;

    bool has_qnn_backend = false;
#if defined(POWERSERVE_WITH_QNN)
    has_qnn_backend = (m_platform->qnn_backend != nullptr);
#endif

    const bool use_opencl = m_platform->using_opencl(m_config->model_id);
    const int segment_layers = ggml_runtime::get_ggml_segment_layers();
    const bool use_segmented_ggml =
        !lazy_load &&
        !use_opencl &&
        !has_qnn_backend &&
        segment_layers > 0 &&
        static_cast<size_t>(segment_layers) < llm_config.n_layers;

    if (!use_segmented_ggml && !lazy_load && !use_opencl && !has_qnn_backend) {
        auto *ggml_backend = m_platform->ggml_backends[m_config->model_id].get();
        if (ggml_backend->m_kv->slot_mode_enabled()) {
            POWERSERVE_ABORT("slot mode requires segmented execution (set POWERSERVE_GGML_SEGMENT_LAYERS > 0)");
        }
    }

    if (use_segmented_ggml) {
        auto *ggml_backend = m_platform->ggml_backends[m_config->model_id].get();
        auto *ggml_kv = ggml_backend->m_kv.get();
        ggml_backend->reset_kv_batch_size(batch_size);
        auto kv_runtime = ggml_runtime::prepare_kv_runtime(
            m_kv_pager,
            *ggml_kv,
            m_filename,
            m_config->model_id,
            pos,
            llm_config.n_layers
        );

        Tensor segment_x;
        bool has_segment_x = false;
        Tensor detached_logits;

        if (kv_runtime.pager_active && ggml_kv->slot_mode_enabled()) {
            const size_t keep = ggml_kv->slot_window_size();
            POWERSERVE_ASSERT(
                static_cast<size_t>(segment_layers) <= keep,
                "slot mode requires segment_layers <= window_layers (segment_layers={}, window_layers={})",
                segment_layers,
                keep
            );
        }

        for (size_t begin = 0; begin < llm_config.n_layers; begin += static_cast<size_t>(segment_layers)) {
            const size_t end = std::min(
                begin + static_cast<size_t>(segment_layers),
                static_cast<size_t>(llm_config.n_layers)
            );
            const bool is_last_segment = end == llm_config.n_layers;
            const size_t tokens_before_step = pos.empty() ? 0 : static_cast<size_t>(pos.front());
            const size_t tokens_after_step = pos.empty() ? 0 : static_cast<size_t>(pos.back() + 1);

            ggml_runtime::prepare_kv_segment(
                m_kv_pager,
                *ggml_kv,
                kv_runtime,
                begin,
                end,
                llm_config.n_layers,
                tokens_before_step,
                tokens_after_step
            );

            Graph g(m_config->model_id);
            TensorNode *x = nullptr;

            if (!has_segment_x) {
                auto embd_tb = g.add_tensor(m_weights->token_embedding_table);
                x = g.get_embedding(embd_tb, tokens);
            } else {
                x = g.add_tensor(segment_x);
            }

            for (size_t L = begin; L < end; ++L) {
                auto [k_cache, v_cache] = ggml_backend->m_kv->get_cache(L);
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask, true);
                auto ffn_o = m_ffn->build(g, att_o, L);
                x = ffn_o;
            }

            TensorNode *logits = nullptr;
            if (is_last_segment && lm_head) {
                auto rms_final_w = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w = g.add_tensor(m_weights->output_weight);
                logits = g.mat_mul(output_w, final_rms_norm);
            }

            Executor executor(*m_platform, g);
            executor.allocate_buffers();
            ggml_runtime::wait_kv_segment_ready(
                m_kv_pager,
                *ggml_kv,
                kv_runtime,
                begin,
                end,
                tokens_before_step
            );
            executor.run();

            ggml_runtime::finish_kv_segment(
                m_kv_pager,
                *ggml_kv,
                kv_runtime,
                begin,
                end,
                llm_config.n_layers,
                tokens_before_step,
                tokens_after_step
            );

            if (batch_size == 1 && !pos.empty()) {
                update_decode_clusters_for_layers(begin, end, pos.front());
            }

            if (is_last_segment) {
                if (lm_head) {
                    detached_logits = *logits;
                }
            } else {
                segment_x = *x;
                has_segment_x = true;
            }
        }

        ggml_runtime::sync_kv_runtime_if_needed(m_kv_pager, kv_runtime);

        ggml_backend->m_kv->advance(batch_size);

        if (!lm_head) {
            return LogitsVector();
        }

        return LogitsVector(detached_logits.m_data, m_config->llm.vocab_size, batch_size);
    }

    Graph g(m_config->model_id);
    // input embedding
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    TensorNode *logits = nullptr;
    const bool enable_ggml_layer_profile = is_ggml_layer_profile_enabled();
    std::vector<std::pair<int, int>> layer_op_ranges;
    std::vector<int64_t> layer_time_ns;
    OpAfterExecHook prev_hook = nullptr;

#if defined(POWERSERVE_WITH_QNN)
    if (m_platform->qnn_backend) {
        auto size            = llm_config.dim;
        bool use_qnn_lm_head = m_platform->qnn_backend->m_models[m_config->model_id]->m_config.lm_heads.size() > 0;
        if (use_qnn_lm_head) {
            size   = llm_config.vocab_size;
            logits = g.qnn_forward(x, pos, mask, size, lm_head);
        } else {
            x = g.qnn_forward(x, pos, mask, size, lm_head);
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    } else
#endif
    {
        if (!lazy_load) {
            const bool use_opencl = m_platform->using_opencl(m_config->model_id);
            if (use_opencl) {
                m_platform->opencl_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
            } else {
                m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
                if (enable_ggml_layer_profile) {
                    layer_op_ranges.reserve(llm_config.n_layers);
                    layer_time_ns.assign(llm_config.n_layers, 0);
                }
            }
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                const int layer_op_begin = (!use_opencl) ? static_cast<int>(g.ops.size()) : -1;
                if (use_opencl) {
                    auto [k_cache, v_cache] = m_platform->opencl_backends[m_config->model_id]->get_cache_tensors(L);
                    auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask, true);
                    auto ffn_o = m_ffn->build(g, att_o, L);
                    x          = ffn_o;
                } else {
                    auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                    auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask, true);
                    auto ffn_o = m_ffn->build(g, att_o, L);
                    x          = ffn_o;
                }
                if (!use_opencl && enable_ggml_layer_profile) {
                    layer_op_ranges.emplace_back(layer_op_begin, static_cast<int>(g.ops.size()));
                }
            }
            // TODO: cpu and qnn reuse
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    Timer op_timer;
    size_t current_layer_idx = 0;
    bool first_profiled_op = true;
    if (enable_ggml_layer_profile && !layer_op_ranges.empty()) {
        prev_hook = get_op_after_exec_hook();
        op_timer.reset();
        set_op_after_exec_hook(
            [prev_hook, &op_timer, &layer_op_ranges, &layer_time_ns, &current_layer_idx, &first_profiled_op](
                int op_idx, const OpNode *op
            ) {
                // Ignore the first tick to avoid mixing Executor::plan() into layer-0 timing.
                if (first_profiled_op) {
                    first_profiled_op = false;
                    op_timer.reset();
                    if (prev_hook) {
                        prev_hook(op_idx, op);
                    }
                    return;
                }

                const int64_t op_time_ns = op_timer.tick_ns();
                while (current_layer_idx < layer_op_ranges.size() &&
                       op_idx >= layer_op_ranges[current_layer_idx].second) {
                    current_layer_idx++;
                }
                if (current_layer_idx < layer_op_ranges.size()) {
                    const auto [begin, end] = layer_op_ranges[current_layer_idx];
                    if (op_idx >= begin && op_idx < end) {
                        layer_time_ns[current_layer_idx] += op_time_ns;
                    }
                }
                if (prev_hook) {
                    prev_hook(op_idx, op);
                }
            }
        );
    }
    executor.run();
    if (enable_ggml_layer_profile && !layer_op_ranges.empty()) {
        set_op_after_exec_hook(prev_hook);

        double mean_layer_ms = 0.0;
        double mad_layer_ms = 0.0;
        double min_layer_ms = 1e30;
        double max_layer_ms = 0.0;
        for (size_t L = 0; L < layer_time_ns.size(); ++L) {
            const double layer_ms = layer_time_ns[L] / 1e6;
            mean_layer_ms += layer_ms;
            min_layer_ms = std::min(min_layer_ms, layer_ms);
            max_layer_ms = std::max(max_layer_ms, layer_ms);
        }
        mean_layer_ms /= std::max<size_t>(1, layer_time_ns.size());
        for (size_t L = 0; L < layer_time_ns.size(); ++L) {
            const double layer_ms = layer_time_ns[L] / 1e6;
            mad_layer_ms += std::fabs(layer_ms - mean_layer_ms);
        }
        mad_layer_ms /= std::max<size_t>(1, layer_time_ns.size());
        const double span_layer_ms = max_layer_ms - min_layer_ms;

        const bool is_decode = tokens.size() == 1;
        auto &accum = is_decode ? decode_accum : prefill_accum;
        accum.forwards += 1;
        accum.sum_mean_layer_ms += mean_layer_ms;
        accum.sum_mad_layer_ms += mad_layer_ms;
        accum.sum_span_layer_ms += span_layer_ms;

        const bool should_print_decode = is_decode && (accum.forwards % kDecodePrintInterval == 0);
        const bool should_print_prefill = !is_decode;
        if (should_print_decode || should_print_prefill) {
            const double avg_mean_layer_ms = accum.sum_mean_layer_ms / accum.forwards;
            const double avg_mad_layer_ms = accum.sum_mad_layer_ms / accum.forwards;
            const double avg_span_layer_ms = accum.sum_span_layer_ms / accum.forwards;
            POWERSERVE_LOG_INFO(
                "[GGML layer profile][{}] fwds={} cur(mean={:.3f} ms, mad={:.3f} ms, span={:.3f} ms) "
                "avg(mean={:.3f} ms, mad={:.3f} ms, span={:.3f} ms)",
                is_decode ? "decode" : "prefill",
                accum.forwards,
                mean_layer_ms,
                mad_layer_ms,
                span_layer_ms,
                avg_mean_layer_ms,
                avg_mad_layer_ms,
                avg_span_layer_ms
            );
        }
    }

    if (!lazy_load && !use_opencl && !has_qnn_backend && batch_size == 1 && !pos.empty()) {
        update_decode_clusters_for_layers(0, llm_config.n_layers, pos.front());
    }
#if defined(POWERSERVE_WITH_QNN)
    if (!m_platform->qnn_backend)
#endif
    {
        m_platform->ggml_backends[m_config->model_id]->m_kv->advance(batch_size);
    }

    if (!lm_head) {
        return LogitsVector();
    }

    // ziqian add: 增加把返回的opencl buffer转成cpu buffer的逻辑
    // If logits buffer is not CPUBuffer (e.g., OpenCLBuffer), do a D2H copy first.
    // LogitsVector currently assumes CPUBuffer.  :contentReference[oaicite:4]{index=4}
    if (dynamic_cast<CPUBuffer*>(logits->m_data.get()) == nullptr) {
        Tensor host_logits(DataType::FP32, logits->m_shape);
        host_logits.m_data = CPUBuffer::create_buffer<float>(logits->m_shape);

        // D2H (OpenCL->CPU) via backend copy() interface :contentReference[oaicite:5]{index=5}
        auto *backend = m_platform->get_backend(m_config->model_id);  // :contentReference[oaicite:6]{index=6}
        backend->copy(&host_logits, logits);

        return LogitsVector(host_logits.m_data, m_config->llm.vocab_size, batch_size);
    }

    return LogitsVector(logits->m_data, m_config->llm.vocab_size, batch_size);
    // ziqian end
}

auto Qwen2Model::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;
    for (auto logits : ret.logits_vector) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        auto next = probs.greedy_sample().token;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto Qwen2Model::generate(
    const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
) -> std::shared_ptr<TokenIterator> {
    return std::make_shared<ModelTokenIterator>(*this, tokenizer, sampler, prompt, steps, batch_size);
}

} // namespace powerserve
