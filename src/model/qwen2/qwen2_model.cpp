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
#include "core/logger.hpp"
#include "core/perfetto_trace.hpp"
#include "core/timer.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/qwen2/qwen2_weight.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace powerserve {

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

auto Qwen2Model::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> LogitsVector {
    Graph g(m_config->model_id);
    // input embedding
    size_t batch_size  = tokens.size();
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    TensorNode *logits = nullptr;

    auto &llm_config = m_config->llm;
    bool enable_ggml_layer_profile = false;
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
                enable_ggml_layer_profile = true;
                layer_op_ranges.reserve(llm_config.n_layers);
                layer_time_ns.assign(llm_config.n_layers, 0);
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
                if (!use_opencl) {
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
    if (enable_ggml_layer_profile && !layer_op_ranges.empty()) {
        prev_hook = get_op_after_exec_hook();
        op_timer.reset();
        set_op_after_exec_hook(
            [prev_hook, &op_timer, &layer_op_ranges, &layer_time_ns, &current_layer_idx](int op_idx, const OpNode *op) {
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
        int64_t layers_total_ns = 0;
        for (size_t L = 0; L < layer_time_ns.size(); L++) {
            layers_total_ns += layer_time_ns[L];
            POWERSERVE_LOG_INFO("[GGML layer profile] layer {}: {:.3f} ms", L, layer_time_ns[L] / 1e6);
        }
        POWERSERVE_LOG_INFO("[GGML layer profile] layers total: {:.3f} ms", layers_total_ns / 1e6);
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
