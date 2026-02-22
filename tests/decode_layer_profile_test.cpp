#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace powerserve;

// Adjust for your local environment.
static const char *MODEL_DIR = "/data/local/tmp/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-powerserve";
static const char *PROMPT =
    "In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of "
    "large language models (LLMs). Traditionally, these models have been deployed in cloud environments due to "
    "their computational demands. However, the emergence of on-edge LLMs is reshaping how AI can be utilized at "
    "the edge of networks, offering numerous advantages in terms of latency, privacy, and accessibility.";

static int N_THREADS = 8;
static size_t BATCH_SIZE = 1;

static int argmax_span(const std::span<const float> v) {
    if (v.empty()) return 0;
    int best_i = 0;
    float best_v = v[0];
    for (int i = 1; i < (int)v.size(); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best_i = i;
        }
    }
    return best_i;
}

int main() {
    POWERSERVE_LOG_INFO("==== GGML decode-first-step layer profile test ====");
    POWERSERVE_LOG_INFO("MODEL_DIR={}", MODEL_DIR);

    HyperParams hparams;
    hparams.n_threads = N_THREADS;
    hparams.batch_size = BATCH_SIZE;

    auto model = load_model(MODEL_DIR);
    model->m_attn = std::make_shared<powerserve::NormAttention>(model->m_config->llm, model->m_weights);
    model->m_config->model_id = "ggml_decode_profile";

    auto platform = std::make_shared<Platform>();
    model->m_platform = platform;
    platform->init_ggml_backend(model->m_config, hparams);

    auto &model_id = model->m_config->model_id;
    platform->reset_kv_position(model_id);
    platform->ggml_backends[model_id]->setup_threadpool();

    std::string vocab_path = std::string(MODEL_DIR) + "/" + MODEL_VOCAB_FILENAME;
    Tokenizer tokenizer(vocab_path);
    std::vector<Token> tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (tokens.size() < 2) {
        POWERSERVE_LOG_ERROR("Prompt produced too few tokens: {}", tokens.size());
        platform->ggml_backends[model_id]->reset_threadpool();
        return 1;
    }
    POWERSERVE_LOG_INFO("Prompt token count: {}", tokens.size());

    // Prepare KV cache with prompt[:-1] so next single-token forward is a true decode step.
    std::vector<Token> prefill_tokens(tokens.begin(), tokens.end() - 1);
    std::vector<int> prefill_pos(prefill_tokens.size());
    for (size_t i = 0; i < prefill_pos.size(); ++i) {
        prefill_pos[i] = (int)i;
    }
    auto prefill_mask = CausalAttentionMask(prefill_tokens.size());

    POWERSERVE_LOG_INFO("Running prefill (lm_head=false) ...");
    (void)model->forward(prefill_tokens, prefill_pos, prefill_mask, false);

    const size_t decode_pos = platform->get_kv_position(model_id);
    std::vector<Token> decode_tokens(1, tokens.back());
    std::vector<int> decode_pos_v(1, (int)decode_pos);
    auto decode_mask = CausalAttentionMask(1);

    POWERSERVE_LOG_INFO("Running decode step-0 (pos={}, lm_head=true) ...", decode_pos);
    auto decode_ret = model->forward(decode_tokens, decode_pos_v, decode_mask, true);
    if (decode_ret.logits_vector.empty()) {
        POWERSERVE_LOG_ERROR("Decode returned empty logits");
        platform->ggml_backends[model_id]->reset_threadpool();
        return 2;
    }

    auto logits = decode_ret.logits_vector.back();
    int next_token = argmax_span(logits);
    POWERSERVE_LOG_INFO("Decode step-0 argmax token: {}", next_token);

    platform->ggml_backends[model_id]->reset_threadpool();
    return 0;
}
