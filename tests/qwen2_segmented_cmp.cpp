#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace powerserve;

static const char *DEFAULT_MODEL_DIR = "/data/local/tmp/ziqian/models/qwen2-0.5b-work/qwen2-0.5b";
static const char *DEFAULT_PROMPT =
    "In recent years, the landscape of artificial intelligence has been significantly transformed.";

static int get_env_int(const char *name, int def) {
    const char *v = std::getenv(name);
    if (!v) {
        return def;
    }
    int x = std::atoi(v);
    return x > 0 ? x : def;
}

static std::string get_env_str(const char *name, const char *def) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string(def);
}

static void set_segment_layers(int n) {
    const std::string value = std::to_string(n);
#if defined(_WIN32)
    _putenv_s("POWERSERVE_GGML_SEGMENT_LAYERS", value.c_str());
#else
    setenv("POWERSERVE_GGML_SEGMENT_LAYERS", value.c_str(), 1);
#endif
}

static bool allclose(const std::span<const float> &a, const std::span<const float> &b, float atol, float rtol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const float ai = a[i];
        const float bi = b[i];
        const float d = std::fabs(ai - bi);
        const float tol = atol + rtol * std::fabs(bi);
        if (d > tol) {
            return false;
        }
    }
    return true;
}

static int argmax(const std::span<const float> &v) {
    if (v.empty()) {
        return 0;
    }
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

int main() {
    const std::string model_dir = get_env_str("POWERSERVE_TEST_MODEL_DIR", DEFAULT_MODEL_DIR);
    const std::string prompt = get_env_str("POWERSERVE_TEST_PROMPT", DEFAULT_PROMPT);
    const int n_threads = get_env_int("POWERSERVE_TEST_THREADS", 8);
    const int segment_layers = get_env_int("POWERSERVE_TEST_SEGMENT_LAYERS", 2);
    const int decode_steps = get_env_int("POWERSERVE_TEST_DECODE_STEPS", 16);
    const float atol = 1e-6f;
    const float rtol = 1e-6f;

    POWERSERVE_LOG_INFO("==== qwen2 segmented compare (ggml full vs segmented) ====");
    POWERSERVE_LOG_INFO("model_dir={}", model_dir);
    POWERSERVE_LOG_INFO("segment_layers={} decode_steps={}", segment_layers, decode_steps);

    HyperParams hparams;
    hparams.n_threads = n_threads;
    hparams.batch_size = 1;

    auto model_full = load_model(model_dir);
    auto model_seg = load_model(model_dir);
    if (model_full->m_config->arch != "qwen2" || model_seg->m_config->arch != "qwen2") {
        POWERSERVE_LOG_ERROR("This test expects qwen2 model, got {}", model_full->m_config->arch);
        return 2;
    }

    model_full->m_attn = std::make_shared<NormAttention>(model_full->m_config->llm, model_full->m_weights);
    model_seg->m_attn = std::make_shared<NormAttention>(model_seg->m_config->llm, model_seg->m_weights);

    model_full->m_config->model_id = "ggml_full";
    model_seg->m_config->model_id = "ggml_segmented";

    auto platform = std::make_shared<Platform>();
    model_full->m_platform = platform;
    model_seg->m_platform = platform;
    platform->init_ggml_backend(model_full->m_config, hparams);
    platform->init_ggml_backend(model_seg->m_config, hparams);

    auto &id_full = model_full->m_config->model_id;
    auto &id_seg = model_seg->m_config->model_id;
    platform->reset_kv_position(id_full);
    platform->reset_kv_position(id_seg);
    platform->ggml_backends[id_full]->setup_threadpool();
    platform->ggml_backends[id_seg]->setup_threadpool();

    Tokenizer tokenizer(model_dir + "/" + MODEL_VOCAB_FILENAME);
    std::vector<Token> prompt_tokens = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    if (prompt_tokens.size() < 2) {
        POWERSERVE_LOG_ERROR("Need at least 2 prompt tokens, got {}", prompt_tokens.size());
        return 3;
    }

    for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
        std::vector<Token> in_tok{prompt_tokens[i]};
        std::vector<int> in_pos{static_cast<int>(i)};
        CausalAttentionMask mask(1);

        set_segment_layers(0);
        (void)model_full->forward(in_tok, in_pos, mask, false);
        set_segment_layers(segment_layers);
        (void)model_seg->forward(in_tok, in_pos, mask, false);
    }

    Token cur_full = prompt_tokens.back();
    Token cur_seg = prompt_tokens.back();

    for (int step = 0; step < decode_steps; ++step) {
        size_t pos_full = platform->get_kv_position(id_full);
        size_t pos_seg = platform->get_kv_position(id_seg);
        if (pos_full != pos_seg) {
            POWERSERVE_LOG_ERROR("KV pos mismatch at step {}: full={} seg={}", step, pos_full, pos_seg);
            return 4;
        }

        std::vector<Token> in_full{cur_full};
        std::vector<Token> in_seg{cur_seg};
        std::vector<int> pos{static_cast<int>(pos_full)};
        CausalAttentionMask mask(1);

        set_segment_layers(0);
        auto out_full = model_full->forward(in_full, pos, mask, true);
        set_segment_layers(segment_layers);
        auto out_seg = model_seg->forward(in_seg, pos, mask, true);

        if (out_full.logits_vector.empty() || out_seg.logits_vector.empty()) {
            POWERSERVE_LOG_ERROR("Empty logits at step {}", step);
            return 5;
        }

        const auto logits_full = out_full.logits_vector[0];
        const auto logits_seg = out_seg.logits_vector[0];
        if (!allclose(logits_full, logits_seg, atol, rtol)) {
            int best_full = argmax(logits_full);
            int best_seg = argmax(logits_seg);
            POWERSERVE_LOG_ERROR(
                "logits mismatch at step {} (argmax full={} seg={})",
                step,
                best_full,
                best_seg
            );
            return 6;
        }

        cur_full = static_cast<Token>(argmax(logits_full));
        cur_seg = static_cast<Token>(argmax(logits_seg));
        if (cur_full != cur_seg) {
            POWERSERVE_LOG_ERROR("next token mismatch at step {}: full={} seg={}", step, cur_full, cur_seg);
            return 7;
        }
    }

    platform->ggml_backends[id_full]->reset_threadpool();
    platform->ggml_backends[id_seg]->reset_threadpool();

    POWERSERVE_LOG_INFO("qwen2 segmented compare passed");
    return 0;
}
