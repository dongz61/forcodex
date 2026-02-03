// tests/tokens_cmp.cpp
#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"
#include "executor/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

using namespace powerserve;

// ======================
// CONFIG (adjust if needed)
// ======================
static const char *MODEL_DIR = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT    = "你好，请介绍你自己";

// threading / batch
static int    N_THREADS  = 8;
static size_t BATCH_SIZE = 4;

// compare controls
static int   TOPK                = 10;     // top-k for behavior equivalence
static int   DECODE_STEPS        = 32;     // how many free-run steps
static bool  STRICT_ARGMAX       = true;   // if true: require argmax equal; else allow opencl argmax in ggml top-k
static bool  REQUIRE_TOPK_ORDER  = false;  // if true: require exact top-k ordering match (stronger than argmax)
static bool  USE_TRUNCATED_SOFTMAX = true; // compute distribution distance on union(topk_ref, topk_test)

// distribution distance thresholds (tune these by running a few times)
static float KL_MAX        = 1e-3f;  // KL(P_ref || P_test)
static float COS_MIN       = 0.999f; // cosine similarity between probability vectors
static float EPS_PROB      = 1e-12f; // epsilon to avoid log(0)

// ======================
// Helpers: top-k / softmax / metrics
// ======================
static int argmax_span(std::span<const float> v) {
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

static std::vector<int> topk_ids(std::span<const float> logits, int k) {
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) idx[i] = i;

    int kk = std::min(k, (int)idx.size());
    std::partial_sort(
        idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; }
    );
    idx.resize(kk);
    return idx;
}

static void dump_topk(std::span<const float> logits, int k, const char *tag) {
    auto ids = topk_ids(logits, k);
    fmt::print("[{} top{}] ", tag, (int)ids.size());
    for (int id : ids) {
        fmt::print("({} {:.6f}) ", id, logits[id]);
    }
    fmt::print("\n");
}

// Stable softmax over selected indices.
// Returns probabilities aligned with `ids` order.
static std::vector<float> softmax_selected(std::span<const float> logits, const std::vector<int> &ids) {
    std::vector<float> p;
    p.resize(ids.size(), 0.f);
    if (ids.empty()) return p;

    // find max logit among selected
    float m = -std::numeric_limits<float>::infinity();
    for (int id : ids) {
        float x = logits[id];
        if (x > m) m = x;
    }

    // exp and sum
    double sum = 0.0;
    for (size_t i = 0; i < ids.size(); ++i) {
        double e = std::exp((double)logits[ids[i]] - (double)m);
        p[i] = (float)e;
        sum += e;
    }
    if (sum <= 0.0) {
        // fallback: uniform
        float u = 1.0f / (float)ids.size();
        std::fill(p.begin(), p.end(), u);
        return p;
    }

    // normalize
    for (float &x : p) x = (float)((double)x / sum);
    return p;
}

static float kl_div(std::span<const float> p, std::span<const float> q, float eps) {
    // KL(p||q) = sum p * log(p/q)
    if (p.size() != q.size()) return std::numeric_limits<float>::infinity();
    double acc = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        double pi = std::max((double)p[i], (double)eps);
        double qi = std::max((double)q[i], (double)eps);
        acc += pi * std::log(pi / qi);
    }
    return (float)acc;
}

static float cosine_sim(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size() || a.empty()) return 0.f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = (double)a[i];
        double y = (double)b[i];
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    if (na <= 0.0 || nb <= 0.0) return 0.f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

static void abs_diff_stats(std::span<const float> a, std::span<const float> b,
                           float *max_abs, float *mean_abs) {
    if (!max_abs || !mean_abs) return;
    *max_abs = 0.f;
    *mean_abs = 0.f;
    if (a.size() != b.size() || a.empty()) return;

    double sum = 0.0;
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        sum += d;
        if (d > mx) mx = d;
    }
    *max_abs = (float)mx;
    *mean_abs = (float)(sum / (double)a.size());
}

static bool ids_equal(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

static bool id_in_set(int x, const std::vector<int> &ids) {
    for (int id : ids) if (id == x) return true;
    return false;
}

// Build union(topk_ref, topk_test) while keeping ref order first, then new ones from test.
static std::vector<int> union_topk(const std::vector<int> &ref, const std::vector<int> &test) {
    std::vector<int> out;
    out.reserve(ref.size() + test.size());
    std::unordered_set<int> seen;
    for (int id : ref) {
        if (seen.insert(id).second) out.push_back(id);
    }
    for (int id : test) {
        if (seen.insert(id).second) out.push_back(id);
    }
    return out;
}

// Hybrid comparator: (1) argmax/topk behavior, (3) distribution distance on (union) topk
struct CompareResult {
    bool ok = false;
    int  argmax_ref = -1;
    int  argmax_test = -1;
    float kl = 0.f;
    float cosine = 0.f;
    float max_abs = 0.f;
    float mean_abs = 0.f;
    std::vector<int> topk_ref;
    std::vector<int> topk_test;
    std::vector<int> dist_ids; // ids used for distribution metrics
};

static CompareResult compare_logits_hybrid(std::span<const float> ref_logits,
                                          std::span<const float> test_logits) {
    CompareResult r;
    r.argmax_ref  = argmax_span(ref_logits);
    r.argmax_test = argmax_span(test_logits);

    r.topk_ref  = topk_ids(ref_logits, TOPK);
    r.topk_test = topk_ids(test_logits, TOPK);

    // behavior gate
    bool behavior_ok = true;
    if (REQUIRE_TOPK_ORDER) {
        behavior_ok = ids_equal(r.topk_ref, r.topk_test);
    } else if (STRICT_ARGMAX) {
        behavior_ok = (r.argmax_ref == r.argmax_test);
    } else {
        // allow test argmax inside ref topk
        behavior_ok = id_in_set(r.argmax_test, r.topk_ref);
    }

    // stats (full logits abs diff; not a hard gate)
    abs_diff_stats(ref_logits, test_logits, &r.max_abs, &r.mean_abs);

    // distribution metric
    if (USE_TRUNCATED_SOFTMAX) {
        r.dist_ids = union_topk(r.topk_ref, r.topk_test);
    } else {
        // full-vocab distribution is expensive; keep truncated by default
        r.dist_ids = r.topk_ref;
    }

    auto p = softmax_selected(ref_logits, r.dist_ids);
    auto q = softmax_selected(test_logits, r.dist_ids);

    r.kl = kl_div(p, q, EPS_PROB);
    r.cosine = cosine_sim(p, q);

    bool dist_ok = (r.kl <= KL_MAX) && (r.cosine >= COS_MIN);

    r.ok = behavior_ok && dist_ok;
    return r;
}

static void dump_cmp_failure(const char *tag,
                             int step,
                             size_t pos_ref,
                             size_t pos_test,
                             int token_in_ref,
                             int token_in_test,
                             const CompareResult &cr,
                             std::span<const float> ref_logits,
                             std::span<const float> test_logits) {
    fmt::print("\n[{} FAIL] step={} pos_ref={} pos_test={} token_in_ref={} token_in_test={}\n",
               tag, step, pos_ref, pos_test, token_in_ref, token_in_test);

    fmt::print("  argmax_ref={} argmax_test={}\n", cr.argmax_ref, cr.argmax_test);
    fmt::print("  metrics: KL={:.6g} (<= {:.6g}), cosine={:.6g} (>= {:.6g}), max_abs={:.6g}, mean_abs={:.6g}\n",
               cr.kl, KL_MAX, cr.cosine, COS_MIN, cr.max_abs, cr.mean_abs);

    dump_topk(ref_logits, TOPK, "ggml");
    dump_topk(test_logits, TOPK, "opencl");

    if (!cr.dist_ids.empty()) {
        fmt::print("  dist_ids ({}): ", (int)cr.dist_ids.size());
        for (int id : cr.dist_ids) fmt::print("{} ", id);
        fmt::print("\n");
    }
}

// ======================
// Main
// ======================
int main() {
    POWERSERVE_LOG_INFO("==== tokens_cmp (ggml vs opencl) ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("THREADS={}, BATCH_SIZE={}", N_THREADS, BATCH_SIZE);
    POWERSERVE_LOG_INFO("TOPK={}, DECODE_STEPS={}, STRICT_ARGMAX={}, REQUIRE_TOPK_ORDER={}, TRUNC_SOFTMAX={}",
                        TOPK, DECODE_STEPS, (int)STRICT_ARGMAX, (int)REQUIRE_TOPK_ORDER, (int)USE_TRUNCATED_SOFTMAX);
    POWERSERVE_LOG_INFO("THRESH: KL_MAX={}, COS_MIN={}", KL_MAX, COS_MIN);

    HyperParams hparams;
    hparams.n_threads  = N_THREADS;
    hparams.batch_size = BATCH_SIZE;

    auto model_ggml = load_model(MODEL_DIR);
    auto model_ocl  = load_model(MODEL_DIR);

    model_ggml->m_attn = std::make_shared<powerserve::NormAttention>(model_ggml->m_config->llm, model_ggml->m_weights);
    model_ocl->m_attn  = std::make_shared<powerserve::NormAttention>(model_ocl->m_config->llm,  model_ocl->m_weights);

    model_ggml->m_config->model_id = "ggml_ref";
    model_ocl->m_config->model_id  = "opencl_test";

    auto platform = std::make_shared<Platform>();
    model_ggml->m_platform = platform;
    model_ocl->m_platform  = platform;

    platform->init_ggml_backend(model_ggml->m_config, hparams);
    platform->init_ggml_backend(model_ocl->m_config,  hparams);
    platform->init_opencl_backend(model_ocl->m_config, hparams);

    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        platform->ggml_backends[id_g]->setup_threadpool();
        platform->ggml_backends[id_o]->setup_threadpool();
    }

    std::string vocab_path = std::string(MODEL_DIR) + "/" + MODEL_VOCAB_FILENAME;
    Tokenizer tokenizer(vocab_path);

    std::vector<Token> prompt_tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (prompt_tokens.empty()) {
        POWERSERVE_LOG_ERROR("Prompt tokenization returned empty tokens.");
        return 1;
    }

    std::vector<int> prompt_pos(prompt_tokens.size());
    for (size_t i = 0; i < prompt_tokens.size(); ++i) prompt_pos[i] = (int)i;

    POWERSERVE_LOG_INFO("Prompt token count = {}", prompt_tokens.size());

    auto forward_both =
        [&](const std::vector<Token> &in_tokens,
            const std::vector<int>   &in_pos,
            const CausalAttentionMask &in_mask,
            bool lm_head) -> std::pair<LogitsVector, LogitsVector> {

            auto ret_g = model_ggml->forward(in_tokens, in_pos, in_mask, lm_head);
            auto ret_o = model_ocl->forward(in_tokens, in_pos, in_mask, lm_head);
            return {ret_g, ret_o};
        };

    // =========================
    // 1) PREFILL behavior check
    // =========================
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        auto mask = CausalAttentionMask(prompt_tokens.size());
        auto [ret_g, ret_o] = forward_both(prompt_tokens, prompt_pos, mask, /*lm_head*/true);

        auto lg = ret_g.logits_vector.back();
        auto lo = ret_o.logits_vector.back();

        auto cr = compare_logits_hybrid(lg, lo);
        if (!cr.ok) {
            dump_cmp_failure("PREFILL", /*step*/-1,
                             platform->get_kv_position(id_g),
                             platform->get_kv_position(id_o),
                             /*token_in_ref*/-1, /*token_in_test*/-1,
                             cr, lg, lo);
            return 1;
        }

        fmt::print("[PREFILL PASS] argmax_ref={} argmax_test={} KL={:.6g} cosine={:.6g} max_abs={:.6g} mean_abs={:.6g}\n",
                   cr.argmax_ref, cr.argmax_test, cr.kl, cr.cosine, cr.max_abs, cr.mean_abs);
    }

    // =========================
    // 2) DECODE free-run
    // =========================
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        // prefill prompt[:-1] with lm_head=false (same as your logits_cmp decode setup)
        if (prompt_tokens.size() > 1) {
            std::vector<Token> prefill_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
            std::vector<int> prefill_pos(prefill_tokens.size());
            for (size_t i = 0; i < prefill_pos.size(); ++i) prefill_pos[i] = (int)i;

            auto prefill_mask = CausalAttentionMask(prefill_tokens.size());
            (void)forward_both(prefill_tokens, prefill_pos, prefill_mask, /*lm_head*/false);
        }

        int token_in_g = prompt_tokens.back();
        int token_in_o = prompt_tokens.back();

        for (int step = 0; step < DECODE_STEPS; ++step) {
            size_t pos_g = platform->get_kv_position(id_g);
            size_t pos_o = platform->get_kv_position(id_o);

            std::vector<Token> step_tok_g(1, token_in_g);
            std::vector<Token> step_tok_o(1, token_in_o);

            std::vector<int> step_pos_g(1, (int)pos_g);
            std::vector<int> step_pos_o(1, (int)pos_o);

            auto step_mask = CausalAttentionMask(1);

            // forward separately (because tokens differ in free-run)
            auto ret_g = model_ggml->forward(step_tok_g, step_pos_g, step_mask, /*lm_head*/true);
            auto ret_o = model_ocl->forward(step_tok_o, step_pos_o, step_mask, /*lm_head*/true);

            auto lg = ret_g.logits_vector.back();
            auto lo = ret_o.logits_vector.back();

            auto cr = compare_logits_hybrid(lg, lo);

            int next_g = cr.argmax_ref;
            int next_o = cr.argmax_test;

            // First: behavior equivalence check (token decision)
            bool token_ok = true;
            if (STRICT_ARGMAX || REQUIRE_TOPK_ORDER) {
                token_ok = (next_g == next_o);
            } else {
                // relaxed mode: allow opencl argmax inside ggml top-k
                token_ok = id_in_set(next_o, cr.topk_ref);
            }

            if (!token_ok || !cr.ok) {
                dump_cmp_failure("DECODE", step, pos_g, pos_o, token_in_g, token_in_o, cr, lg, lo);
                fmt::print("  next_token_ref={} next_token_test={}\n", next_g, next_o);
                return 1;
            }

            fmt::print("[decode step {:3d}] pos_g={} pos_o={} token_in_g={} token_in_o={} -> next={}\n",
                       step, pos_g, pos_o, token_in_g, token_in_o, next_g);

            // free-run update
            token_in_g = next_g;
            token_in_o = next_o;
        }

        POWERSERVE_LOG_INFO("Decode free-run tokens_cmp PASS ({} steps)", DECODE_STEPS);
    }

    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
