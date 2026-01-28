// tests/logits_cmp.cpp
#include "backend/platform.hpp"
#include "core/logger.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "tokenizer/tokenizer.hpp"
#include "graph/op_type.hpp"
#include "core/tensor.hpp"
#include "executor/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>

using namespace powerserve;

// ======================
// CONFIG (adjust if needed)
// ======================
static const char *MODEL_DIR   = "/home/intern/ziqian/models/qwen2-0.5b-work/qwen2-0.5b-gguf";
static const char *PROMPT      = "你好，请介绍你自己";

static int    N_THREADS   = 8;
static size_t BATCH_SIZE  = 1;

static float ATOL  = 1e-6f;
static float RTOL  = 1e-6f;
static int   TOPK  = 10;
static int   DECODE_STEPS = 16;

// last-writer dump depth for a given cl_mem
static int   LAST_WRITER_RECENT_N = 24;

// ======================
// Helpers
// ======================
static const char *op_type_to_string(OpType t) {
    switch (t) {
    case OpType::GET_EMBEDDING:   return "GET_EMBEDDING";
    case OpType::ADD:             return "ADD";
    case OpType::MAT_MUL:         return "MAT_MUL";
    case OpType::RMS_NORM:        return "RMS_NORM";
    case OpType::SILU_HADAMARD:   return "SILU_HADAMARD";
    case OpType::ROPE:            return "ROPE";
    case OpType::SOFTMAX:         return "SOFTMAX";
    case OpType::COPY:            return "COPY";
    case OpType::ADD_CACHE:       return "ADD_CACHE";
    case OpType::PERMUTE:         return "PERMUTE";
    case OpType::CONT:            return "CONT";
    case OpType::VIEW:            return "VIEW";
    case OpType::SOFTMAX_EXT:     return "SOFTMAX_EXT";
    case OpType::GET_MASK:        return "GET_MASK";
    case OpType::TRANSPOSE:       return "TRANSPOSE";
    case OpType::PRINT:           return "PRINT";
    default:                      return "UNKNOWN";
    }
}

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

static inline bool allclose_span(std::span<const float> a, std::span<const float> b,
                                 float atol, float rtol,
                                 size_t *bad_i=nullptr, float *diff=nullptr) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        float ai = a[i];
        float bi = b[i];

        if (ai == bi) continue;

        if (std::isnan(ai) || std::isnan(bi)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = std::numeric_limits<float>::quiet_NaN();
            return false;
        }

        if (std::isinf(ai) || std::isinf(bi)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = std::numeric_limits<float>::infinity();
            return false;
        }

        float d   = std::fabs(ai - bi);
        float tol = atol + rtol * std::fabs(bi);
        if (!(d <= tol)) {
            if (bad_i) *bad_i = i;
            if (diff)  *diff  = d;
            return false;
        }
    }
    return true;
}

static void dump_topk(std::span<const float> logits, int k, const char *tag) {
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) idx[i] = i;

    std::partial_sort(
        idx.begin(), idx.begin() + std::min(k, (int)idx.size()), idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; }
    );

    fmt::print("[{} top{}] ", tag, k);
    for (int i = 0; i < k && i < (int)idx.size(); ++i) {
        int id = idx[i];
        fmt::print("({} {:.6f}) ", id, logits[id]);
    }
    fmt::print("\n");
}

static inline uint64_t op_out_key(int op_idx, int out_idx) {
    return (uint64_t(uint32_t(op_idx)) << 32) | uint32_t(out_idx);
}

// ======================
// Tensor -> FP32 vector (CPU/OpenCL)
// ======================
static std::vector<float> tensor_to_f32_vec_cpu(const Tensor *t) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32) return out;
    if (!t->m_data) return out;

    auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get());
    if (!cb) return {};

    auto shape  = t->m_shape;
    auto stride = cb->m_stride;

    size_t n = t->n_elements();
    out.resize(n);

    size_t idx = 0;
    for (size_t i3 = 0; i3 < shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < shape[0]; ++i0) {
                    float *ptr = (float *)((char *)cb->m_data +
                                           i3 * stride[3] + i2 * stride[2] +
                                           i1 * stride[1] + i0 * stride[0]);
                    out[idx++] = *ptr;
                }
            }
        }
    }
    return out;
}

static std::vector<float> tensor_to_f32_vec_opencl(const Tensor *t, powerserve::opencl::OpenCLBackend *cl_backend) {
    std::vector<float> out;
    if (!t || t->m_dtype != DataType::FP32) return out;
    if (!t->m_data) return out;
    POWERSERVE_ASSERT(cl_backend);

    Tensor tmp_cpu(DataType::FP32, t->m_shape);
    tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(t->m_shape);
    cl_backend->copy(&tmp_cpu, t);
    return tensor_to_f32_vec_cpu(&tmp_cpu);
}

static std::vector<float> tensor_to_f32_vec_any(const Tensor *t,
                                                powerserve::opencl::OpenCLBackend *cl_backend) {
    if (!t || t->m_dtype != DataType::FP32) return {};
    if (!t->m_data) return {};

    if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
        return tensor_to_f32_vec_opencl(t, cl_backend);
    }
    if (dynamic_cast<CPUBuffer*>(t->m_data.get())) {
        return tensor_to_f32_vec_cpu(t);
    }
    return {};
}

// ======================
// Meta printing (minimal, includes OpenCL view identity)
// ======================
static void print_tensor_meta(const Tensor *t, const char *tag) {
    if (!t) {
        fmt::print("  {}: <null>\n", tag);
        return;
    }
    auto s = t->m_shape;

    if (!t->m_data) {
        fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] [NO_DATA]\n",
                   tag, (int)t->m_dtype, s[0], s[1], s[2], s[3]);
        return;
    }

    if (auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get())) {
        // NOTE: these names match the debug style you used earlier (device_buffer/base_offset/stride/m_size).
        fmt::print(
            "  {}: dtype={} shape=[{}, {}, {}, {}] strideB=[{}, {}, {}, {}] [OpenCL] dev={} base_off={} size={}\n",
            tag, (int)t->m_dtype,
            s[0], s[1], s[2], s[3],
            clb->m_stride[0], clb->m_stride[1], clb->m_stride[2], clb->m_stride[3],
            (void*)clb->get_device_buffer(),
            (size_t)clb->get_base_offset(),
            (size_t)clb->m_size
        );
        return;
    }

    if (auto *cb = dynamic_cast<CPUBuffer*>(t->m_data.get())) {
        fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] strideB=[{}, {}, {}, {}] [CPU]\n",
                   tag, (int)t->m_dtype,
                   s[0], s[1], s[2], s[3],
                   cb->m_stride[0], cb->m_stride[1], cb->m_stride[2], cb->m_stride[3]);
        return;
    }

    fmt::print("  {}: dtype={} shape=[{}, {}, {}, {}] [UNKNOWN_BUFFER]\n",
               tag, (int)t->m_dtype, s[0], s[1], s[2], s[3]);
}

// ======================
// OpenCL range last-writer tracking (key for VIEW root-cause)
// ======================
struct ClRange {
    void  *dev  = nullptr; // cl_mem identity
    size_t off  = 0;       // base offset (bytes)
    size_t size = 0;       // size (bytes)
    bool   ok   = false;
};

static inline bool ranges_overlap(size_t a_off, size_t a_sz, size_t b_off, size_t b_sz) {
    if (a_sz == 0 || b_sz == 0) return false;
    size_t a_end = a_off + a_sz;
    size_t b_end = b_off + b_sz;
    return (a_off < b_end) && (b_off < a_end);
}

static ClRange get_cl_range(const Tensor *t) {
    ClRange r;
    if (!t || !t->m_data) return r;
    auto *clb = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(t->m_data.get());
    if (!clb) return r;

    r.dev  = (void*)clb->get_device_buffer();
    r.off  = (size_t)clb->get_base_offset();
    r.size = (size_t)clb->m_size;
    r.ok   = (r.dev != nullptr && r.size > 0);
    return r;
}

struct WriteRec {
    int   op_idx  = -1;
    int   out_idx = -1;
    OpType op     = OpType::PRINT;
    size_t off    = 0;
    size_t size   = 0;
};

static void dump_recent_writes(void *dev,
                               const std::unordered_map<void*, std::vector<WriteRec>> &cl_writes,
                               int limit_n) {
    auto it = cl_writes.find(dev);
    if (it == cl_writes.end() || it->second.empty()) {
        fmt::print("  [CL-WRITE] dev={} <no writes recorded>\n", dev);
        return;
    }

    auto &v = it->second;
    int n = (int)v.size();
    int start = std::max(0, n - std::max(1, limit_n));

    fmt::print("  [CL-WRITE] dev={} recent {} writes (total={}):\n", dev, n - start, n);
    for (int i = start; i < n; ++i) {
        auto &wr = v[i];
        fmt::print("    - op#{:4d} type={:<12} out#{} off={} size={}\n",
                   wr.op_idx, op_type_to_string(wr.op), wr.out_idx, wr.off, wr.size);
    }
}

static int find_last_writer_for_range(void *dev,
                                      size_t q_off,
                                      size_t q_sz,
                                      int cur_op_idx,
                                      const std::unordered_map<void*, std::vector<WriteRec>> &cl_writes,
                                      WriteRec *out_wr=nullptr) {
    if (out_wr) *out_wr = {};
    auto it = cl_writes.find(dev);
    if (it == cl_writes.end()) return -1;
    auto &v = it->second;

    for (int i = (int)v.size() - 1; i >= 0; --i) {
        const auto &wr = v[i];
        if (wr.op_idx >= cur_op_idx) continue;
        if (ranges_overlap(q_off, q_sz, wr.off, wr.size)) {
            if (out_wr) *out_wr = wr;
            return wr.op_idx;
        }
    }
    return -1;
}

int main() {
    POWERSERVE_LOG_INFO("==== Qwen2 logits compare test (ggml vs opencl) ====");
    POWERSERVE_LOG_INFO("PROMPT={}", PROMPT);
    POWERSERVE_LOG_INFO("THREADS={}, BATCH_SIZE={}", N_THREADS, BATCH_SIZE);

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

    std::vector<Token> tokens = tokenizer.tokenize(PROMPT, tokenizer.m_vocab.tokenizer_add_bos);
    if (tokens.empty()) {
        POWERSERVE_LOG_ERROR("Prompt tokenization returned empty tokens.");
        return 1;
    }

    std::vector<int> pos(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) pos[i] = (int)i;

    POWERSERVE_LOG_INFO("Prompt token count = {}", tokens.size());

    // =========================
    // Run one forward: per-op compare (GGML vs OpenCL)
    // =========================
    auto run_forward_per_op_cmp =
        [&](const std::vector<Token> &in_tokens,
            const std::vector<int>   &in_pos,
            const CausalAttentionMask &in_mask,
            bool lm_head,
            const char *phase_tag) -> std::pair<LogitsVector, LogitsVector> {

            // cache ggml FP32 outputs: (op_idx, out_idx)->vec
            std::unordered_map<uint64_t, std::vector<float>> ggml_op_outs;

            // ---- GGML pass hook ----
            set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
                for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                    Tensor *out = op->next[oi]->tensor();
                    if (!out || !out->m_data) continue;
                    if (out->m_dtype != DataType::FP32) continue;
                    ggml_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_cpu(out);
                }
            });

            auto ret_g = model_ggml->forward(in_tokens, in_pos, in_mask, lm_head);
            set_op_after_exec_hook(nullptr);

            // ---- OpenCL pass ----
            auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
                platform->get_backend(model_ocl->m_config->model_id));
            POWERSERVE_ASSERT(cl_backend && "OpenCLBackend is null");

            // cache opencl FP32 outputs: for backtrace dump
            std::unordered_map<uint64_t, std::vector<float>> ocl_op_outs;
            std::unordered_map<int, const OpNode*> ocl_op_nodes;

            // producer map: Tensor* -> op_idx
            std::unordered_map<const Tensor*, int> producer;

            // (cl_mem -> ordered writes)
            std::unordered_map<void*, std::vector<WriteRec>> cl_writes;

            auto record_op_writes = [&](int op_idx, const OpNode *op) {
                if (!op) return;
                // NOTE: 为了抓“写偏/没写”根因，这里采取“保守记录”：只要 op 有 OpenCL 输出，就当作一次写记录。
                // 如果某些 op 实际是 alias/view 不写（比如 VIEW/CONT），它们也会出现记录，但不会影响“最后写者”定位太多。
                // 如果你后续想更精准，可以把 VIEW/CONT 过滤掉。
                for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                    const Tensor *out = op->next[oi] ? op->next[oi]->tensor() : nullptr;
                    if (!out || !out->m_data) continue;
                    auto r = get_cl_range(out);
                    if (!r.ok) continue;
                    cl_writes[r.dev].push_back(WriteRec{op_idx, oi, op->op, r.off, r.size});
                }
            };

            // Try to find a likely consumer index for VIEW output (best-effort)
            auto find_consumer_of_view_output = [&](int view_op_idx, const OpNode *view_op) -> int {
                if (!view_op || view_op->op != OpType::VIEW) return -1;
                if (view_op->next.empty() || !view_op->next[0]) return -1;
                const Tensor *view_out = view_op->next[0]->tensor();
                if (!view_out) return -1;

                // brute-force among known op nodes (already executed so far): find first op that uses it as input
                // (only gives you "consumer already executed" which is useful for order bugs)
                for (const auto &kv : ocl_op_nodes) {
                    int op_idx = kv.first;
                    if (op_idx <= view_op_idx) continue;
                    const OpNode *node = kv.second;
                    if (!node) continue;
                    for (auto *p : node->prev) {
                        const Tensor *t = p ? p->tensor() : nullptr;
                        if (t == view_out) return op_idx;
                    }
                }
                return -1;
            };

            auto dump_view_root_cause =
                [&](int op_idx, const OpNode *op) {
                    if (!op || op->op != OpType::VIEW) return;

                    const Tensor *parent = (op->prev.size() > 0 && op->prev[0]) ? op->prev[0]->tensor() : nullptr;
                    const Tensor *view   = (op->next.size() > 0 && op->next[0]) ? op->next[0]->tensor() : nullptr;

                    fmt::print("\n[VIEW-ROOTCAUSE] op#{} VIEW\n", op_idx);
                    print_tensor_meta(parent, "parent(in0)");
                    print_tensor_meta(view,   "view(out0)");

                    // parent/view must alias same dev; if not, it’s already suspicious.
                    ClRange pr = get_cl_range(parent);
                    ClRange vr = get_cl_range(view);

                    if (!pr.ok || !vr.ok) {
                        fmt::print("  [VIEW-ROOTCAUSE] parent/view cl_range invalid (not OpenCLBuffer?)\n");
                        return;
                    }

                    fmt::print("  [VIEW-ROOTCAUSE] alias_same_dev={} (parent_dev={} view_dev={})\n",
                               (pr.dev == vr.dev), pr.dev, vr.dev);

                    // last-writer of *parent range* before this VIEW
                    WriteRec wr{};
                    int lw = find_last_writer_for_range(pr.dev, pr.off, pr.size, op_idx, cl_writes, &wr);
                    if (lw < 0) {
                        fmt::print("  [VIEW-ROOTCAUSE] last_writer(parent_range) = <none>\n");
                    } else {
                        fmt::print("  [VIEW-ROOTCAUSE] last_writer(parent_range) = op#{} type={} out#{} off={} size={}\n",
                                   wr.op_idx, op_type_to_string(wr.op), wr.out_idx, wr.off, wr.size);
                    }

                    // last-writer of *view range* before this VIEW (usually same as parent, but view.size may be smaller)
                    WriteRec wr2{};
                    int lw2 = find_last_writer_for_range(vr.dev, vr.off, vr.size, op_idx, cl_writes, &wr2);
                    if (lw2 < 0) {
                        fmt::print("  [VIEW-ROOTCAUSE] last_writer(view_range)   = <none>\n");
                    } else {
                        fmt::print("  [VIEW-ROOTCAUSE] last_writer(view_range)   = op#{} type={} out#{} off={} size={}\n",
                                   wr2.op_idx, op_type_to_string(wr2.op), wr2.out_idx, wr2.off, wr2.size);
                    }

                    // dump recent writes to this dev buffer (helps see “writes all go to other offsets”)
                    dump_recent_writes(pr.dev, cl_writes, LAST_WRITER_RECENT_N);

                    // quick order hint: has consumer already executed?
                    int consumer = find_consumer_of_view_output(op_idx, op);
                    if (consumer >= 0) {
                        fmt::print("  [VIEW-ROOTCAUSE] consumer_hint: first consumer after VIEW appears at op#{}\n", consumer);
                    } else {
                        fmt::print("  [VIEW-ROOTCAUSE] consumer_hint: not found among executed ops (may appear later)\n");
                    }
                };

            auto dump_mismatch_detail =
                [&](int op_idx, const OpNode *op, int out_idx,
                    const Tensor *out,
                    const std::vector<float> &gg_vec,
                    const std::vector<float> &ocl_vec,
                    size_t bad_i, float diff) {

                    fmt::print("\n[MISMATCH][{}] op#{} type={} out#{}\n",
                               phase_tag, op_idx, op_type_to_string(op->op), out_idx);
                    fmt::print("  bad_i={} ggml={} ocl={} diff={}\n",
                               bad_i, gg_vec[bad_i], ocl_vec[bad_i], diff);

                    print_tensor_meta(out, "out");
                    for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                        Tensor *in = op->prev[pi] ? op->prev[pi]->tensor() : nullptr;
                        char name[64];
                        std::snprintf(name, sizeof(name), "in[%d]", pi);
                        print_tensor_meta(in, name);
                    }

                    // If this is VIEW mismatch, dump root-cause info immediately
                    if (op->op == OpType::VIEW) {
                        dump_view_root_cause(op_idx, op);
                    } else {
                        // For non-VIEW mismatch, if any input has no producer, show last-writer for that input range too
                        for (int pi = 0; pi < (int)op->prev.size(); ++pi) {
                            const Tensor *t = op->prev[pi] ? op->prev[pi]->tensor() : nullptr;
                            if (!t) continue;

                            auto itp = producer.find(t);
                            if (itp != producer.end()) continue; // normal case

                            ClRange r = get_cl_range(t);
                            if (!r.ok) continue;

                            WriteRec wr{};
                            int lw = find_last_writer_for_range(r.dev, r.off, r.size, op_idx, cl_writes, &wr);

                            fmt::print("\n  [INPUT-NOPRODUCER] in[{}] has no producer_op, try last-writer by cl_range:\n", pi);
                            print_tensor_meta(t, "  in(no_prod)");
                            if (lw < 0) {
                                fmt::print("    last_writer = <none>\n");
                            } else {
                                fmt::print("    last_writer = op#{} type={} out#{} off={} size={}\n",
                                           wr.op_idx, op_type_to_string(wr.op), wr.out_idx, wr.off, wr.size);
                            }
                            dump_recent_writes(r.dev, cl_writes, LAST_WRITER_RECENT_N);
                        }
                    }
                };

            // Simple backtrace: follow producer chain using first traceable input
            auto backtrace_chain = [&](int start_op_idx) {
                fmt::print("\n========== Backtrace (producer chain) [{}] ==========\n", phase_tag);

                int cur = start_op_idx;
                std::unordered_set<int> seen;

                while (true) {
                    if (seen.count(cur)) {
                        fmt::print("  [BT] loop detected at op#{}\n", cur);
                        break;
                    }
                    seen.insert(cur);

                    auto itop = ocl_op_nodes.find(cur);
                    const OpNode *cur_op = (itop == ocl_op_nodes.end()) ? nullptr : itop->second;
                    if (!cur_op) {
                        fmt::print("  [BT] op#{} missing node, stop.\n", cur);
                        break;
                    }

                    int p = -1;
                    int prev_i = -1;
                    for (int i = 0; i < (int)cur_op->prev.size(); ++i) {
                        const Tensor *t = cur_op->prev[i] ? cur_op->prev[i]->tensor() : nullptr;
                        if (!t) continue;
                        auto itp = producer.find(t);
                        if (itp != producer.end()) {
                            p = itp->second;
                            prev_i = i;
                            break;
                        }
                    }

                    if (p < 0) {
                        fmt::print("  [BT] op#{} type={} no producer-traceable input, stop.\n",
                                   cur, op_type_to_string(cur_op->op));
                        break;
                    }

                    auto itg = ggml_op_outs.find(op_out_key(p, 0));
                    auto ito = ocl_op_outs.find(op_out_key(p, 0));
                    auto itpnode = ocl_op_nodes.find(p);
                    const OpNode *pnode = (itpnode == ocl_op_nodes.end()) ? nullptr : itpnode->second;

                    if (itg == ggml_op_outs.end() || ito == ocl_op_outs.end() || !pnode) {
                        fmt::print("  [BT] producer op#{} missing cached out0/opnode, stop.\n", p);
                        break;
                    }

                    size_t bad_i = 0;
                    float diff = 0.f;
                    if (allclose_span(ito->second, itg->second, ATOL, RTOL, &bad_i, &diff)) {
                        fmt::print("  [BT] producer op#{} type={} via prev[{}] out0 matches (atol/rtol). stop.\n",
                                   p, op_type_to_string(pnode->op), prev_i);
                        break;
                    }

                    fmt::print("  [BT] mismatch at producer op#{} type={} via prev[{}] bad_i={} ocl={} ggml={} diff={}\n",
                               p, op_type_to_string(pnode->op), prev_i,
                               bad_i, ito->second[bad_i], itg->second[bad_i], diff);

                    cur = p;
                }

                fmt::print("=====================================================\n");
            };

            auto compare_out0 = [&](int op_idx, const OpNode *op) {
                if ((int)op->next.size() <= 0) return;
                Tensor *out = op->next[0] ? op->next[0]->tensor() : nullptr;
                if (!out || !out->m_data) return;
                if (out->m_dtype != DataType::FP32) return;

                auto itg = ggml_op_outs.find(op_out_key(op_idx, 0));
                if (itg == ggml_op_outs.end()) return;

                auto ocl_vec = tensor_to_f32_vec_any(out, cl_backend);
                auto &gg_vec = itg->second;

                if (ocl_vec.size() != gg_vec.size()) {
                    fmt::print("\n[MISMATCH][{}] op#{} type={} out#0 size mismatch ocl={} ggml={}\n",
                               phase_tag, op_idx, op_type_to_string(op->op), ocl_vec.size(), gg_vec.size());
                    print_tensor_meta(out, "out");
                    if (op->op == OpType::VIEW) dump_view_root_cause(op_idx, op);
                    backtrace_chain(op_idx);
                    std::exit(1);
                }

                size_t bad_i = 0;
                float diff = 0.f;
                if (!allclose_span(ocl_vec, gg_vec, ATOL, RTOL, &bad_i, &diff)) {
                    dump_mismatch_detail(op_idx, op, 0, out, gg_vec, ocl_vec, bad_i, diff);
                    backtrace_chain(op_idx);
                    std::exit(1);
                }
            };

            // ---- OpenCL pass hook ----
            set_op_after_exec_hook([&](int op_idx, const OpNode *op) {
                ocl_op_nodes[op_idx] = op;

                // record writes by CL range
                record_op_writes(op_idx, op);

                // build producer mapping
                for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                    Tensor *o2 = op->next[oi] ? op->next[oi]->tensor() : nullptr;
                    if (o2 && producer.find(o2) == producer.end()) producer[o2] = op_idx;
                }

                // cache FP32 outputs for backtrace
                for (int oi = 0; oi < (int)op->next.size(); ++oi) {
                    Tensor *out = op->next[oi] ? op->next[oi]->tensor() : nullptr;
                    if (!out || !out->m_data) continue;
                    if (out->m_dtype != DataType::FP32) continue;
                    ocl_op_outs[op_out_key(op_idx, oi)] = tensor_to_f32_vec_any(out, cl_backend);
                }

                // live compare
                compare_out0(op_idx, op);
            });

            auto ret_o = model_ocl->forward(in_tokens, in_pos, in_mask, lm_head);
            set_op_after_exec_hook(nullptr);

            return {ret_g, ret_o};
        };

    // =========================
    // PREFILL compare
    // =========================
    {
        auto mask = CausalAttentionMask(tokens.size());

        auto [ret_g, ret_o] = run_forward_per_op_cmp(tokens, pos, mask, /*lm_head*/true, "PREFILL");

        auto lg = ret_g.logits_vector.back();
        auto lo = ret_o.logits_vector.back();

        size_t bad_i = 0;
        float diff = 0.f;
        if (!allclose_span(lo, lg, ATOL, RTOL, &bad_i, &diff)) {
            fmt::print("PREFILL logits mismatch (end-to-end): bad_i={}, ocl={}, ggml={}, diff={}\n",
                       bad_i, lo[bad_i], lg[bad_i], diff);
            dump_topk(lg, TOPK, "ggml");
            dump_topk(lo, TOPK, "opencl");
            return 1;
        }

        POWERSERVE_LOG_INFO("Prefill per-op compare PASS");
    }

    // =========================
    // DECODE compare (N steps)
    // =========================
    {
        auto &id_g = model_ggml->m_config->model_id;
        auto &id_o = model_ocl->m_config->model_id;

        platform->reset_kv_position(id_g);
        platform->reset_kv_position(id_o);

        // prefill prompt[:-1] with lm_head=false
        if (tokens.size() > 1) {
            std::vector<Token> prefill_tokens(tokens.begin(), tokens.end() - 1);
            std::vector<int> prefill_pos(prefill_tokens.size());
            for (size_t i = 0; i < prefill_pos.size(); ++i) prefill_pos[i] = (int)i;

            auto prefill_mask = CausalAttentionMask(prefill_tokens.size());
            (void)run_forward_per_op_cmp(prefill_tokens, prefill_pos, prefill_mask, /*lm_head*/false, "DECODE_PREFILL");
        }

        int token_in = tokens.back();
        for (int step = 0; step < DECODE_STEPS; ++step) {
            size_t pos_g = platform->get_kv_position(id_g);
            size_t pos_o = platform->get_kv_position(id_o);
            POWERSERVE_ASSERT(pos_g == pos_o);

            std::vector<Token> step_tokens(1, token_in);
            std::vector<int> step_pos(1, (int)pos_g);
            auto step_mask = CausalAttentionMask(1);

            auto [ret_g, ret_o] = run_forward_per_op_cmp(step_tokens, step_pos, step_mask, /*lm_head*/true, "DECODE");

            auto lg = ret_g.logits_vector.back();
            auto lo = ret_o.logits_vector.back();

            size_t bad_i = 0;
            float diff = 0.f;
            if (!allclose_span(lo, lg, ATOL, RTOL, &bad_i, &diff)) {
                fmt::print("\n[DECODE logits mismatch] step={} pos={} token_in={}\n", step, pos_g, token_in);
                fmt::print("  bad_i={}, ocl={}, ggml={}, diff={}\n", bad_i, lo[bad_i], lg[bad_i], diff);
                dump_topk(lg, TOPK, "ggml");
                dump_topk(lo, TOPK, "opencl");
                return 1;
            }

            int next_token = argmax_span(lg);
            fmt::print("[decode step {}] pos={} token_in={} -> next_token={}\n", step, pos_g, token_in, next_token);
            token_in = next_token;
        }

        POWERSERVE_LOG_INFO("Decode per-op compare PASS ({} steps)", DECODE_STEPS);
    }

    platform->ggml_backends[model_ggml->m_config->model_id]->reset_threadpool();
    platform->ggml_backends[model_ocl->m_config->model_id]->reset_threadpool();
    return 0;
}
