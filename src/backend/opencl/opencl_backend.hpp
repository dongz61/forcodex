#pragma once

#include "backend/backend.hpp"

#include "backend/opencl/opencl_kv_cache.hpp"

#include "backend/opencl/opencl_memory.hpp"
#include "backend/opencl/opencl_kernel_manager.hpp"
#include "backend/opencl/opencl_buffer.hpp"

#include "backend/ggml/ggml.hpp"     
#include "ggml.h"    

#include "core/config.hpp"
#include "core/tensor.hpp"
#include "core/thread_pool.hpp"
#include "graph/node.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace powerserve::opencl {

struct OpenCLBackend final : powerserve::Backend {
public:
    // ========= Backup carried structs (helpers) =========
    struct WorkData {
        std::vector<char> buffer;
        size_t size = 0;
    };

    // 旧 rope 参数结构：保留给内部 kernel/兼容层使用（不是 Backend 接口）
    struct RopeParams {
        int n_past     = 0;
        int n_dims     = 0;
        int mode       = 0;
        int n_ctx_orig = 0;

        float freq_base   = 10000.f;
        float freq_scale  = 1.f;
        float ext_factor  = 0.f;
        float attn_factor = 1.f;
        float beta_fast   = 32.f;
        float beta_slow   = 1.f;

        int32_t sections[4] = {0, 0, 0, 0};
    };

public:
    // ========= Core components (current + backup merged) =========

    // backup (optional, keep if you still use them)
    std::shared_ptr<OpenCLKernelManager> kernel_manager;
    std::unique_ptr<ThreadPool> thread_pool;
    std::shared_ptr<OpenCLMemoryPool> memory_pool;
    std::shared_ptr<OpenCLContext> context;

    // work/config/state (backup)
    WorkData work_data;
    std::vector<ThreadConfig> thread_config;
    std::string device_preference;
    bool initialized = false;

    // current
    int num_threads = 1;

public:
    explicit OpenCLBackend(const ModelConfig::LLMConfig &config, const HyperParams &hparams);
    ~OpenCLBackend() override;

public:
    // ========= Lifecycle =========
    bool initialize();
    void cleanup();

public:
    // ========= Backend interface (MUST match base) =========
    void plan(std::vector<std::shared_ptr<OpNode>> &ops) override;

    void add_broadcast(Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void add_minimal(Tensor * dst, const Tensor * src0, const Tensor * src1) const;
    void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const override;
    void get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const override;
    void matmul_minimal(Tensor * dst, const Tensor * src0, const Tensor * src1) const;
    void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const override;
    void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight, float eps) const override;

    void rope(
        Tensor *out,
        const Tensor *src,
        const std::vector<int> &pos,
        const ModelConfig::LLMConfig::RopeConfig &rope_cfg
    ) const override;

    void softmax(const Tensor *out, const Tensor *x) const override;
    void permute(const Tensor *out, const Tensor *x, Shape axes) const override;
    void cont(const Tensor *out, const Tensor *x) const override;

    void softmax_ext(
        const Tensor *out,
        const Tensor *x,
        const Tensor *mask,
        float scale,
        float max_bias
    ) const override;

    void silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const override;
    void copy(const Tensor *dst, const Tensor *src) const override;

    // print 是 optional 接口：你现在 override 了就保留
    void print(const Tensor *x, size_t size) const override;

    void reset_kv_batch_size(const size_t batch_size) const override;

    void add_cache(
        const Tensor *k,
        const Tensor *v,
        size_t L,
        const std::vector<int> &pos,
        size_t head_id
    ) override;

    void transpose(const Tensor *out, const Tensor *x) const override;

public:    
    // Lightweight cache tensor wrappers for graph construction.
    std::pair<Tensor, Tensor> get_cache_tensors(size_t L) const;

    // 张量属性检查 / 并行任务估算（如果还需要）
    bool is_contiguous(const Tensor *tensor, int n) const;
    enum ggml_type get_vec_dot_type(const Tensor *tensor);

    // buffer 创建
    std::shared_ptr<OpenCLBuffer> create_buffer(Shape shape, DataType dtype) const;

    bool is_initialized() const { return initialized; }

    // rope接口专用小工具（backup 保留）
    static inline int dim4(const Tensor *t, int i) { return t ? (int)t->m_shape[i] : 1; }
    static inline int imin(int a, int b) { return a < b ? a : b; }

#ifndef GGML_ROPE_TYPE_MROPE
#define GGML_ROPE_TYPE_MROPE (1 << 8)
#endif
#ifndef GGML_ROPE_TYPE_VISION
#define GGML_ROPE_TYPE_VISION (1 << 9)
#endif
#ifndef GGML_ROPE_TYPE_IMROPE
#define GGML_ROPE_TYPE_IMROPE (1 << 10)
#endif

private:
    // ========= Internal helpers =========
    void setup_default_config();

    // matmul helper
    void matmul_cpu_ggml_fallback(
        const Tensor *dst,
        const Tensor *src0,
        const Tensor *src1
    ) const;

private:
    // Tensor -> OpenCL buffer mapping (legacy path; if you已经把 buffer 放在 Tensor 内部，可逐步淘汰)
    mutable std::unordered_map<const Tensor *, cl_mem> tensor_buffers_;
    mutable std::mutex buffer_mutex_;

    // ---- GGML reusable fallback executor for CPU ops (matmul etc.) ----
    mutable std::unique_ptr<powerserve::ggml::GGMLBackend> m_ggml_fallback;
    mutable size_t m_ggml_fallback_wsize = 0;

        // ===== Quant weight split cache (for OpenCL quant kernels expecting separate qs/d arrays) =====
    struct QuantSplitKey {
        cl_mem   mem = nullptr;
        size_t   base_offset = 0;
        DataType dtype = DataType::FP32;
        Shape    shape{};
    };

    struct QuantSplitKeyHash {
        size_t operator()(const QuantSplitKey& k) const noexcept {
            size_t h = std::hash<void*>{}((void*)k.mem);
            auto mix = [&](size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix(std::hash<size_t>{}(k.base_offset));
            mix(std::hash<int>{}((int)k.dtype));
            for (int i = 0; i < (int)k.shape.size(); ++i) mix(std::hash<size_t>{}(k.shape[i]));
            return h;
        }
    };

    struct QuantSplitKeyEq {
        bool operator()(const QuantSplitKey& a, const QuantSplitKey& b) const noexcept {
            return a.mem == b.mem &&
                   a.base_offset == b.base_offset &&
                   a.dtype == b.dtype &&
                   a.shape == b.shape;
        }
    };

    struct QuantSplitBuffers {
        std::shared_ptr<OpenCLBuffer> q;   // qs-only buffer (bytes)
        std::shared_ptr<OpenCLBuffer> d;   // d-only buffer (half)
        size_t blocks_total = 0;
    };

    mutable std::unordered_map<QuantSplitKey, QuantSplitBuffers, QuantSplitKeyHash, QuantSplitKeyEq> m_quant_split_cache;
    mutable std::mutex m_quant_split_mutex;

    QuantSplitBuffers get_or_create_split_q4_0(const Tensor* w) const;
    QuantSplitBuffers get_or_create_split_q8_0(const Tensor* w) const;

    void matmul_opencl_f16_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const;
    void matmul_opencl_f32_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const;
    void matmul_opencl_q4_0_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const;
    void matmul_opencl_q8_0_f32(const Tensor* dst, const Tensor* w, const Tensor* x) const;

    void clear_quant_cache() const;

    void ensure_tokens_buffer(size_t token_count) const;
    void ensure_kv_cache_allocated_v0(size_t batch_size);
    mutable ModelConfig::LLMConfig m_llm;   // 保存模型参数来源
    mutable HyperParams m_hparams;          // 如果未来要用也方便
    mutable std::unique_ptr<powerserve::opencl::OpenCLKV> m_kv;

    mutable std::shared_ptr<OpenCLBuffer> m_tokens_buffer;
    mutable size_t m_tokens_capacity = 0;
    mutable std::mutex m_tokens_mutex;
};

} // namespace powerserve::opencl
