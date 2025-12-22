#pragma once

#include "backend/backend.hpp"          // Backend interface (do NOT redefine Backend here)

// --- OpenCL backend building blocks (current) ---
#include "backend/opencl/opencl_env.hpp"
#include "backend/opencl/opencl_kv_cache.hpp"

// --- OpenCL backend building blocks (backup, keep if still exists) ---
#include "backend/opencl/opencl_memory.hpp"
#include "backend/opencl/opencl_kernel_manager.hpp"
#include "backend/opencl/opencl_buffer.hpp"

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
    // current
    std::shared_ptr<OpenCLEnv> m_env;

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
    // ========= Extra helpers / legacy APIs (NOT Backend interface) =========
    // (debug only)
    std::shared_ptr<OpenCLBuffer> debug_get_k_cache(size_t L) const;
    std::shared_ptr<OpenCLBuffer> debug_get_v_cache(size_t L) const;
    
    // 张量属性检查 / 并行任务估算（如果还需要）
    bool is_contiguous(const Tensor *tensor, int n) const;
    int  get_n_tasks(std::shared_ptr<OpNode> op);
    enum ggml_type get_vec_dot_type(const Tensor *tensor);

    // 其他算子（如果你项目里仍有调用，可以保留）
    void silu(const Tensor *dst, const Tensor *src0) const;
    void gelu(const Tensor *dst, const Tensor *src0) const;

    // 旧接口风格的软最大/rope/rms_norm：保留给内部迁移使用
    void rms_norm_legacy(Tensor *dst, const Tensor *src, float eps) const;
    void softmax_legacy(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rope_legacy(
        const Tensor *dst,
        const Tensor *src0,
        const Tensor *src1,
        const RopeParams &p,
        const Tensor *src2
    ) const;

    void diag_mask_inf(const Tensor *dst, const Tensor *src0, int n_past) const;

    // plan/工作区
    void setup_work_data(size_t work_size);

    // 内存管理/数据传输（如果你仍想保留这些“直接 OpenCL API”入口）
    cl_mem allocate_device_memory(size_t size);
    void   free_device_memory(cl_mem buffer);

    bool copy_to_device(cl_mem dst, const void *src, size_t size);
    bool copy_to_host(void *dst, cl_mem src, size_t size);

    // buffer 创建（更贴近你之前的实现；Executor 不应该直接用它）
    std::shared_ptr<OpenCLBuffer> create_buffer(Shape shape, DataType dtype);

    // 线程池管理
    void reset_threadpool();

    // 设备信息
    size_t      get_device_memory() const;

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

    cl_mem    get_cl_buffer(const Tensor *tensor) const;
    cl_kernel get_kernel_for_type(const std::string &base_name, DataType dtype) const;
    void set_kernel_args_from_tensors(
        cl_kernel kernel,
        const std::vector<const Tensor *> &tensors
    ) const;

private:
    // Tensor -> OpenCL buffer mapping (legacy path; if you已经把 buffer 放在 Tensor 内部，可逐步淘汰)
    mutable std::unordered_map<const Tensor *, cl_mem> tensor_buffers_;
    mutable std::mutex buffer_mutex_;

    void ensure_kv_cache_allocated_v0();
    mutable ModelConfig::LLMConfig m_llm;   // 保存模型参数来源
    mutable HyperParams m_hparams;          // 如果未来要用也方便
    mutable std::unique_ptr<powerserve::opencl::OpenCLKV> m_kv;
};

} // namespace powerserve::opencl
