// opencl_backend.hpp
#pragma once

#include "backend/backend.hpp"
#include "backend/opencl/opencl_context.hpp"
#include "backend/opencl/opencl_memory.hpp"
#include "backend/opencl/opencl_kernel_manager.hpp"
#include "backend/opencl/opencl_buffer.hpp"
#include "core/tensor.hpp"
#include "core/thread_pool.hpp"
#include "graph/node.hpp"

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace powerserve::opencl {

struct OpenCLBackend : Backend {
public:
    struct WorkData {
        std::vector<char> buffer;
        size_t size = 0;
    };
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
    
    // 核心组件
    std::shared_ptr<OpenCLContext> context;
    std::shared_ptr<OpenCLMemoryPool> memory_pool;
    std::shared_ptr<OpenCLKernelManager> kernel_manager;
    std::unique_ptr<ThreadPool> thread_pool;
    WorkData work_data;
    
    // 配置
    std::vector<ThreadConfig> thread_config;
    int num_threads = 1;
    std::string device_preference;
    
    // 状态
    bool initialized = false;
    
public:
    explicit OpenCLBackend(const ModelConfig::LLMConfig &config, const HyperParams &hparams);
    ~OpenCLBackend();
    
    // 初始化方法
    bool initialize();
    void cleanup();
    
    // === 主要算子接口（仿照GGMLBackend）===
    
    // 张量运算
    void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const;
    void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rms_norm(Tensor * dst, const Tensor * src, float eps) const;
    void rope(const Tensor *dst, const Tensor *src0, const Tensor *src1, const RopeParams &p, const Tensor *src2 ) const;
    void softmax(const Tensor * dst, const Tensor * src0, const Tensor * src1) const;
    void permute(const Tensor *out, const Tensor *x, Shape axes) const;
    void cont(const Tensor *out, const Tensor *x) const;
    void softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias) const;
    
    // 张量属性检查
    bool is_contiguous(const Tensor *tensor, int n) const;
    int get_n_tasks(std::shared_ptr<OpNode> op);
    enum ggml_type get_vec_dot_type(const Tensor *tensor);
    
    // 其他算子
    void silu(const Tensor* dst, const Tensor* src0) const;
    void gelu(const Tensor* dst, const Tensor* src0) const;
    void copy(const Tensor *dst, const Tensor *src) const;
    void print(const Tensor *x, size_t size) const;
    void reset_kv_batch_size(const size_t batch_size) const;
    void add_cache(const Tensor *k, const Tensor *v, size_t L, const std::vector<int> &pos, size_t head_id);
    void transpose(const Tensor *out, const Tensor *x) const;
    void diag_mask_inf(const Tensor *dst, const Tensor *src0, int n_past) const;
    
    // 计划调度
    void plan(std::vector<std::shared_ptr<OpNode>> &ops);
    void setup_work_data(size_t work_size);
    
    // 内存管理接口
    cl_mem allocate_device_memory(size_t size);
    void free_device_memory(cl_mem buffer);
    
    // 数据传输
    bool copy_to_device(cl_mem dst, const void* src, size_t size);
    bool copy_to_host(void* dst, cl_mem src, size_t size);
    
    // 缓冲区创建
    std::shared_ptr<OpenCLBuffer> create_buffer(Shape shape, DataType dtype);
    
    // 线程池管理
    void reset_threadpool();
    
    // 设备信息
    std::string get_device_name() const;
    size_t get_device_memory() const;
    
    // 状态检查
    bool is_initialized() const { return initialized; }

    // rope接口专用
    static inline int dim4(const Tensor *t, int i) {
        return t ? (int)t->m_shape[i] : 1;
    }
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
    void setup_default_config();
    
    // 辅助函数
    cl_mem get_cl_buffer(const Tensor* tensor) const;
    cl_kernel get_kernel_for_type(const std::string& base_name, DataType dtype) const;
    void set_kernel_args_from_tensors(cl_kernel kernel, const std::vector<const Tensor*>& tensors) const;
    
    // Tensor到OpenCL buffer的映射
    mutable std::unordered_map<const Tensor*, cl_mem> tensor_buffers_;
    mutable std::mutex buffer_mutex_;
};

} // namespace powerserve::opencl