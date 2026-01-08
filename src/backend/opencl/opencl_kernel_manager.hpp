// opencl_kernel_manager.hpp
#pragma once

#include "backend/opencl/opencl_context.hpp"
#include "core/logger.hpp"
#include "core/data_type.hpp"

#include <CL/cl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace powerserve::opencl {

// 编译选项结构体
struct OpenCLCompileOptions {
    std::string opencl_c_std = "CL3.0";  // OpenCL C版本
    bool enable_mad = true;              // -cl-mad-enable
    bool unsafe_math = true;             // -cl-unsafe-math-optimizations
    bool finite_math = true;             // -cl-finite-math-only
    bool fast_relaxed_math = true;       // -cl-fast-relaxed-math
    std::string extra_options = "";      // 额外选项
    
    std::string to_string() const {
        std::string result = "-cl-std=" + opencl_c_std;
        if (enable_mad) result += " -cl-mad-enable";
        if (unsafe_math) result += " -cl-unsafe-math-optimizations";
        if (finite_math) result += " -cl-finite-math-only";
        if (fast_relaxed_math) result += " -cl-fast-relaxed-math";
        if (!extra_options.empty()) result += " " + extra_options;
        return result;
    }
};

// 内核缓存项
struct KernelCacheItem {
    cl_kernel kernel = nullptr;
    std::string name;
    mutable size_t last_used = 0;  // 用于LRU缓存
};

// 程序缓存项
struct ProgramCacheItem {
    cl_program program = nullptr;
    std::string source_hash;  // 源代码哈希，用于避免重复编译
    std::unordered_map<std::string, cl_kernel> kernels;  // 此program中的所有kernels
};

class OpenCLKernelManager {
public:
    explicit OpenCLKernelManager(std::shared_ptr<OpenCLContext> context);
    ~OpenCLKernelManager();
    
    // 禁用拷贝
    OpenCLKernelManager(const OpenCLKernelManager&) = delete;
    OpenCLKernelManager& operator=(const OpenCLKernelManager&) = delete;
    
    // 初始化
    bool initialize(const OpenCLCompileOptions& options = {});
    void cleanup();
    
    // === 内核编译接口 ===
    
    // 从源代码编译program并提取所有kernels
    bool compile_program(const std::string& program_name,
                         const std::string& source_code,
                         const std::string& extra_options = "");
    
    // 从文件加载并编译
    bool load_program_from_file(const std::string& program_name,
                                const std::string& file_path,
                                const std::string& extra_options = "");
    
    // 编译单个kernel（不推荐，通常一起编译一个program的多个kernels）
    cl_kernel compile_kernel(const std::string& kernel_name,
                             const std::string& source_code,
                             const std::string& extra_options = "");
    
    // === 内核获取接口 ===
    
    // 获取已编译的内核
    cl_kernel get_kernel(const std::string& kernel_name) const;
    cl_kernel get_cpy_kernel(powerserve::DataType src_t, powerserve::DataType dst_t) const;
    
    // 检查内核是否存在
    bool has_kernel(const std::string& kernel_name) const;
    
    // 获取program中的所有kernels
    std::vector<cl_kernel> get_all_kernels(const std::string& program_name) const;
    
    // === 内核参数设置辅助函数（仿照llama.cpp） ===
    
    template<typename T>
    static bool set_kernel_arg(cl_kernel kernel, cl_uint arg_index, const T& value) {
        cl_int ret = clSetKernelArg(kernel, arg_index, sizeof(T), &value);
        if (ret != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to set kernel arg {}: {}", arg_index, 
                                OpenCLContext::get_error_string(ret));
            return false;
        }
        return true;
    }
    
    // 特殊化处理：cl_mem
    static bool set_kernel_arg(cl_kernel kernel, cl_uint arg_index, cl_mem value) {
        cl_int ret = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &value);
        if (ret != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to set kernel arg {} (cl_mem): {}", arg_index,
                                OpenCLContext::get_error_string(ret));
            return false;
        }
        return true;
    }
    
    // 特殊化处理：指针
    template<typename T>
    static bool set_kernel_arg_ptr(cl_kernel kernel, cl_uint arg_index, const T* value) {
        cl_int ret = clSetKernelArg(kernel, arg_index, sizeof(T*), value);
        if (ret != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to set kernel arg {} (pointer): {}", arg_index,
                                OpenCLContext::get_error_string(ret));
            return false;
        }
        return true;
    }
    
    // === 内核执行辅助函数 ===
    
    // 执行内核（1D）
    bool enqueue_kernel_1d(cl_kernel kernel, 
                          size_t global_size,
                          size_t local_size = 0,
                          cl_event* event = nullptr);
    
    // 执行内核（2D）
    bool enqueue_kernel_2d(cl_kernel kernel,
                          size_t global_size_x, size_t global_size_y,
                          size_t local_size_x = 0, size_t local_size_y = 0,
                          cl_event* event = nullptr);
    
    // 执行内核（3D）- 仿照llama.cpp的常见模式
    bool enqueue_kernel_3d(cl_kernel kernel,
                          size_t global_size_x, size_t global_size_y, size_t global_size_z,
                          size_t local_size_x = 0, size_t local_size_y = 0, size_t local_size_z = 0,
                          cl_event* event = nullptr);
    
    // === 信息查询 ===
    
    size_t get_program_count() const { return programs_.size(); }
    size_t get_kernel_count() const { return kernel_cache_.size(); }
    std::vector<std::string> get_available_kernels() const;
    std::vector<std::string> get_available_programs() const;
    
    // === 嵌入式内核支持（仿照GGML_OPENCL_EMBED_KERNELS） ===
    
    // 注册嵌入式内核源码（在编译时嵌入）
    void register_embedded_source(const std::string& program_name,
                                  const std::string& source_code);
    
    // 检查是否有嵌入式源码
    bool has_embedded_source(const std::string& program_name) const;
    
private:
    std::shared_ptr<OpenCLContext> context_;
    OpenCLCompileOptions compile_options_;
    
    // 缓存
    std::unordered_map<std::string, ProgramCacheItem> programs_;
    std::unordered_map<std::string, KernelCacheItem> kernel_cache_;  // 快速查找
    std::unordered_map<std::string, std::string> embedded_sources_;  // 嵌入式源码
    
    mutable std::mutex mutex_;  // 线程安全
    
    // 内部方法
    std::string build_compile_options(const std::string& extra_options) const;
    cl_program compile_program_impl(const std::string& source_code, 
                                   const std::string& options);
    bool extract_kernels_from_program(cl_program program, 
                                     const std::string& program_name);
    
    // 错误处理
    bool check_build_error(cl_program program, cl_device_id device) const;
    std::string get_program_build_log(cl_program program) const;
    
    // 工具函数
    static std::string compute_source_hash(const std::string& source);
    static std::vector<std::string> split_kernel_names(const std::string& source);
    
    bool compile_embedded_kernels();
};

} // namespace powerserve::opencl