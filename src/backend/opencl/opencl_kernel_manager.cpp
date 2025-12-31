// opencl_kernel_manager.cpp
#include "opencl_kernel_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#ifdef POWERSERVE_OPENCL_EMBED_KERNELS
#include "opencl_embedded_kernels.hpp"
#endif

namespace powerserve::opencl {

// 构造函数和析构函数
OpenCLKernelManager::OpenCLKernelManager(std::shared_ptr<OpenCLContext> context)
    : context_(std::move(context)) {
}

OpenCLKernelManager::~OpenCLKernelManager() {
    cleanup();
}

// 初始化方法
bool OpenCLKernelManager::initialize(const OpenCLCompileOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    compile_options_ = options;
    
    // 编译嵌入式内核
#ifdef POWERSERVE_OPENCL_EMBED_KERNELS
    bool success = compile_embedded_kernels();
    if (!success) {
        POWERSERVE_LOG_ERROR("Failed to compile embedded OpenCL kernels");
        return false;
    }
#else
    POWERSERVE_LOG_DEBUG("POWERSERVE_OPENCL_EMBED_KERNELS is NOT defined");
#endif
    
    return true;
}

bool OpenCLKernelManager::compile_embedded_kernels() {
#ifdef POWERSERVE_OPENCL_EMBED_KERNELS
    
    bool all_success = true;
    
    // 1. 编译 copy 内核
#ifdef OPENCL_CPY_CL_AVAILABLE
    {
        const std::string& cpy_source = ::powerserve::opencl::embedded::cpy_cl_source;
        
        if (!cpy_source.empty()) {
            if (!compile_program("copy_kernels", cpy_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile copy kernels");
                all_success = false;
            }
        }
    }
#endif // OPENCL_CPY_CL_AVAILABLE
    
    // 2. 编译 add 内核
#ifdef OPENCL_ADD_CL_AVAILABLE
    {
        const std::string& add_source = ::powerserve::opencl::embedded::add_cl_source;
        
        if (!add_source.empty()) {
            if (!compile_program("add_kernels", add_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile add kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("Add kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_ADD_CL_AVAILABLE

    // 3. 编译 silu 内核
#ifdef OPENCL_SILU_CL_AVAILABLE
    {
        const std::string& silu_source = ::powerserve::opencl::embedded::silu_cl_source;
        
        if (!silu_source.empty()) {
            if (!compile_program("silu_kernels", silu_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile silu kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("silu kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_SILU_CL_AVAILABLE

    // 4. 编译 gelu 内核
#ifdef OPENCL_GELU_CL_AVAILABLE
    {
        const std::string& gelu_source = ::powerserve::opencl::embedded::gelu_cl_source;
        
        if (!gelu_source.empty()) {
            if (!compile_program("gelu_kernels", gelu_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile gelu kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("gelu kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_GELU_CL_AVAILABLE

    // 5. 编译 matmul 内核
#ifdef OPENCL_MATMUL_CL_AVAILABLE
    {
        const std::string& matmul_source = ::powerserve::opencl::embedded::mul_mat_f16_f32_cl_source;
        
        if (!matmul_source.empty()) {
            if (!compile_program("matmul_kernels", matmul_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile matmul kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("matmul kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_MATMUL_CL_AVAILABLE

    // 6. 编译 rms_norm 内核
#ifdef OPENCL_RMS_NORM_CL_AVAILABLE
    {
        const std::string& rms_norm_source = ::powerserve::opencl::embedded::rms_norm_cl_source;
        
        if (!rms_norm_source.empty()) {
            if (!compile_program("rms_norm_kernels", rms_norm_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile rms_norm kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("rms_norm kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_RMS_NORM_CL_AVAILABLE

    // 7. 编译 softmax 内核
#ifdef OPENCL_SOFTMAX_CL_AVAILABLE
    {
        const std::string& softmax_source = ::powerserve::opencl::embedded::softmax_f32_cl_source;
        
        if (!softmax_source.empty()) {
            if (!compile_program("softmax_kernels", softmax_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile softmax kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("softmax kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_SOFTMAX_CL_AVAILABLE

    // 8. 编译 rope 内核
#ifdef OPENCL_ROPE_CL_AVAILABLE
    {
        const std::string& rope_source = ::powerserve::opencl::embedded::rope_cl_source;
        
        if (!rope_source.empty()) {
            if (!compile_program("rope_kernels", rope_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile rope kernels");
                all_success = false;
            }
        }
    }
#endif // OPENCL_ROPE_CL_AVAILABLE

    // 9. 编译 diag_mask_inf 内核
#ifdef OPENCL_DIAG_MASK_INF_CL_AVAILABLE
    {
        const std::string& diag_mask_inf_source = ::powerserve::opencl::embedded::diag_mask_inf_cl_source;
        
        if (!diag_mask_inf_source.empty()) {
            if (!compile_program("diag_mask_inf_kernels", diag_mask_inf_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile diag_mask_inf kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("diag_mask_inf kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_DIAG_MASK_INF_CL_AVAILABLE

    return all_success;
    
#else
    POWERSERVE_LOG_DEBUG("Embedded kernels not enabled");
    return true; // 不视为错误
#endif // POWERSERVE_OPENCL_EMBED_KERNELS
}

// 编译program（核心方法）
bool OpenCLKernelManager::compile_program(const std::string& program_name,
                                         const std::string& source_code,
                                         const std::string& extra_options) {
    
    
    // 检查是否已存在
    if (programs_.find(program_name) != programs_.end()) {
        POWERSERVE_LOG_WARN("Program '{}' already compiled", program_name);
        return true;
    }
    
    // 检查源代码是否为空
    if (source_code.empty()) {
        POWERSERVE_LOG_ERROR("Empty source code for program: {}", program_name);
        return false;
    }
    
    // 构建编译选项
    std::string options = build_compile_options(extra_options);
    
    // 编译program
    cl_program program = compile_program_impl(source_code, options);
    
    if (!program) {
        POWERSERVE_LOG_ERROR("Failed to compile program '{}'", program_name);
        return false;
    }
    
    std::vector<std::string> kernel_names = split_kernel_names(source_code);
    
    // 如果没找到，输出源码片段帮助调试
    if (kernel_names.empty()) {
        POWERSERVE_LOG_WARN("No kernels found in program: {}", program_name);
        
        // 输出源码前几行看看格式
        std::istringstream source_stream(source_code);
        std::string line;
        int line_count = 0;
        POWERSERVE_LOG_DEBUG("First 10 lines of source:");
        while (std::getline(source_stream, line) && line_count < 10) {
            POWERSERVE_LOG_DEBUG("  Line {}: {}", line_count + 1, line);
            line_count++;
        }
        
        // 查找可能的kernel定义
        size_t kernel_pos = source_code.find("kernel");
        if (kernel_pos != std::string::npos) {
            size_t sample_start = (kernel_pos > 50) ? kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found 'kernel' at position {}, sample:", kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
        
        // 也查找带下划线的版本
        size_t underscore_kernel_pos = source_code.find("__kernel");
        if (underscore_kernel_pos != std::string::npos) {
            size_t sample_start = (underscore_kernel_pos > 50) ? underscore_kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), underscore_kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found '__kernel' at position {}, sample:", underscore_kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
    } else {
        // 输出找到的内核名
        // for (const auto& kernel_name : kernel_names) {
        //     POWERSERVE_LOG_DEBUG("  Kernel: {}", kernel_name);
        // }
    }
    
    // 为每个kernel创建cl_kernel对象
    std::unordered_map<std::string, cl_kernel> kernels;
    for (const auto& kernel_name : kernel_names) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to create kernel '{}': {}", 
                               kernel_name, context_->get_error_string(err));
            // 继续尝试其他kernels
            continue;
        }
        
        kernels[kernel_name] = kernel;
        
        // 同时添加到kernel_cache_
        KernelCacheItem cache_item;
        cache_item.kernel = kernel;
        cache_item.name = kernel_name;
        cache_item.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        kernel_cache_[kernel_name] = cache_item;
        
        // POWERSERVE_LOG_DEBUG("Created kernel: {}", kernel_name);
    }
    
    // 创建缓存项
    ProgramCacheItem item;
    item.program = program;
    item.source_hash = compute_source_hash(source_code);
    item.kernels = std::move(kernels);
    
    programs_[program_name] = std::move(item);
    
    return true;
}

// 从program中提取所有kernels（仿照llama.cpp模式）
bool OpenCLKernelManager::extract_kernels_from_program(cl_program program,
                                                      const std::string& program_name) {
    // 这里可以分析源码自动提取kernel名，或者预定义
    // 对于简单情况，我们可以让调用者指定要提取的kernels
    
    // 临时方案：先不自动提取，需要手动通过get_kernel创建
    return true;
}

// 获取内核（如果没有则从program中创建）
cl_kernel OpenCLKernelManager::get_kernel(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查缓存
    auto it = kernel_cache_.find(kernel_name);
    if (it != kernel_cache_.end()) {
        it->second.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        return it->second.kernel;
    }
    
    // 需要知道这个kernel属于哪个program
    // 这里需要一个映射：kernel_name -> program_name
    // 暂时简化：假设program名就是kernel的前缀（如"add" -> "kernel_add"）
    
    POWERSERVE_LOG_ERROR("Kernel '{}' not found. Need to implement program-kernel mapping", 
                        kernel_name);
    return nullptr;
}

// 内核执行方法（1D）
bool OpenCLKernelManager::enqueue_kernel_1d(cl_kernel kernel,
                                           size_t global_size,
                                           size_t local_size,
                                           cl_event* event) {
    cl_command_queue queue = context_->get_queue();
    cl_int ret;
    
    if (local_size == 0) {
        // 让OpenCL驱动选择合适的工作组大小
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, nullptr,
                                    0, nullptr, event);
    } else {
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, event);
    }
    
    if (ret != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to enqueue 1D kernel: {}", 
                            context_->get_error_string(ret));
        return false;
    }
    
    return true;
}
cl_kernel OpenCLKernelManager::compile_kernel(const std::string& kernel_name,
                                              const std::string& source_code,
                                              const std::string& extra_options) {
    
    // 构建编译选项
    std::string options = build_compile_options(extra_options);
    
    // 编译program
    cl_program program = compile_program_impl(source_code, options);
    if (!program) {
        return nullptr;
    }
    
    // 创建kernel
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to create kernel '{}': {}", 
                            kernel_name, context_->get_error_string(err));
        clReleaseProgram(program);
        return nullptr;
    }
    
    // 缓存kernel - 使用正确的成员变量名 kernel_cache_
    KernelCacheItem item;
    item.kernel = kernel;
    item.name = kernel_name;
    item.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
    
    kernel_cache_[kernel_name] = item;
    
    // 也缓存program（可选）
    ProgramCacheItem prog_item;
    prog_item.program = program;
    prog_item.source_hash = compute_source_hash(source_code);
    programs_[kernel_name + "_program"] = prog_item;
    
    return kernel;
}

// 修改 cleanup 函数
void OpenCLKernelManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 释放所有kernels - 使用 kernel_cache_
    for (auto& [name, item] : kernel_cache_) {
        if (item.kernel) {
            clReleaseKernel(item.kernel);
        }
    }
    kernel_cache_.clear();
    
    // 释放所有programs
    for (auto& [name, item] : programs_) {
        if (item.program) {
            clReleaseProgram(item.program);
        }
    }
    programs_.clear();
    
    embedded_sources_.clear();
}

// 构建编译选项
std::string OpenCLKernelManager::build_compile_options(const std::string& extra_options) const {
    std::string options = compile_options_.to_string();
    if (!extra_options.empty()) {
        options += " " + extra_options;
    }
    return options;
}

// 编译program实现
cl_program OpenCLKernelManager::compile_program_impl(const std::string& source_code,
                                                    const std::string& options) {
    
    cl_int err;
    const char* source_cstr = source_code.c_str();
    size_t source_len = source_code.length();
    
    cl_program program = clCreateProgramWithSource(context_->get_context(), 1,
                                                   &source_cstr, &source_len, &err);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to create program: {}", 
                            context_->get_error_string(err));
        return nullptr;
    }
    
    cl_device_id device = context_->get_device();
    
    err = clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        // 获取构建日志
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        POWERSERVE_LOG_ERROR("Failed to build program: {}", context_->get_error_string(err));
        POWERSERVE_LOG_ERROR("Build log:\n{}", log.data());
        
        clReleaseProgram(program);
        return nullptr;
    }
    return program;
}

// 计算源码哈希
std::string OpenCLKernelManager::compute_source_hash(const std::string& source) {
    // 简单实现：使用字符串长度和部分内容作为哈希
    std::hash<std::string> hasher;
    return std::to_string(hasher(source));
}

// 检查构建错误
bool OpenCLKernelManager::check_build_error(cl_program program, cl_device_id device) const {
    cl_build_status status;
    cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                                      sizeof(status), &status, nullptr);
    return (err == CL_SUCCESS && status == CL_BUILD_SUCCESS);
}

// 获取构建日志
std::string OpenCLKernelManager::get_program_build_log(cl_program program) const {
    cl_device_id device = context_->get_device();
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    
    return std::string(log.data());
}

// 分割内核名
std::vector<std::string> OpenCLKernelManager::split_kernel_names(const std::string& source) {
    std::vector<std::string> kernels;
    
    // 更智能的搜索：跳过注释
    bool in_block_comment = false;
    bool in_line_comment = false;
    
    for (size_t i = 0; i < source.length(); i++) {
        // 处理块注释 /* */
        if (!in_line_comment && i + 1 < source.length() && 
            source[i] == '/' && source[i+1] == '*') {
            in_block_comment = true;
            i++; // 跳过 '*'
            continue;
        }
        
        if (in_block_comment && i + 1 < source.length() && 
            source[i] == '*' && source[i+1] == '/') {
            in_block_comment = false;
            i++; // 跳过 '/'
            continue;
        }
        
        // 处理行注释 //
        if (!in_block_comment && i + 1 < source.length() && 
            source[i] == '/' && source[i+1] == '/') {
            in_line_comment = true;
            i++; // 跳过第二个 '/'
            continue;
        }
        
        if (in_line_comment && source[i] == '\n') {
            in_line_comment = false;
            continue;
        }
        
        // 如果不在注释中，查找 kernel 关键字
        if (!in_block_comment && !in_line_comment) {
            // 查找 "kernel" 关键字
            if (i + 5 < source.length() && 
                source.substr(i, 6) == "kernel") {
                
                // 跳过 "kernel" 关键字
                size_t pos = i + 6;
                
                // 跳过空白
                while (pos < source.length() && std::isspace(source[pos])) {
                    pos++;
                }
                
                // 检查是否是 "void"（kernel void xxx）
                if (pos + 3 < source.length() && source.substr(pos, 4) == "void") {
                    pos += 4; // 跳过 "void"
                    
                    // 跳过空白
                    while (pos < source.length() && std::isspace(source[pos])) {
                        pos++;
                    }
                    
                    // 提取内核名
                    size_t name_start = pos;
                    while (pos < source.length() && 
                           (std::isalnum(source[pos]) || source[pos] == '_')) {
                        pos++;
                    }
                    
                    if (pos > name_start) {
                        std::string kernel_name = source.substr(name_start, pos - name_start);
                        kernels.push_back(kernel_name);
                        // POWERSERVE_LOG_DEBUG("Found kernel: {}", kernel_name);
                        i = pos - 1; // 继续从当前位置搜索
                    }
                }
            }
        }
    }
    
    if (kernels.empty()) {
        // 备用方法：直接搜索 kernel_ 开头的函数名
        size_t pos = 0;
        while ((pos = source.find("kernel_", pos)) != std::string::npos) {
            // 检查前面是否有注释
            bool is_commented = false;
            
            // 检查前面是否有 //
            for (size_t i = pos; i > 0 && i > pos - 100; i--) {
                if (source[i] == '\n') break;
                if (i >= 1 && source[i-1] == '/' && source[i] == '/') {
                    is_commented = true;
                    break;
                }
                if (i >= 1 && source[i-1] == '/' && source[i] == '*') {
                    is_commented = true;
                    break;
                }
            }
            
            if (!is_commented) {
                size_t name_end = source.find('(', pos);
                if (name_end != std::string::npos) {
                    std::string kernel_name = source.substr(pos, name_end - pos);
                    kernels.push_back(kernel_name);
                    POWERSERVE_LOG_DEBUG("Found kernel via backup search: {}", kernel_name);
                }
            }
            pos += 7; // "kernel_"的长度
        }
    }
    
    return kernels;
}
// 2D和3D内核执行方法
bool OpenCLKernelManager::enqueue_kernel_2d(cl_kernel kernel,
                                           size_t global_size_x, size_t global_size_y,
                                           size_t local_size_x, size_t local_size_y,
                                           cl_event* event) {
    cl_command_queue queue = context_->get_queue();
    cl_int ret;
    
    size_t global_size[] = {global_size_x, global_size_y};
    size_t local_size[] = {local_size_x, local_size_y};
    
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                 global_size, local_size,
                                 0, nullptr, event);
    
    if (ret != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to enqueue 2D kernel: {}", 
                            context_->get_error_string(ret));
        return false;
    }
    
    return true;
}

bool OpenCLKernelManager::enqueue_kernel_3d(cl_kernel kernel,
                                           size_t global_size_x, size_t global_size_y, size_t global_size_z,
                                           size_t local_size_x, size_t local_size_y, size_t local_size_z,
                                           cl_event* event) {
    cl_command_queue queue = context_->get_queue();
    cl_int ret;
    
    size_t global_size[] = {global_size_x, global_size_y, global_size_z};
    size_t local_size[] = {local_size_x, local_size_y, local_size_z};
    
    ret = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                 global_size, local_size,
                                 0, nullptr, event);
    
    if (ret != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to enqueue 3D kernel: {}", 
                            context_->get_error_string(ret));
        return false;
    }
    
    return true;
}

// 获取可用内核列表
std::vector<std::string> OpenCLKernelManager::get_available_kernels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> kernels;
    
    for (const auto& [name, item] : kernel_cache_) {
        kernels.push_back(name);
    }
    
    return kernels;
}

// 获取可用程序列表
std::vector<std::string> OpenCLKernelManager::get_available_programs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> program_names;
    
    for (const auto& [name, item] : programs_) {
        program_names.push_back(name);
    }
    
    return program_names;
}

// 注册嵌入式源码
void OpenCLKernelManager::register_embedded_source(const std::string& program_name,
                                                   const std::string& source_code) {
    std::lock_guard<std::mutex> lock(mutex_);
    embedded_sources_[program_name] = source_code;
}

// 检查是否有嵌入式源码
bool OpenCLKernelManager::has_embedded_source(const std::string& program_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return embedded_sources_.find(program_name) != embedded_sources_.end();
}

} // namespace powerserve::opencl