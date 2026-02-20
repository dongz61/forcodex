// opencl_kernel_manager.cpp
#include "opencl_kernel_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#ifdef POWERSERVE_OPENCL_EMBED_KERNELS
#include "opencl_embedded_kernels.hpp"
#endif

namespace powerserve::opencl {

OpenCLKernelManager::OpenCLKernelManager(std::shared_ptr<OpenCLContext> context)
    : context_(std::move(context)) {
}

OpenCLKernelManager::~OpenCLKernelManager() {
    cleanup();
}

bool OpenCLKernelManager::initialize(const OpenCLCompileOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    compile_options_ = options;
    
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

#ifdef OPENCL_Q8_ALIGN_X_F32_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::q8_align_x_f32_cl_source;
        if (!src.empty()) {
            if (!compile_program("q8_align_x_f32_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile q8_align_x_f32 kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("q8_align_x_f32 kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MV_Q4_0_F32_8X_FLAT_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mv_q4_0_f32_8x_flat_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mv_q4_0_f32_8x_flat_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mv_q4_0_f32_8x_flat kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mv_q4_0_f32_8x_flat kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MV_Q8_0_F32_FLAT_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mv_q8_0_f32_flat_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mv_q8_0_f32_flat_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mv_q8_0_f32_flat kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mv_q8_0_f32_flat kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MM_Q8_0_F32_L4_LM_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mm_q8_0_f32_l4_lm_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mm_q8_0_f32_l4_lm_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mm_q8_0_f32_l4_lm kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mm_q8_0_f32_l4_lm kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MM_F16_F32_L4_LM_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mm_f16_f32_l4_lm_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mm_f16_f32_l4_lm_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mm_f16_f32_l4_lm kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mm_f16_f32_l4_lm kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MM_F32_F32_L4_LM_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mm_f32_f32_l4_lm_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mm_f32_f32_l4_lm_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mm_f32_f32_l4_lm kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mm_f32_f32_l4_lm kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MV_F16_F32_1ROW_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mv_f16_f32_1row_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mv_f16_f32_1row_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mv_f16_f32_1row kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mv_f16_f32_1row kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MV_F16_F32_L4_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mv_f16_f32_l4_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mv_f16_f32_l4_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mv_f16_f32_l4 kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mv_f16_f32_l4 kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MV_F32_F32_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mv_f32_f32_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mv_f32_f32_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mv_f32_f32 kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mv_f32_f32 kernel source is empty!");
            all_success = false;
        }
    }
#endif

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

#ifdef OPENCL_GET_ROWS_CL_AVAILABLE
    {
        const std::string& get_rows_source = ::powerserve::opencl::embedded::get_rows_cl_source;

        if (!get_rows_source.empty()) {
            if (!compile_program("get_rows_kernels", get_rows_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile get_rows kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("get_rows kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_GET_ROWS_CL_AVAILABLE

#ifdef OPENCL_GET_MASK_CL_AVAILABLE
    {
        const std::string& get_mask_source = ::powerserve::opencl::embedded::get_mask_cl_source;

        if (!get_mask_source.empty()) {
            if (!compile_program("get_mask_kernels", get_mask_source)) {
                POWERSERVE_LOG_ERROR("Failed to compile get_mask kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("get_mask kernel source is empty!");
            all_success = false;
        }
    }
#endif // OPENCL_GET_MASK_CL_AVAILABLE

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
    return true; 
#endif // POWERSERVE_OPENCL_EMBED_KERNELS
}

bool OpenCLKernelManager::compile_program(const std::string& program_name,
                                         const std::string& source_code,
                                         const std::string& extra_options) {
    
    
    if (programs_.find(program_name) != programs_.end()) {
        POWERSERVE_LOG_WARN("Program '{}' already compiled", program_name);
        return true;
    }
    
    if (source_code.empty()) {
        POWERSERVE_LOG_ERROR("Empty source code for program: {}", program_name);
        return false;
    }
    
    std::string options = build_compile_options(extra_options);
    
    cl_program program = compile_program_impl(source_code, options);
    
    if (!program) {
        POWERSERVE_LOG_ERROR("Failed to compile program '{}'", program_name);
        return false;
    }
    
    std::vector<std::string> kernel_names = split_kernel_names(source_code);
    
    if (kernel_names.empty()) {
        POWERSERVE_LOG_WARN("No kernels found in program: {}", program_name);
        
        std::istringstream source_stream(source_code);
        std::string line;
        int line_count = 0;
        POWERSERVE_LOG_DEBUG("First 10 lines of source:");
        while (std::getline(source_stream, line) && line_count < 10) {
            POWERSERVE_LOG_DEBUG("  Line {}: {}", line_count + 1, line);
            line_count++;
        }
        
        size_t kernel_pos = source_code.find("kernel");
        if (kernel_pos != std::string::npos) {
            size_t sample_start = (kernel_pos > 50) ? kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found 'kernel' at position {}, sample:", kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
        
        size_t underscore_kernel_pos = source_code.find("__kernel");
        if (underscore_kernel_pos != std::string::npos) {
            size_t sample_start = (underscore_kernel_pos > 50) ? underscore_kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), underscore_kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found '__kernel' at position {}, sample:", underscore_kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
    } else {
        // for (const auto& kernel_name : kernel_names) {
        //     POWERSERVE_LOG_DEBUG("  Kernel: {}", kernel_name);
        // }
    }
    
    std::unordered_map<std::string, cl_kernel> kernels;
    for (const auto& kernel_name : kernel_names) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to create kernel '{}': {}", 
                               kernel_name, context_->get_error_string(err));
            continue;
        }
        
        kernels[kernel_name] = kernel;
        
        KernelCacheItem cache_item;
        cache_item.kernel = kernel;
        cache_item.name = kernel_name;
        cache_item.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        kernel_cache_[kernel_name] = cache_item;
        
        // POWERSERVE_LOG_DEBUG("Created kernel: {}", kernel_name);
    }
    
    ProgramCacheItem item;
    item.program = program;
    item.source_hash = compute_source_hash(source_code);
    item.kernels = std::move(kernels);
    
    programs_[program_name] = std::move(item);
    
    return true;
}

cl_kernel OpenCLKernelManager::get_kernel(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = kernel_cache_.find(kernel_name);
    if (it != kernel_cache_.end()) {
        it->second.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        return it->second.kernel;
    }
    
    
    POWERSERVE_LOG_ERROR("Kernel '{}' not found. Need to implement program-kernel mapping", 
                        kernel_name);
    return nullptr;
}

cl_kernel OpenCLKernelManager::get_cpy_kernel(powerserve::DataType src_t,
                                              powerserve::DataType dst_t) const {
    if (src_t == powerserve::DataType::FP16 && dst_t == powerserve::DataType::FP16) {
        return get_kernel("kernel_cpy_f16_f16");
    }
    if (src_t == powerserve::DataType::FP16 && dst_t == powerserve::DataType::FP32) {
        return get_kernel("kernel_cpy_f16_f32");
    }
    if (src_t == powerserve::DataType::FP32 && dst_t == powerserve::DataType::FP16) {
        return get_kernel("kernel_cpy_f32_f16");
    }
    if (src_t == powerserve::DataType::FP32 && dst_t == powerserve::DataType::FP32) {
        return get_kernel("kernel_cpy_f32_f32");
    }
    return nullptr;
}

void OpenCLKernelManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, item] : kernel_cache_) {
        if (item.kernel) {
            clReleaseKernel(item.kernel);
        }
    }
    kernel_cache_.clear();
    
    for (auto& [name, item] : programs_) {
        if (item.program) {
            clReleaseProgram(item.program);
        }
    }
    programs_.clear();
    
    embedded_sources_.clear();
}

std::string OpenCLKernelManager::build_compile_options(const std::string& extra_options) const {
    std::string options = compile_options_.to_string();
    if (!extra_options.empty()) {
        options += " " + extra_options;
    }
    return options;
}

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

std::string OpenCLKernelManager::compute_source_hash(const std::string& source) {
    std::hash<std::string> hasher;
    return std::to_string(hasher(source));
}

std::vector<std::string> OpenCLKernelManager::split_kernel_names(const std::string& source) {
    std::vector<std::string> kernels;
    
    bool in_block_comment = false;
    bool in_line_comment = false;
    
    for (size_t i = 0; i < source.length(); i++) {
        if (!in_line_comment && i + 1 < source.length() && 
            source[i] == '/' && source[i+1] == '*') {
            in_block_comment = true;
            i++; // 璺宠繃 '*'
            continue;
        }
        
        if (in_block_comment && i + 1 < source.length() && 
            source[i] == '*' && source[i+1] == '/') {
            in_block_comment = false;
            i++; // 璺宠繃 '/'
            continue;
        }
        
        if (!in_block_comment && i + 1 < source.length() && 
            source[i] == '/' && source[i+1] == '/') {
            in_line_comment = true;
            i++; // 璺宠繃绗簩涓?'/'
            continue;
        }
        
        if (in_line_comment && source[i] == '\n') {
            in_line_comment = false;
            continue;
        }
        
        if (!in_block_comment && !in_line_comment) {
            if (i + 5 < source.length() && 
                source.substr(i, 6) == "kernel") {
                
                size_t pos = i + 6;
                
                while (pos < source.length() && std::isspace(source[pos])) {
                    pos++;
                }
                
                if (pos + 3 < source.length() && source.substr(pos, 4) == "void") {
                    pos += 4; // 璺宠繃 "void"
                    
                    while (pos < source.length() && std::isspace(source[pos])) {
                        pos++;
                    }
                    
                    size_t name_start = pos;
                    while (pos < source.length() && 
                           (std::isalnum(source[pos]) || source[pos] == '_')) {
                        pos++;
                    }
                    
                    if (pos > name_start) {
                        std::string kernel_name = source.substr(name_start, pos - name_start);
                        kernels.push_back(kernel_name);
                        // POWERSERVE_LOG_DEBUG("Found kernel: {}", kernel_name);
                        i = pos - 1; // 缁х画浠庡綋鍓嶄綅缃悳绱?
                    }
                }
            }
        }
    }
    
    if (kernels.empty()) {
        size_t pos = 0;
        while ((pos = source.find("kernel_", pos)) != std::string::npos) {
            bool is_commented = false;
            
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
            pos += 7; // "kernel_"鐨勯暱搴?
        }
    }
    
    return kernels;
}

} // namespace powerserve::opencl
