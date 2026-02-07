// opencl_kernel_manager.cpp
#include "opencl_kernel_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#ifdef POWERSERVE_OPENCL_EMBED_KERNELS
#include "opencl_embedded_kernels.hpp"
#endif

namespace powerserve::opencl {

// 鏋勯€犲嚱鏁板拰鏋愭瀯鍑芥暟
OpenCLKernelManager::OpenCLKernelManager(std::shared_ptr<OpenCLContext> context)
    : context_(std::move(context)) {
}

OpenCLKernelManager::~OpenCLKernelManager() {
    cleanup();
}

// 鍒濆鍖栨柟娉?
bool OpenCLKernelManager::initialize(const OpenCLCompileOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    compile_options_ = options;
    
    // 缂栬瘧宓屽叆寮忓唴鏍?
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
    
    // 1. 缂栬瘧 copy 鍐呮牳
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
    
    // 2. 缂栬瘧 add 鍐呮牳
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

    // 3. 缂栬瘧 silu 鍐呮牳
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

    // 5. 缂栬瘧 matmul 鍐呮牳
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

     // 5.1 缂栬瘧閫氱敤 simple quant matmul kernels锛堟棤 subgroups锛孨VIDIA 鍙敤锛?
#ifdef OPENCL_MUL_MAT_Q4_0_F32_SIMPLE_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mat_q4_0_f32_simple_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mat_q4_0_f32_simple_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mat_q4_0_f32_simple kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mat_q4_0_f32_simple kernel source is empty!");
            all_success = false;
        }
    }
#endif

#ifdef OPENCL_MUL_MAT_Q8_0_F32_SIMPLE_CL_AVAILABLE
    {
        const std::string& src = ::powerserve::opencl::embedded::mul_mat_q8_0_f32_simple_cl_source;
        if (!src.empty()) {
            if (!compile_program("mul_mat_q8_0_f32_simple_kernels", src)) {
                POWERSERVE_LOG_ERROR("Failed to compile mul_mat_q8_0_f32_simple kernels");
                all_success = false;
            }
        } else {
            POWERSERVE_LOG_ERROR("mul_mat_q8_0_f32_simple kernel source is empty!");
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

    // 6.1 compile additional matmul kernels
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

    // 6. compile rms_norm kernel
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

    // 7. 缂栬瘧 softmax 鍐呮牳
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

    // 8. 缂栬瘧 rope 鍐呮牳
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

    // 9. 缂栬瘧 get_rows 鍐呮牳
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

    // 10. 缂栬瘧 diag_mask_inf 鍐呮牳
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
    return true; // 涓嶈涓洪敊璇?
#endif // POWERSERVE_OPENCL_EMBED_KERNELS
}

// 缂栬瘧program锛堟牳蹇冩柟娉曪級
bool OpenCLKernelManager::compile_program(const std::string& program_name,
                                         const std::string& source_code,
                                         const std::string& extra_options) {
    
    
    // 妫€鏌ユ槸鍚﹀凡瀛樺湪
    if (programs_.find(program_name) != programs_.end()) {
        POWERSERVE_LOG_WARN("Program '{}' already compiled", program_name);
        return true;
    }
    
    // 妫€鏌ユ簮浠ｇ爜鏄惁涓虹┖
    if (source_code.empty()) {
        POWERSERVE_LOG_ERROR("Empty source code for program: {}", program_name);
        return false;
    }
    
    // 鏋勫缓缂栬瘧閫夐」
    std::string options = build_compile_options(extra_options);
    
    // 缂栬瘧program
    cl_program program = compile_program_impl(source_code, options);
    
    if (!program) {
        POWERSERVE_LOG_ERROR("Failed to compile program '{}'", program_name);
        return false;
    }
    
    std::vector<std::string> kernel_names = split_kernel_names(source_code);
    
    // 濡傛灉娌℃壘鍒帮紝杈撳嚭婧愮爜鐗囨甯姪璋冭瘯
    if (kernel_names.empty()) {
        POWERSERVE_LOG_WARN("No kernels found in program: {}", program_name);
        
        // 杈撳嚭婧愮爜鍓嶅嚑琛岀湅鐪嬫牸寮?
        std::istringstream source_stream(source_code);
        std::string line;
        int line_count = 0;
        POWERSERVE_LOG_DEBUG("First 10 lines of source:");
        while (std::getline(source_stream, line) && line_count < 10) {
            POWERSERVE_LOG_DEBUG("  Line {}: {}", line_count + 1, line);
            line_count++;
        }
        
        // 鏌ユ壘鍙兘鐨刱ernel瀹氫箟
        size_t kernel_pos = source_code.find("kernel");
        if (kernel_pos != std::string::npos) {
            size_t sample_start = (kernel_pos > 50) ? kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found 'kernel' at position {}, sample:", kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
        
        // 涔熸煡鎵惧甫涓嬪垝绾跨殑鐗堟湰
        size_t underscore_kernel_pos = source_code.find("__kernel");
        if (underscore_kernel_pos != std::string::npos) {
            size_t sample_start = (underscore_kernel_pos > 50) ? underscore_kernel_pos - 50 : 0;
            size_t sample_end = std::min(source_code.length(), underscore_kernel_pos + 100);
            // POWERSERVE_LOG_DEBUG("Found '__kernel' at position {}, sample:", underscore_kernel_pos);
            // POWERSERVE_LOG_DEBUG("  ...{}...", source_code.substr(sample_start, sample_end - sample_start));
        }
    } else {
        // 杈撳嚭鎵惧埌鐨勫唴鏍稿悕
        // for (const auto& kernel_name : kernel_names) {
        //     POWERSERVE_LOG_DEBUG("  Kernel: {}", kernel_name);
        // }
    }
    
    // 涓烘瘡涓猭ernel鍒涘缓cl_kernel瀵硅薄
    std::unordered_map<std::string, cl_kernel> kernels;
    for (const auto& kernel_name : kernel_names) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to create kernel '{}': {}", 
                               kernel_name, context_->get_error_string(err));
            // 缁х画灏濊瘯鍏朵粬kernels
            continue;
        }
        
        kernels[kernel_name] = kernel;
        
        // 鍚屾椂娣诲姞鍒発ernel_cache_
        KernelCacheItem cache_item;
        cache_item.kernel = kernel;
        cache_item.name = kernel_name;
        cache_item.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        kernel_cache_[kernel_name] = cache_item;
        
        // POWERSERVE_LOG_DEBUG("Created kernel: {}", kernel_name);
    }
    
    // 鍒涘缓缂撳瓨椤?
    ProgramCacheItem item;
    item.program = program;
    item.source_hash = compute_source_hash(source_code);
    item.kernels = std::move(kernels);
    
    programs_[program_name] = std::move(item);
    
    return true;
}

// 浠巔rogram涓彁鍙栨墍鏈塳ernels锛堜豢鐓lama.cpp妯″紡锛?
bool OpenCLKernelManager::extract_kernels_from_program(cl_program program,
                                                      const std::string& program_name) {
    // 杩欓噷鍙互鍒嗘瀽婧愮爜鑷姩鎻愬彇kernel鍚嶏紝鎴栬€呴瀹氫箟
    // 瀵逛簬绠€鍗曟儏鍐碉紝鎴戜滑鍙互璁╄皟鐢ㄨ€呮寚瀹氳鎻愬彇鐨刱ernels
    
    // 涓存椂鏂规锛氬厛涓嶈嚜鍔ㄦ彁鍙栵紝闇€瑕佹墜鍔ㄩ€氳繃get_kernel鍒涘缓
    return true;
}

// 鑾峰彇鍐呮牳锛堝鏋滄病鏈夊垯浠巔rogram涓垱寤猴級
cl_kernel OpenCLKernelManager::get_kernel(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 妫€鏌ョ紦瀛?
    auto it = kernel_cache_.find(kernel_name);
    if (it != kernel_cache_.end()) {
        it->second.last_used = std::chrono::steady_clock::now().time_since_epoch().count();
        return it->second.kernel;
    }
    
    // 闇€瑕佺煡閬撹繖涓猭ernel灞炰簬鍝釜program
    // 杩欓噷闇€瑕佷竴涓槧灏勶細kernel_name -> program_name
    // 鏆傛椂绠€鍖栵細鍋囪program鍚嶅氨鏄痥ernel鐨勫墠缂€锛堝"add" -> "kernel_add"锛?
    
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

// 淇敼 cleanup 鍑芥暟
void OpenCLKernelManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 閲婃斁鎵€鏈塳ernels - 浣跨敤 kernel_cache_
    for (auto& [name, item] : kernel_cache_) {
        if (item.kernel) {
            clReleaseKernel(item.kernel);
        }
    }
    kernel_cache_.clear();
    
    // 閲婃斁鎵€鏈塸rograms
    for (auto& [name, item] : programs_) {
        if (item.program) {
            clReleaseProgram(item.program);
        }
    }
    programs_.clear();
    
    embedded_sources_.clear();
}

// 鏋勫缓缂栬瘧閫夐」
std::string OpenCLKernelManager::build_compile_options(const std::string& extra_options) const {
    std::string options = compile_options_.to_string();
    if (!extra_options.empty()) {
        options += " " + extra_options;
    }
    return options;
}

// 缂栬瘧program瀹炵幇
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
        // 鑾峰彇鏋勫缓鏃ュ織
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

// 璁＄畻婧愮爜鍝堝笇
std::string OpenCLKernelManager::compute_source_hash(const std::string& source) {
    // 绠€鍗曞疄鐜帮細浣跨敤瀛楃涓查暱搴﹀拰閮ㄥ垎鍐呭浣滀负鍝堝笇
    std::hash<std::string> hasher;
    return std::to_string(hasher(source));
}

// 妫€鏌ユ瀯寤洪敊璇?
bool OpenCLKernelManager::check_build_error(cl_program program, cl_device_id device) const {
    cl_build_status status;
    cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                                      sizeof(status), &status, nullptr);
    return (err == CL_SUCCESS && status == CL_BUILD_SUCCESS);
}

// 鑾峰彇鏋勫缓鏃ュ織
std::string OpenCLKernelManager::get_program_build_log(cl_program program) const {
    cl_device_id device = context_->get_device();
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    
    return std::string(log.data());
}

// 鍒嗗壊鍐呮牳鍚?
std::vector<std::string> OpenCLKernelManager::split_kernel_names(const std::string& source) {
    std::vector<std::string> kernels;
    
    // 鏇存櫤鑳界殑鎼滅储锛氳烦杩囨敞閲?
    bool in_block_comment = false;
    bool in_line_comment = false;
    
    for (size_t i = 0; i < source.length(); i++) {
        // 澶勭悊鍧楁敞閲?/* */
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
        
        // 澶勭悊琛屾敞閲?//
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
        
        // 濡傛灉涓嶅湪娉ㄩ噴涓紝鏌ユ壘 kernel 鍏抽敭瀛?
        if (!in_block_comment && !in_line_comment) {
            // 鏌ユ壘 "kernel" 鍏抽敭瀛?
            if (i + 5 < source.length() && 
                source.substr(i, 6) == "kernel") {
                
                // 璺宠繃 "kernel" 鍏抽敭瀛?
                size_t pos = i + 6;
                
                // 璺宠繃绌虹櫧
                while (pos < source.length() && std::isspace(source[pos])) {
                    pos++;
                }
                
                // 妫€鏌ユ槸鍚︽槸 "void"锛坘ernel void xxx锛?
                if (pos + 3 < source.length() && source.substr(pos, 4) == "void") {
                    pos += 4; // 璺宠繃 "void"
                    
                    // 璺宠繃绌虹櫧
                    while (pos < source.length() && std::isspace(source[pos])) {
                        pos++;
                    }
                    
                    // 鎻愬彇鍐呮牳鍚?
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
        // 澶囩敤鏂规硶锛氱洿鎺ユ悳绱?kernel_ 寮€澶寸殑鍑芥暟鍚?
        size_t pos = 0;
        while ((pos = source.find("kernel_", pos)) != std::string::npos) {
            // 妫€鏌ュ墠闈㈡槸鍚︽湁娉ㄩ噴
            bool is_commented = false;
            
            // 妫€鏌ュ墠闈㈡槸鍚︽湁 //
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




