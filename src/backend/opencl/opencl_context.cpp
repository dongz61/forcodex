// opencl_context.cpp
#include "opencl_context.hpp"
#include "core/logger.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace powerserve::opencl {

OpenCLContext::OpenCLContext() {
    parse_environment_variables();
}

OpenCLContext::~OpenCLContext() {
    cleanup();
}

void OpenCLContext::parse_environment_variables() {
    char* platform_str = std::getenv("GGML_OPENCL_PLATFORM");
    char* device_str = std::getenv("GGML_OPENCL_DEVICE");
    
    if (platform_str) {
        std::string platform_val(platform_str);
        // 检查是否为数字
        if (std::all_of(platform_val.begin(), platform_val.end(), ::isdigit)) {
            env_platform_num_ = std::atoi(platform_val.c_str());
        } else {
            env_platform_ = platform_val;
        }
    }
    
    if (device_str) {
        std::string device_val(device_str);
        if (std::all_of(device_val.begin(), device_val.end(), ::isdigit)) {
            env_device_num_ = std::atoi(device_val.c_str());
        } else {
            env_device_ = device_val;
        }
    }
}

bool OpenCLContext::initialize(const std::string& preferred_device) {
    // 发现所有可用设备
    auto devices = discover_devices();
    if (devices.empty()) {
        POWERSERVE_LOG_ERROR("No OpenCL devices found");
        return false;
    }
    
    // 设备选择逻辑
    bool selected = false;
    
    // 1. 首先检查环境变量
    if (!env_platform_.empty() || env_platform_num_ != -1 || 
        !env_device_.empty() || env_device_num_ != -1) {
        selected = select_by_environment(devices);
    }
    
    // 2. 检查参数指定的设备
    if (!selected && !preferred_device.empty()) {
        selected = select_device_by_name(preferred_device);
    }
    
    // 3. 选择默认设备（GPU优先）
    if (!selected) {
        selected = select_default_device(devices);
    }
    
    if (!selected) {
        POWERSERVE_LOG_ERROR("Failed to select OpenCL device");
        return false;
    }
    
    // 创建上下文和命令队列
    if (!create_context_and_queue()) {
        return false;
    }
    
    // 查询设备信息
    if (!query_device_info(device_)) {
        return false;
    }
    
    POWERSERVE_LOG_INFO("OpenCL initialized: {} on platform {}", 
                       device_info_.name, device_info_.platform_name);
    
    return true;
}

std::vector<OpenCLDeviceInfo> OpenCLContext::discover_devices() const {
    std::vector<OpenCLDeviceInfo> devices;
    
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        POWERSERVE_LOG_ERROR("Failed to get OpenCL platforms: {}", get_error_string(err));
        return devices;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to get platform IDs: {}", get_error_string(err));
        return devices;
    }
    
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platform_name[128] = {0};
        char platform_vendor[128] = {0};
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
        
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }
        
        std::vector<cl_device_id> device_ids(num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }
        
        for (cl_uint j = 0; j < num_devices; ++j) {
            OpenCLDeviceInfo info;
            info.platform_id = platforms[i];
            info.platform_name = platform_name;
            info.platform_vendor = platform_vendor;
            info.id = device_ids[j];
            
            // 获取设备基本信息
            char device_name[128] = {0};
            char device_version[128] = {0};
            
            clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_VERSION, sizeof(device_version), device_version, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE, sizeof(info.type), &info.type, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.global_mem_size), 
                          &info.global_mem_size, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info.local_mem_size), 
                          &info.local_mem_size, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info.max_work_group_size), 
                          &info.max_work_group_size, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.max_compute_units), 
                          &info.max_compute_units, nullptr);
            clGetDeviceInfo(device_ids[j], CL_DEVICE_AVAILABLE, sizeof(info.available), 
                          &info.available, nullptr);
            
            info.name = device_name;
            info.version = device_version;
            
            devices.push_back(info);
            
            POWERSERVE_LOG_DEBUG("Found OpenCL device: {} (Platform: {})", device_name, platform_name);
        }
    }
    
    return devices;
}

bool OpenCLContext::select_by_environment(const std::vector<OpenCLDeviceInfo>& devices) {
    std::vector<OpenCLDeviceInfo> filtered = devices;
    
    // 根据环境变量过滤设备
    if (!env_platform_.empty()) {
        auto it = std::remove_if(filtered.begin(), filtered.end(),
            [this](const OpenCLDeviceInfo& info) {
                return info.platform_name.find(env_platform_) == std::string::npos &&
                       info.platform_vendor.find(env_platform_) == std::string::npos;
            });
        filtered.erase(it, filtered.end());
    }
    
    if (env_platform_num_ != -1 && env_platform_num_ < static_cast<int>(filtered.size())) {
        // 选择特定平台的设备
        cl_platform_id target_platform = filtered[0].platform_id; // 简化处理
        auto it = std::remove_if(filtered.begin(), filtered.end(),
            [target_platform](const OpenCLDeviceInfo& info) {
                return info.platform_id != target_platform;
            });
        filtered.erase(it, filtered.end());
    }
    
    if (!env_device_.empty()) {
        auto it = std::remove_if(filtered.begin(), filtered.end(),
            [this](const OpenCLDeviceInfo& info) {
                return info.name.find(env_device_) == std::string::npos;
            });
        filtered.erase(it, filtered.end());
    }
    
    if (env_device_num_ != -1 && env_device_num_ < static_cast<int>(filtered.size())) {
        device_ = filtered[env_device_num_].id;
        platform_ = filtered[env_device_num_].platform_id;
        return true;
    }
    
    if (!filtered.empty()) {
        device_ = filtered[0].id;
        platform_ = filtered[0].platform_id;
        return true;
    }
    
    return false;
}

bool OpenCLContext::select_default_device(const std::vector<OpenCLDeviceInfo>& devices) {
    // 优先选择GPU
    for (const auto& device : devices) {
        if (device.type == CL_DEVICE_TYPE_GPU && device.available) {
            device_ = device.id;
            platform_ = device.platform_id;
            return true;
        }
    }
    
    // 如果没有GPU，选择第一个可用的设备
    for (const auto& device : devices) {
        if (device.available) {
            device_ = device.id;
            platform_ = device.platform_id;
            return true;
        }
    }
    
    // 最后选择任何设备
    if (!devices.empty()) {
        device_ = devices[0].id;
        platform_ = devices[0].platform_id;
        return true;
    }
    
    return false;
}

bool OpenCLContext::create_context_and_queue() {
    if (!device_ || !platform_) {
        return false;
    }
    
    cl_int err = CL_SUCCESS;
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform_),
        0
    };
    
    // 创建上下文
    context_ = clCreateContext(properties, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to create OpenCL context: {}", get_error_string(err));
        return false;
    }
    
    // 创建命令队列
    cl_command_queue_properties queue_props = 0;
#ifdef POWERSERVE_OPENCL_PROFILING
    queue_props = CL_QUEUE_PROFILING_ENABLE;
#endif
    
    queue_ = clCreateCommandQueue(context_, device_, queue_props, &err);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to create command queue: {}", get_error_string(err));
        clReleaseContext(context_);
        context_ = nullptr;
        return false;
    }
    
    return true;
}

bool OpenCLContext::query_device_info(cl_device_id device) {
    if (!device) return false;
    
    char name[256] = {0};
    char version[256] = {0};
    char platform_name[256] = {0};
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, nullptr);
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
    
    device_info_.id = device;
    device_info_.platform_id = platform_;
    device_info_.name = name;
    device_info_.version = version;
    device_info_.platform_name = platform_name;
    
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_info_.type), &device_info_.type, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info_.global_mem_size), 
                   &device_info_.global_mem_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_info_.local_mem_size), 
                   &device_info_.local_mem_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_info_.max_work_group_size), 
                   &device_info_.max_work_group_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_info_.max_compute_units), 
                   &device_info_.max_compute_units, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(device_info_.available), 
                   &device_info_.available, nullptr);
    
    return true;
}

void OpenCLContext::cleanup() {
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    device_ = nullptr;
    platform_ = nullptr;
}

cl_int OpenCLContext::flush() const {
    if (!queue_) return CL_INVALID_COMMAND_QUEUE;
    return clFlush(queue_);
}

cl_int OpenCLContext::finish() const {
    if (!queue_) return CL_INVALID_COMMAND_QUEUE;
    return clFinish(queue_);
}

std::string OpenCLContext::get_error_string(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "Unknown OpenCL error";
    }
}

bool OpenCLContext::check_error(cl_int error, const std::string& operation) const {
    if (error != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("OpenCL {} failed: {}", operation, get_error_string(error));
        return false;
    }
    return true;
}

bool OpenCLContext::select_device_by_name(const std::string& device_name) {
    auto devices = discover_devices();
    
    for (const auto& device : devices) {
        if (device.name.find(device_name) != std::string::npos) {
            device_ = device.id;
            platform_ = device.platform_id;
            return true;
        }
    }
    
    return false;
}

bool OpenCLContext::select_device(size_t device_index) {
    auto devices = discover_devices();
    
    if (device_index >= devices.size()) {
        return false;
    }
    
    device_ = devices[device_index].id;
    platform_ = devices[device_index].platform_id;
    return true;
}

bool OpenCLContext::select_device_by_type(cl_device_type type) {
    auto devices = discover_devices();
    
    for (const auto& device : devices) {
        if (device.type == type && device.available) {
            device_ = device.id;
            platform_ = device.platform_id;
            return true;
        }
    }
    
    return false;
}

} // namespace powerserve::opencl