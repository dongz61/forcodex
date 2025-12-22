// opencl_context.hpp
#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#define CL_TARGET_OPENCL_VERSION 300

namespace powerserve::opencl {

struct OpenCLDeviceInfo {
    std::string name;
    std::string version;
    std::string platform_name;
    std::string platform_vendor;
    cl_device_type type;
    size_t global_mem_size;
    size_t local_mem_size;
    size_t max_work_group_size;
    cl_uint max_compute_units;
    cl_bool available;
    cl_device_id id;
    cl_platform_id platform_id;
};

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();
    
    // 初始化方法
    bool initialize(const std::string& preferred_device = "");
    void cleanup();
    
    // 设备发现和管理
    std::vector<OpenCLDeviceInfo> discover_devices() const;
    bool select_device(size_t device_index);
    bool select_device_by_name(const std::string& device_name);
    bool select_device_by_type(cl_device_type type);
    
    // 环境变量支持（仿照llama.cpp）
    void parse_environment_variables();
    
    // 获取OpenCL对象
    cl_context get_context() const { return context_; }
    cl_device_id get_device() const { return device_; }
    cl_command_queue get_queue() const { return queue_; }
    cl_platform_id get_platform() const { return platform_; }
    
    // 设备信息
    const OpenCLDeviceInfo& get_device_info() const { return device_info_; }
    size_t get_max_work_group_size() const { return device_info_.max_work_group_size; }
    size_t get_global_mem_size() const { return device_info_.global_mem_size; }
    std::string get_device_name() const { return device_info_.name; }
    
    // 命令队列操作
    cl_int flush() const;
    cl_int finish() const;
    
    // 错误处理
    static std::string get_error_string(cl_int error);
    bool check_error(cl_int error, const std::string& operation) const;
    
private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    OpenCLDeviceInfo device_info_;
    
    // 环境变量配置
    std::string env_platform_;
    std::string env_device_;
    int env_platform_num_ = -1;
    int env_device_num_ = -1;
    
    // 内部方法
    bool enumerate_platforms_and_devices();
    bool create_context_and_queue();
    bool query_device_info(cl_device_id device);
    bool setup_command_queue();
    
    // 设备选择逻辑
    bool select_default_device(const std::vector<OpenCLDeviceInfo>& devices);
    bool select_by_environment(const std::vector<OpenCLDeviceInfo>& devices);
    bool filter_devices_by_criteria(const std::vector<OpenCLDeviceInfo>& all_devices,
                                    std::vector<OpenCLDeviceInfo>& filtered_devices);
};

} // namespace powerserve::opencl