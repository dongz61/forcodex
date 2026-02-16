// opencl_memory.hpp
#pragma once

#include "opencl_context.hpp"
#include <CL/cl.h>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>

namespace powerserve::opencl {

struct MemoryBlock {
    cl_mem buffer = nullptr;
    size_t size = 0;
    bool in_use = false;
    size_t allocation_id = 0;
};

class OpenCLMemoryPool {
public:
    explicit OpenCLMemoryPool(std::shared_ptr<OpenCLContext> context);
    ~OpenCLMemoryPool();
    
    // 内存分配
    cl_mem allocate(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    
    // 内存池分配（仿照llama.cpp）
    cl_mem allocate_pooled(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void free(cl_mem buffer);
    void free_pooled(cl_mem buffer);
    void clear_pool();
    
    // 数据传输
    bool copy_host_to_device(cl_mem dst, const void* src, size_t size, size_t offset = 0);
    bool copy_device_to_host(void* dst, cl_mem src, size_t size, size_t offset = 0);
    bool copy_device_to_device(
        cl_mem dst,
        cl_mem src,
        size_t size,
        size_t dst_offset = 0,
        size_t src_offset = 0);
    
private:
    std::shared_ptr<OpenCLContext> context_;
    std::vector<MemoryBlock> memory_pool_;
    size_t total_allocated_ = 0;
    size_t peak_usage_ = 0;
    size_t next_allocation_id_ = 0;
    mutable std::mutex mutex_;
    
    // 私有方法
    cl_mem find_suitable_pool_entry(size_t size);
    cl_mem allocate_impl(size_t size, cl_mem_flags flags, bool update_stats);
    void update_peak_usage();
    
    // 辅助函数：对齐内存大小
    static size_t align_size(size_t size, size_t alignment = 256) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};

} // namespace powerserve::opencl
