// opencl_memory.cpp
#include "opencl_memory.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace powerserve::opencl {

OpenCLMemoryPool::OpenCLMemoryPool(std::shared_ptr<OpenCLContext> context)
    : context_(context) {
    POWERSERVE_LOG_DEBUG("OpenCLMemoryPool created");
}

OpenCLMemoryPool::~OpenCLMemoryPool() {
    clear_pool();
    POWERSERVE_LOG_DEBUG("OpenCLMemoryPool destroyed");
}

// 私有实现：实际分配内存
cl_mem OpenCLMemoryPool::allocate_impl(size_t size, cl_mem_flags flags, bool update_stats) {
    if (size == 0) {
        POWERSERVE_LOG_WARN("Attempting to allocate zero-sized buffer, using size 1");
        size = 1;
    }
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(context_->get_context(), flags, size, nullptr, &err);
    
    if (!context_->check_error(err, "clCreateBuffer")) {
        POWERSERVE_LOG_ERROR("Failed to allocate {} bytes", size);
        return nullptr;
    }
    
    if (update_stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        total_allocated_ += size;
        update_peak_usage();
        // POWERSERVE_LOG_DEBUG("Allocated {} bytes of OpenCL memory, total: {}", size, total_allocated_);
    }
    
    return buffer;
}

// 公共接口：分配内存（不池化）
cl_mem OpenCLMemoryPool::allocate(size_t size, cl_mem_flags flags) {
    size = align_size(size);
    return allocate_impl(size, flags, true);
}

// 公共接口：分配池化内存
cl_mem OpenCLMemoryPool::allocate_pooled(size_t size, cl_mem_flags flags) {
    size = align_size(size);
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 首先在内存池中寻找合适的块
    cl_mem buffer = find_suitable_pool_entry(size);
    if (buffer) {
        return buffer;
    }
    
    // 释放锁，分配新内存
    lock.unlock();
    buffer = allocate_impl(size, flags, false);
    lock.lock();
    
    if (!buffer) {
        POWERSERVE_LOG_ERROR("Failed to allocate pooled buffer of size {}", size);
        return nullptr;
    }
    
    // 更新统计信息
    total_allocated_ += size;
    update_peak_usage();
    
    // 添加到内存池
    MemoryBlock block;
    block.buffer = buffer;
    block.size = size;
    block.in_use = true;
    block.allocation_id = next_allocation_id_++;
    
    memory_pool_.push_back(block);
    
    return buffer;
}

cl_mem OpenCLMemoryPool::find_suitable_pool_entry(size_t size) {
    // 使用最佳适配策略：找到最小但足够大的空闲块
    MemoryBlock* best_fit = nullptr;
    
    for (auto& block : memory_pool_) {
        if (!block.in_use && block.size >= size) {
            if (!best_fit || block.size < best_fit->size) {
                best_fit = &block;
            }
        }
    }
    
    if (best_fit) {
        best_fit->in_use = true;
        return best_fit->buffer;
    }
    
    return nullptr;
}

void OpenCLMemoryPool::free(cl_mem buffer) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 查找buffer是否在池中
    auto it = std::find_if(memory_pool_.begin(), memory_pool_.end(),
        [buffer](const MemoryBlock& block) {
            return block.buffer == buffer;
        });
    
    if (it != memory_pool_.end()) {
        // 在池中：记录大小然后释放
        size_t freed_size = it->size;
        clReleaseMemObject(buffer);
        memory_pool_.erase(it);
        total_allocated_ -= freed_size;
        
    } else {
        // 不在池中：直接释放
        clReleaseMemObject(buffer);
    }
}

void OpenCLMemoryPool::free_pooled(cl_mem buffer) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : memory_pool_) {
        if (block.buffer == buffer) {
            block.in_use = false;
            return;
        }
    }
    
    // 如果不在池中，记录警告
    POWERSERVE_LOG_WARN("Attempted to free_pooled a buffer not in pool");
    clReleaseMemObject(buffer);
}

void OpenCLMemoryPool::clear_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    POWERSERVE_LOG_DEBUG("Clearing memory pool with {} buffers", memory_pool_.size());
    
    for (auto& block : memory_pool_) {
        clReleaseMemObject(block.buffer);
    }
    
    memory_pool_.clear();
    total_allocated_ = 0;
    peak_usage_ = 0;
    next_allocation_id_ = 0;
    
    POWERSERVE_LOG_DEBUG("Memory pool cleared");
}

// for copy
static inline bool get_mem_size(OpenCLContext* ctx, cl_mem mem, size_t* out_size) {
    if (!mem || !out_size) return false;
    cl_int err = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), out_size, nullptr);
    if (!ctx->check_error(err, "clGetMemObjectInfo(CL_MEM_SIZE)")) return false;
    return true;
}
// for copy end

bool OpenCLMemoryPool::copy_host_to_device(cl_mem dst, const void* src, size_t size, size_t offset) {

    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_host_to_device");
        return false;
    }
    if (size == 0) {
        POWERSERVE_LOG_WARN("copy_host_to_device: size == 0, skip");
        return true;
    }

    // query dst mem size
    size_t dst_size = 0;
    if (!get_mem_size(context_.get(), dst, &dst_size)) {
        POWERSERVE_LOG_ERROR("copy_host_to_device: failed to query dst mem size");
        return false;
    }

    // OOB check
    if (offset + size > dst_size) {
        POWERSERVE_LOG_ERROR("H2D OOB: offset+size={} > dst_size={}",
                             offset + size, dst_size);
        return false;
    }
    
    // volatile uint8_t probe = 0;
    // const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    // probe ^= p[0];
    // probe ^= p[size - 1];
    // (void)probe;
    // POWERSERVE_LOG_ERROR(">>> PROBE_END src={} size={}", src, size);

    cl_int err = clEnqueueWriteBuffer(context_->get_queue(), dst, CL_TRUE,
                                  offset, size, src, 0, nullptr, nullptr);
    if (!context_->check_error(err, "clEnqueueWriteBuffer")) return false;

    // ✅ 新增：强制 flush + finish，确保拷贝真正完成并同步 driver
    err = clFinish(context_->get_queue());
    if (!context_->check_error(err, "clFinish(after H2D)")) return false;

    return true;
}

bool OpenCLMemoryPool::copy_device_to_host(void* dst, cl_mem src, size_t size, size_t offset) {
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_device_to_host");
        return false;
    }

    // size check
    size_t src_size = 0;
    if (!get_mem_size(context_.get(), src, &src_size)) {
        POWERSERVE_LOG_ERROR("copy_device_to_host: failed to query src mem size");
        return false;
    }

    if (offset + size > src_size) {
        POWERSERVE_LOG_ERROR("D2H OOB: offset+size={} > src_size={}",
                             offset + size, src_size);
        return false;
    }

    cl_int err = clEnqueueReadBuffer(context_->get_queue(), src, CL_TRUE,
                                     offset, size, dst, 0, nullptr, nullptr);
    return context_->check_error(err, "clEnqueueReadBuffer");
}


bool OpenCLMemoryPool::copy_device_to_device(cl_mem dst, cl_mem src, size_t size) {
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_device_to_device");
        return false;
    }

    size_t src_size = 0, dst_size = 0;
    if (!get_mem_size(context_.get(), src, &src_size)) {
        POWERSERVE_LOG_ERROR("copy_device_to_device: failed to query src mem size");
        return false;
    }
    if (!get_mem_size(context_.get(), dst, &dst_size)) {
        POWERSERVE_LOG_ERROR("copy_device_to_device: failed to query dst mem size");
        return false;
    }

    if (size > src_size || size > dst_size) {
        POWERSERVE_LOG_ERROR("D2D OOB: size={} src_size={} dst_size={}",
                             size, src_size, dst_size);
        return false;
    }


    cl_int err = clEnqueueCopyBuffer(context_->get_queue(), src, dst,
                                     0, 0, size, 0, nullptr, nullptr);
    return context_->check_error(err, "clEnqueueCopyBuffer");
}

bool OpenCLMemoryPool::copy_host_to_device_async(cl_mem dst, const void* src, size_t size, 
                                                 cl_event* event, size_t offset) {
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_host_to_device_async");
        return false;
    }
    
    cl_int err = clEnqueueWriteBuffer(context_->get_queue(), dst, CL_FALSE, 
                                      offset, size, src, 0, nullptr, event);
    return context_->check_error(err, "clEnqueueWriteBuffer async");
}

bool OpenCLMemoryPool::copy_device_to_host_async(void* dst, cl_mem src, size_t size, 
                                                 cl_event* event, size_t offset) {
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_device_to_host_async");
        return false;
    }
    
    cl_int err = clEnqueueReadBuffer(context_->get_queue(), src, CL_FALSE, 
                                     offset, size, dst, 0, nullptr, event);
    return context_->check_error(err, "clEnqueueReadBuffer async");
}

void* OpenCLMemoryPool::map_memory(cl_mem buffer, size_t offset, size_t size, cl_map_flags flags) {
    if (!buffer) {
        POWERSERVE_LOG_ERROR("Invalid buffer for map_memory");
        return nullptr;
    }
    
    cl_int err;
    void* mapped_ptr = clEnqueueMapBuffer(context_->get_queue(), buffer, CL_TRUE, 
                                          flags, offset, size, 0, nullptr, nullptr, &err);
    
    if (!context_->check_error(err, "clEnqueueMapBuffer")) {
        return nullptr;
    }
    
    return mapped_ptr;
}

bool OpenCLMemoryPool::unmap_memory(cl_mem buffer, void* mapped_ptr) {
    if (!buffer || !mapped_ptr) {
        POWERSERVE_LOG_ERROR("Invalid arguments for unmap_memory");
        return false;
    }
    
    cl_int err = clEnqueueUnmapMemObject(context_->get_queue(), buffer, mapped_ptr, 
                                         0, nullptr, nullptr);
    return context_->check_error(err, "clEnqueueUnmapMemObject");
}

void OpenCLMemoryPool::update_peak_usage() {
    if (total_allocated_ > peak_usage_) {
        peak_usage_ = total_allocated_;
        // POWERSERVE_LOG_DEBUG("New peak memory usage: {} bytes", peak_usage_);
    }
}

void OpenCLMemoryPool::finish_all_operations() {
    cl_int err = clFinish(context_->get_queue());
    context_->check_error(err, "clFinish");
}

} // namespace powerserve::opencl