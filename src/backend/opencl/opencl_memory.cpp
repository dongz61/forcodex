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
    }
    
    return buffer;
}

// Public API: allocate memory (non-pooled)
cl_mem OpenCLMemoryPool::allocate(size_t size, cl_mem_flags flags) {
    size = align_size(size);
    return allocate_impl(size, flags, true);
}

// Public API: allocate pooled memory
cl_mem OpenCLMemoryPool::allocate_pooled(size_t size, cl_mem_flags flags) {
    size = align_size(size);
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // First, try to reuse a suitable free block from pool.
    cl_mem buffer = find_suitable_pool_entry(size);
    if (buffer) {
        return buffer;
    }
    
    // Release lock, then allocate a new block.
    lock.unlock();
    buffer = allocate_impl(size, flags, false);
    lock.lock();
    
    if (!buffer) {
        POWERSERVE_LOG_ERROR("Failed to allocate pooled buffer of size {}", size);
        return nullptr;
    }
    
    // Update memory statistics.
    total_allocated_ += size;
    update_peak_usage();
    
    // Add the new block into the memory pool.
    MemoryBlock block;
    block.buffer = buffer;
    block.size = size;
    block.in_use = true;
    block.allocation_id = next_allocation_id_++;
    
    memory_pool_.push_back(block);
    
    return buffer;
}

cl_mem OpenCLMemoryPool::find_suitable_pool_entry(size_t size) {
    // Best-fit strategy: find the smallest free block that fits.
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
    
    // Check whether this buffer belongs to pool.
    auto it = std::find_if(memory_pool_.begin(), memory_pool_.end(),
        [buffer](const MemoryBlock& block) {
            return block.buffer == buffer;
        });
    
    if (it != memory_pool_.end()) {
        // In pool: record size and release.
        size_t freed_size = it->size;
        clReleaseMemObject(buffer);
        memory_pool_.erase(it);
        total_allocated_ -= freed_size;
        
    } else {
        // Not in pool: release directly.
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
    
    // Warn if buffer is not found in pool.
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

    cl_int err = clEnqueueWriteBuffer(context_->get_queue(), dst, CL_TRUE,
                                  offset, size, src, 0, nullptr, nullptr);
    if (!context_->check_error(err, "clEnqueueWriteBuffer")) return false;

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


bool OpenCLMemoryPool::copy_device_to_device(
    cl_mem dst,
    cl_mem src,
    size_t size,
    size_t dst_offset,
    size_t src_offset) {
    if (!dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_device_to_device");
        return false;
    }
    if (size == 0) {
        return true;
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

    if (src_offset + size > src_size || dst_offset + size > dst_size) {
        POWERSERVE_LOG_ERROR("D2D OOB: src_off={} dst_off={} size={} src_size={} dst_size={}",
                             src_offset, dst_offset, size, src_size, dst_size);
        return false;
    }


    cl_int err = clEnqueueCopyBuffer(context_->get_queue(), src, dst,
                                     src_offset, dst_offset, size, 0, nullptr, nullptr);
    return context_->check_error(err, "clEnqueueCopyBuffer");
}

void OpenCLMemoryPool::update_peak_usage() {
    if (total_allocated_ > peak_usage_) {
        peak_usage_ = total_allocated_;
        // POWERSERVE_LOG_DEBUG("New peak memory usage: {} bytes", peak_usage_);
    }
}

} // namespace powerserve::opencl
