// opencl_buffer.cpp

#include "backend/opencl/opencl_buffer.hpp"

#include "core/logger.hpp"

#include <CL/cl.h>
#include <cstring>
#include <utility>

namespace powerserve::opencl {

OpenCLBuffer::OpenCLBuffer(Stride stride,
                           cl_mem device_buffer,
                           size_t size,
                           std::shared_ptr<OpenCLMemoryPool> pool,
                           bool owns_buffer,
                           bool is_pooled,
                           size_t base_offset) :
    m_stride(stride),
    m_device_buffer(device_buffer),
    m_size(size),
    m_base_offset(base_offset),
    memory_pool(std::move(pool)),
    m_owns_buffer(owns_buffer),
    m_is_pooled(is_pooled) {
}

OpenCLBuffer::~OpenCLBuffer() {
    if (!m_owns_buffer || !m_device_buffer) return;

    if (memory_pool) {
        if (m_is_pooled) memory_pool->free_pooled(m_device_buffer);
        else             memory_pool->free(m_device_buffer);
        m_device_buffer = nullptr;
    } else {
        // 兜底：没有 pool 时避免泄漏
        clReleaseMemObject(m_device_buffer);
        m_device_buffer = nullptr;
    }
}

OpenCLBuffer::OpenCLBuffer(OpenCLBuffer&& other) noexcept :
    m_stride(other.m_stride),
    m_device_buffer(other.m_device_buffer),
    m_size(other.m_size),
    m_base_offset(other.m_base_offset),
    memory_pool(std::move(other.memory_pool)),
    m_owns_buffer(other.m_owns_buffer),
    m_is_pooled(other.m_is_pooled) {
    other.m_device_buffer = nullptr;
    other.m_owns_buffer   = false;
    other.m_base_offset   = 0;
}

OpenCLBuffer& OpenCLBuffer::operator=(OpenCLBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (m_owns_buffer && m_device_buffer) {
        if (m_is_pooled) {
            memory_pool->free_pooled(m_device_buffer);
        } else {
            // 兜底：没有 pool 时避免泄漏
            clReleaseMemObject(m_device_buffer);
        }
    }
    m_device_buffer = nullptr;

    // Move assign
    m_stride        = other.m_stride;
    m_device_buffer = other.m_device_buffer;
    m_size          = other.m_size;
    memory_pool     = std::move(other.memory_pool);
    m_owns_buffer   = other.m_owns_buffer;
    m_is_pooled     = other.m_is_pooled;
    m_base_offset   = other.m_base_offset;

    other.m_device_buffer = nullptr;
    other.m_owns_buffer   = false;
    other.m_base_offset   = 0;

    POWERSERVE_LOG_DEBUG("OpenCLBuffer move-assigned");
    return *this;
}

bool OpenCLBuffer::copy_to_device(const void* host_data, size_t size) {
    if (!memory_pool || !m_device_buffer) {
        POWERSERVE_LOG_ERROR("Invalid buffer or memory pool for copy_to_device");
        return false;
    }
    if (size > m_size) {
        POWERSERVE_LOG_ERROR("Copy size {} exceeds buffer size {}", size, m_size);
        return false;
    }
    return memory_pool->copy_host_to_device(m_device_buffer, host_data, size, m_base_offset);
}

bool OpenCLBuffer::copy_to_host(void* host_data, size_t size) {
    if (!memory_pool || !m_device_buffer || !host_data) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_to_host");
        return false;
    }
    if (size > m_size) {
        POWERSERVE_LOG_ERROR("Copy size {} exceeds buffer size {}", size, m_size);
        return false;
    }
    return memory_pool->copy_device_to_host(host_data, m_device_buffer, size, m_base_offset);
}

bool OpenCLBuffer::copy_to_device_async(const void* host_data, size_t size, cl_event* event) {
    if (!memory_pool || !m_device_buffer) {
        POWERSERVE_LOG_ERROR("Invalid buffer or memory pool for copy_to_device_async");
        return false;
    }
    if (size > m_size) {
        POWERSERVE_LOG_ERROR("Copy size {} exceeds buffer size {}", size, m_size);
        return false;
    }
    return memory_pool->copy_host_to_device_async(m_device_buffer, host_data, size, event, m_base_offset);
}

bool OpenCLBuffer::copy_to_host_async(void* host_data, size_t size, cl_event* event) {
    if (!memory_pool || !m_device_buffer || !host_data) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_to_host_async");
        return false;
    }
    if (size > m_size) {
        POWERSERVE_LOG_ERROR("Copy size {} exceeds buffer size {}", size, m_size);
        return false;
    }
    return memory_pool->copy_device_to_host_async(host_data, m_device_buffer, size, event, m_base_offset);
}

void* OpenCLBuffer::map(cl_map_flags flags, size_t offset, size_t size) {
    if (!memory_pool || !m_device_buffer) {
        POWERSERVE_LOG_ERROR("Invalid buffer or memory pool for map");
        return nullptr;
    }
    if (size == 0) {
        if (offset > m_size) {
            POWERSERVE_LOG_ERROR("Map offset {} exceeds buffer size {}", offset, m_size);
            return nullptr;
        }
        size = m_size - offset;
    }
    if (offset + size > m_size) {
        POWERSERVE_LOG_ERROR("Map range [{}, {}] exceeds buffer size {}", offset, offset + size, m_size);
        return nullptr;
    }
    return memory_pool->map_memory(m_device_buffer, m_base_offset + offset, size, flags);
}

bool OpenCLBuffer::unmap(void* mapped_ptr) {
    if (!memory_pool || !m_device_buffer || !mapped_ptr) {
        POWERSERVE_LOG_ERROR("Invalid arguments for unmap");
        return false;
    }
    return memory_pool->unmap_memory(m_device_buffer, mapped_ptr);
}

void OpenCLBuffer::finish() {
    if (memory_pool) {
        memory_pool->finish_all_operations();
    }
}

bool OpenCLBuffer::clear() {
    if (!memory_pool || !m_device_buffer) {
        return false;
    }
    void* mapped = map(CL_MAP_WRITE);
    if (!mapped) {
        return false;
    }
    std::memset(mapped, 0, m_size);
    return unmap(mapped);
}

} // namespace powerserve::opencl
