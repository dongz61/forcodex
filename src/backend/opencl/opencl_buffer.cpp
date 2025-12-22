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
                           bool is_subbuffer) :
    m_stride(stride),
    m_device_buffer(device_buffer),
    m_size(size),
    memory_pool(std::move(pool)),
    m_owns_buffer(owns_buffer),
    m_is_pooled(is_pooled),
    m_is_subbuffer(is_subbuffer) {
    POWERSERVE_LOG_DEBUG(
        "OpenCLBuffer created: size={} bytes, pooled={}, owns={}, subbuffer={}",
        m_size, m_is_pooled, m_owns_buffer, m_is_subbuffer
    );
}

OpenCLBuffer::~OpenCLBuffer() {
    POWERSERVE_LOG_DEBUG(
        "OpenCLBuffer destroying: size={} bytes, pooled={}, owns={}, subbuffer={}",
        m_size, m_is_pooled, m_owns_buffer, m_is_subbuffer
    );

    if (!m_owns_buffer || !m_device_buffer) {
        return;
    }

    // Sub-buffer must be released via OpenCL API, not memory pool.
    if (m_is_subbuffer) {
        clReleaseMemObject(m_device_buffer);
        m_device_buffer = nullptr;
        return;
    }

    // Normal pooled/non-pooled buffers are managed by memory pool.
    if (memory_pool) {
        if (m_is_pooled) {
            memory_pool->free_pooled(m_device_buffer);
        } else {
            memory_pool->free(m_device_buffer);
        }
        m_device_buffer = nullptr;
    }
}

OpenCLBuffer::OpenCLBuffer(OpenCLBuffer&& other) noexcept :
    m_stride(other.m_stride),
    m_device_buffer(other.m_device_buffer),
    m_size(other.m_size),
    memory_pool(std::move(other.memory_pool)),
    m_owns_buffer(other.m_owns_buffer),
    m_is_pooled(other.m_is_pooled),
    m_is_subbuffer(other.m_is_subbuffer) {
    other.m_device_buffer = nullptr;
    other.m_owns_buffer   = false;
    other.m_is_subbuffer  = false;
    POWERSERVE_LOG_DEBUG("OpenCLBuffer move-constructed");
}

OpenCLBuffer& OpenCLBuffer::operator=(OpenCLBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    // Release current
    this->~OpenCLBuffer();

    // Move assign
    m_stride        = other.m_stride;
    m_device_buffer = other.m_device_buffer;
    m_size          = other.m_size;
    memory_pool     = std::move(other.memory_pool);
    m_owns_buffer   = other.m_owns_buffer;
    m_is_pooled     = other.m_is_pooled;
    m_is_subbuffer  = other.m_is_subbuffer;

    other.m_device_buffer = nullptr;
    other.m_owns_buffer   = false;
    other.m_is_subbuffer  = false;

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
    return memory_pool->copy_host_to_device(m_device_buffer, host_data, size);
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
    return memory_pool->copy_device_to_host(host_data, m_device_buffer, size);
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
    return memory_pool->copy_host_to_device_async(m_device_buffer, host_data, size, event);
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
    return memory_pool->copy_device_to_host_async(host_data, m_device_buffer, size, event);
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
    return memory_pool->map_memory(m_device_buffer, offset, size, flags);
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
