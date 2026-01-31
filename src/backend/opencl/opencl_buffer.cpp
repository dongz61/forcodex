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

} // namespace powerserve::opencl
