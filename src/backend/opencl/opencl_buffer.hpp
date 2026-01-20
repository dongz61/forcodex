// opencl_buffer.hpp
#pragma once

#include "core/buffer.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"
#include "backend/opencl/opencl_memory.hpp"

#include <CL/cl.h>
#include <cstddef>
#include <cstring>
#include <memory>

namespace powerserve::opencl {

struct OpenCLBuffer final : ::powerserve::BaseBuffer {
public:
    using Stride = ::powerserve::Stride;
    using Shape  = ::powerserve::Shape;

    OpenCLBuffer(Stride stride,
                cl_mem device_buffer,
                size_t size,
                std::shared_ptr<OpenCLMemoryPool> pool,
                bool owns_buffer = true,
                bool is_pooled = true,
                bool is_subbuffer = false,
                size_t base_offset = 0);


    ~OpenCLBuffer() override;

    OpenCLBuffer(const OpenCLBuffer&)            = delete;
    OpenCLBuffer& operator=(const OpenCLBuffer&) = delete;

    OpenCLBuffer(OpenCLBuffer&& other) noexcept;
    OpenCLBuffer& operator=(OpenCLBuffer&& other) noexcept;

    // ---- Introspection ----
    cl_mem get_device_buffer() const { return m_device_buffer; }
    size_t get_size() const { return m_size; }
    const Stride& get_stride() const { return m_stride; }
    size_t get_base_offset() const { return m_base_offset; }
    void   set_base_offset(size_t off) { m_base_offset = off; }


    bool is_valid() const { return m_device_buffer != nullptr; }
    bool owns_buffer() const { return m_owns_buffer; }
    bool is_pooled() const { return m_is_pooled; }
    bool is_subbuffer() const { return m_is_subbuffer; }

    // ---- Data movement ----
    bool copy_to_device(const void* host_data, size_t size);
    bool copy_to_host(void* host_data, size_t size);

    bool copy_to_device_async(const void* host_data, size_t size, cl_event* event = nullptr);
    bool copy_to_host_async(void* host_data, size_t size, cl_event* event = nullptr);

    // ---- Mapping ----
    void* map(cl_map_flags flags = CL_MAP_READ | CL_MAP_WRITE, size_t offset = 0, size_t size = 0);
    bool  unmap(void* mapped_ptr);

    // ---- Sync/helpers ----
    void finish();
    bool clear();

public:
    // Keep member layout consistent with prior header (helps minimize churn)
    Stride m_stride;        // In bytes (keep consistent with CPUBuffer)
    cl_mem m_device_buffer; // OpenCL buffer handle
    size_t m_size;          // Buffer size in bytes
    size_t m_base_offset;   // Base offset in bytes into m_device_buffer (scheme-B view)


    std::shared_ptr<OpenCLMemoryPool> memory_pool;

    bool m_owns_buffer;  // whether this object should free/release m_device_buffer
    bool m_is_pooled;    // whether underlying allocation is from pool
    bool m_is_subbuffer; // whether m_device_buffer is a clCreateSubBuffer result

public:
    template <typename T>
    static auto create_buffer(Shape shape, std::shared_ptr<OpenCLMemoryPool> pool)
        -> std::shared_ptr<OpenCLBuffer> {
        if (!pool) {
            POWERSERVE_LOG_ERROR("Invalid memory pool for buffer creation");
            return nullptr;
        }
        if (shape.empty()) {
            POWERSERVE_LOG_ERROR("Empty shape for buffer creation");
            return nullptr;
        }

        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        cl_mem device_buffer = pool->allocate_pooled(size, CL_MEM_READ_WRITE);
        if (!device_buffer) {
            POWERSERVE_LOG_ERROR("Failed to allocate OpenCL buffer of size {}", size);
            return nullptr;
        }

        return std::make_shared<OpenCLBuffer>(stride, device_buffer, size, pool, true, true, false);
    }

    template <typename T>
    static auto create_buffer_non_pooled(Shape shape, std::shared_ptr<OpenCLMemoryPool> pool)
        -> std::shared_ptr<OpenCLBuffer> {
        if (!pool) {
            POWERSERVE_LOG_ERROR("Invalid memory pool for buffer creation");
            return nullptr;
        }
        if (shape.empty()) {
            POWERSERVE_LOG_ERROR("Empty shape for buffer creation");
            return nullptr;
        }

        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        cl_mem device_buffer = pool->allocate(size, CL_MEM_READ_WRITE);
        if (!device_buffer) {
            POWERSERVE_LOG_ERROR("Failed to allocate non-pooled OpenCL buffer of size {}", size);
            return nullptr;
        }

        return std::make_shared<OpenCLBuffer>(stride, device_buffer, size, pool, true, false, false);
    }

    // Scheme-B view: do NOT create sub-buffer.
    // Just alias the same cl_mem and record base_offset.
    template <typename T>
    static auto create_buffer_view(OpenCLBuffer& parent, Shape shape, size_t offset = 0)
        -> std::shared_ptr<OpenCLBuffer> {
        if (!parent.is_valid()) {
            POWERSERVE_LOG_ERROR("Invalid parent buffer for view creation");
            return nullptr;
        }
        if (shape.empty()) {
            POWERSERVE_LOG_ERROR("Empty shape for view creation");
            return nullptr;
        }
        if (!parent.m_device_buffer) {
            POWERSERVE_LOG_ERROR("Parent has null device buffer");
            return nullptr;
        }
        if (offset > parent.m_size) {
            POWERSERVE_LOG_ERROR("View offset {} exceeds parent buffer size {}", offset, parent.m_size);
            return nullptr;
        }

        // NOTE: stride here is only a default contiguous stride.
        // Executor will overwrite out->OpenCLBuffer::m_stride with the real VIEW stride.
        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }

        // Visible size: remaining bytes from base offset to end.
        const size_t visible_size = parent.m_size - offset;

        // Alias: same cl_mem, NOT owning it, NOT pooled, NOT subbuffer.
        auto buf = std::make_shared<OpenCLBuffer>(
            stride,
            parent.m_device_buffer,
            visible_size,
            parent.memory_pool, // keep for copy/map helpers
            /*owns_buffer=*/false,
            /*is_pooled=*/false,
            /*is_subbuffer=*/false,
            /*base_offset=*/parent.m_base_offset + offset
        );
        return buf;
    }

    template <typename T>
    static auto wrap_existing(cl_mem device_buffer, Shape shape, std::shared_ptr<OpenCLMemoryPool> pool)
        -> std::shared_ptr<OpenCLBuffer> {
        if (!device_buffer) {
            POWERSERVE_LOG_ERROR("Invalid device buffer for wrapping");
            return nullptr;
        }
        if (shape.empty()) {
            POWERSERVE_LOG_ERROR("Empty shape for wrapping");
            return nullptr;
        }

        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        // Not owning external cl_mem by default.
        return std::make_shared<OpenCLBuffer>(stride, device_buffer, size, pool, false, false, false);
    }
};

} // namespace powerserve::opencl
