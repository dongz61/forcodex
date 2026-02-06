#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"
#include "ggml.h"

#include <iostream>
#include <mutex>

namespace powerserve::opencl {

static void ensure_ggml_global_init_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_init_params p{};
        p.mem_size   = 1024 * 1024;
        p.mem_buffer = NULL;
        ggml_context* ctx = ggml_init(p);
        ggml_free(ctx);
    });
}

OpenCLBackend::OpenCLBackend(const ModelConfig::LLMConfig &llm,
                             const HyperParams &hparams)
    : m_llm(llm), m_hparams(hparams) {
}

OpenCLBackend::~OpenCLBackend() {
    cleanup();
    POWERSERVE_LOG_DEBUG("OpenCLBackend destructor called");
}

void OpenCLBackend::cleanup() {
    if (!initialized) return;

    std::cout << "[DEBUG] === CLEANUP TEST VERSION ===" << std::endl;

    std::cout << "[DEBUG] 1. Cleaning OpenCL context first..." << std::endl;
    clear_quant_cache();
    if (context) {
        context.reset();
        std::cout << "[DEBUG] Context released" << std::endl;
    }

    std::cout << "[DEBUG] 2. Cleaning memory pool..." << std::endl;
    if (memory_pool) {
        memory_pool.reset();
        std::cout << "[DEBUG] Memory pool released" << std::endl;
    }

    std::cout << "[DEBUG] 3. Now cleaning thread pool..." << std::endl;
    if (thread_pool) {
        thread_pool.reset();
        std::cout << "[DEBUG] Thread pool released" << std::endl;
    }

    m_ggml_fallback.reset();
    m_ggml_fallback_wsize = 0;
    m_tokens_buffer.reset();
    m_tokens_capacity = 0;

    initialized = false;
    std::cout << "[DEBUG] === CLEANUP DONE ===" << std::endl;
}

void OpenCLBackend::ensure_tokens_buffer(size_t token_count) const {
    if (token_count == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_tokens_mutex);
    if (m_tokens_buffer && m_tokens_capacity >= token_count) {
        return;
    }
    Shape shape{token_count, 1, 1, 1};
    auto buffer = create_buffer(shape, DataType::INT32);
    if (!buffer) {
        POWERSERVE_LOG_ERROR("OpenCLBackend::ensure_tokens_buffer failed to allocate tokens buffer");
        return;
    }
    m_tokens_buffer = std::move(buffer);
    m_tokens_capacity = token_count;
}

bool OpenCLBackend::initialize() {
    if (initialized) {
        return true;
    }

    ensure_ggml_global_init_once();

    context = std::make_shared<OpenCLContext>();
    if (!context->initialize(device_preference)) {
        POWERSERVE_LOG_ERROR("Failed to initialize OpenCL context");
        return false;
    }

    memory_pool = std::make_shared<OpenCLMemoryPool>(context);

    kernel_manager = std::make_shared<OpenCLKernelManager>(context);

    OpenCLCompileOptions options;
    options.opencl_c_std = "CL3.0";
    options.enable_mad = false;
    options.unsafe_math = false;
    options.finite_math = false;
    options.fast_relaxed_math = false;

    if (!kernel_manager->initialize(options)) {
        POWERSERVE_LOG_ERROR("Failed to initialize OpenCL kernel manager");
        return false;
    }

    ensure_kv_cache_allocated_v0(1);

    if (!m_ggml_fallback) {
        m_ggml_fallback = std::make_unique<powerserve::ggml::GGMLBackend>(m_llm, m_hparams);
        m_ggml_fallback->setup_threadpool();
    }

    constexpr size_t kMinFallbackWSize = 256ULL * 1024 * 1024;
    if (m_ggml_fallback_wsize < kMinFallbackWSize) {
        m_ggml_fallback->setup_work_data(kMinFallbackWSize);
        m_ggml_fallback_wsize = kMinFallbackWSize;
    }

    initialized = true;

    return true;
}

std::shared_ptr<OpenCLBuffer> OpenCLBackend::create_buffer(Shape shape, DataType dtype) const {
    if (!memory_pool) {
        POWERSERVE_LOG_ERROR("Memory pool not initialized");
        return nullptr;
    }

    switch (dtype) {
        case DataType::FP32:
            return OpenCLBuffer::create_buffer<float>(shape, memory_pool);
        case DataType::FP16:
            return OpenCLBuffer::create_buffer<cl_half>(shape, memory_pool);
        case DataType::INT32:
            return OpenCLBuffer::create_buffer<cl_int>(shape, memory_pool);

        // ===== Quantized GGML buffers =====
        case DataType::GGML_Q4_0:
        case DataType::GGML_Q8_0: {
            // Match ggml tensor layout:
            // nb0 = ggml_type_size(type)
            // nb1 = ggml_row_size(type, ne0)
            // nb2 = nb1 * ne1
            // nb3 = nb2 * ne2
            const ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(dtype);

            Stride stride{};
            stride[0] = (size_t) ggml_type_size(gt);
            stride[1] = (size_t) ggml_row_size(gt, (int64_t) shape[0]);
            stride[2] = stride[1] * (size_t) shape[1];
            stride[3] = stride[2] * (size_t) shape[2];

            const size_t bytes = stride[3] * (size_t) shape[3];

            cl_mem device_buffer = memory_pool->allocate_pooled(bytes, CL_MEM_READ_WRITE);
            if (!device_buffer) {
                POWERSERVE_LOG_ERROR("Failed to allocate OpenCL quant buffer: dtype={} bytes={}",
                                     (int) dtype, bytes);
                return nullptr;
            }

            // Owns buffer, pooled, base_offset = 0
            return std::make_shared<OpenCLBuffer>(stride, device_buffer, bytes, memory_pool,
                                                  /*owns_buffer=*/true,
                                                  /*is_pooled=*/true,
                                                  /*base_offset=*/0);
        }

        default:
            POWERSERVE_LOG_ERROR("Unsupported data type for OpenCL buffer: {}",
                               static_cast<int>(dtype));
            return nullptr;
    }
}


void OpenCLBackend::plan(std::vector<std::shared_ptr<OpNode>> & /*ops*/) {
}

bool OpenCLBackend::is_contiguous(const Tensor *t, int n) const {
    if (!t) return false;
    if (n <= 0) return true;

    size_t expected[GGML_MAX_DIMS] = {0};

    size_t nb0 = 0;
    bool is_quant = (t->m_dtype == DataType::GGML_Q4_0 || t->m_dtype == DataType::GGML_Q8_0);

    if (is_quant) {
        const enum ggml_type gt = powerserve::ggml::convert_datatype_to_ggml(t->m_dtype);
        nb0 = (size_t)ggml_type_size(gt);
        expected[0] = nb0;
        if (n > 1) {
            expected[1] = (size_t)ggml_row_size(gt, (int64_t)t->m_shape[0]);
        }
    } else {
        nb0 = powerserve::get_type_size(t->m_dtype);
        expected[0] = nb0;
        if (n > 1) {
            expected[1] = expected[0] * (size_t)t->m_shape[0];
        }
    }

    for (int i = 2; i < n; ++i) {
        expected[i] = expected[i - 1] * (size_t)t->m_shape[i - 1];
    }

    const size_t *actual = nullptr;
    BaseBuffer& base = const_cast<Tensor*>(t)->get<BaseBuffer>();
    if (auto *buf = dynamic_cast<OpenCLBuffer*>(&base)) {
        actual = buf->m_stride.data();
    } else if (auto *buf = dynamic_cast<powerserve::CPUBuffer*>(&base)) {
        actual = buf->m_stride.data();
    } else {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        if ((size_t)actual[i] != expected[i]) return false;
    }
    return true;
}

void OpenCLBackend::clear_quant_cache() const {
    std::lock_guard<std::mutex> lock(m_quant_split_mutex);
    m_quant_split_cache.clear();
}

} // namespace powerserve::opencl
