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

    initialized = false;
    std::cout << "[DEBUG] === CLEANUP DONE ===" << std::endl;
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
    options.enable_mad = true;
    options.unsafe_math = true;
    options.finite_math = true;
    options.fast_relaxed_math = true;

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

std::shared_ptr<OpenCLBuffer> OpenCLBackend::create_buffer(Shape shape, DataType dtype) {
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

} // namespace powerserve::opencl
