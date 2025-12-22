#pragma once

#include "backend/backend.hpp"
#include "backend/ggml/ggml.hpp"
#include "backend/opencl/opencl_backend.hpp"

#if defined(POWERSERVE_WITH_QNN)
#include "backend/qnn/qnn_backend.hpp"
#endif

#include "core/config.hpp"

#include <map>
#include <memory>
#include <string>

namespace powerserve {

struct Platform {
public:
    // key must be std::string (model_id)
    std::map<std::string, std::unique_ptr<ggml::GGMLBackend>> ggml_backends;
    std::map<std::string, std::unique_ptr<opencl::OpenCLBackend>> opencl_backends;

#if defined(POWERSERVE_WITH_QNN)
    std::unique_ptr<qnn::QNNBackend> qnn_backend;
#endif

public:
    void init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams);
    void destroy_ggml_backend(const std::shared_ptr<ModelConfig> &config);

    void init_opencl_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams);
    void destroy_opencl_backend(const std::shared_ptr<ModelConfig> &config);

    Backend* get_backend(const std::string& model_id);
    const Backend* get_backend(const std::string& model_id) const;
    bool using_opencl(const std::string& model_id) const;

#if defined(POWERSERVE_WITH_QNN)
    void init_qnn_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams);
    void destroy_qnn_backend(const std::shared_ptr<ModelConfig> &config);
#endif

    size_t get_kv_position(std::string &model_id) const;
    void reset_kv_position(std::string &model_id);
};

} // namespace powerserve
