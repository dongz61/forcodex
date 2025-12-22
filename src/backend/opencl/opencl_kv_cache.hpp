#pragma once

#include "backend/opencl/opencl_buffer.hpp"

#include <cstddef>
#include <memory>
#include <vector>


namespace powerserve::opencl {

// KVCache v0 (minimal):
// - batch = 1
// - decode-only: append 1 token each step
// - FP32 only
// - pre-allocated device buffers
struct OpenCLKV {
    size_t kv_dim = 0;
    size_t max_seq_len = 0;
    size_t batch_size = 1;
    size_t position = 0;

    // Per-layer device buffers:
    // shape conceptually: {kv_dim, max_seq_len, 1, 1} FP32 contiguous
    std::vector<std::shared_ptr<powerserve::opencl::OpenCLBuffer>> key;
    std::vector<std::shared_ptr<powerserve::opencl::OpenCLBuffer>> value;

    void reset() { position = 0; }

    bool allocated() const {
        return kv_dim > 0 && max_seq_len > 0 &&
               !key.empty() && key.size() == value.size();
    }

    bool spec_matches(size_t n_layers, size_t kv_dim_, size_t max_seq_len_) const {
        return key.size() == n_layers &&
            value.size() == n_layers &&
            kv_dim == kv_dim_ &&
            max_seq_len == max_seq_len_ &&
            batch_size == 1;
    }

};

} // namespace powerserve::opencl
