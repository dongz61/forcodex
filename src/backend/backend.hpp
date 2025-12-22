// backend/backend.hpp
#pragma once

#include "core/config.hpp"   // ModelConfig::LLMConfig::RopeConfig
#include "core/tensor.hpp"   // Tensor, Shape
#include "graph/node.hpp"    // OpNode

#include <cstddef>
#include <memory>
#include <vector>

namespace powerserve {

// Backend receives Tensor (not TensorNode), consistent with GGMLBackend.
struct Backend {
    virtual ~Backend() = default;

    // graph scheduling / preparation
    virtual void plan(std::vector<std::shared_ptr<OpNode>> &ops) = 0;

    // ops
    virtual void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const = 0;
    virtual void get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const = 0;
    virtual void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const = 0;
    virtual void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight, float eps) const = 0;

    virtual void rope(
        Tensor *out,
        const Tensor *src,
        const std::vector<int> &pos,
        const ModelConfig::LLMConfig::RopeConfig &rope_cfg
    ) const = 0;

    virtual void softmax(const Tensor *out, const Tensor *x) const = 0;
    virtual void permute(const Tensor *out, const Tensor *x, Shape axes) const = 0;
    virtual void print(const Tensor* x, size_t size) const = 0;
    virtual void cont(const Tensor *out, const Tensor *x) const = 0;

    virtual void softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias)
        const = 0;

    // misc
    virtual void silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const = 0;
    virtual void copy(const Tensor *dst, const Tensor *src) const = 0;
    virtual void reset_kv_batch_size(const size_t batch_size) const = 0;

    virtual void add_cache(
        const Tensor *k,
        const Tensor *v,
        size_t L,
        const std::vector<int> &pos,
        size_t head_id
    ) = 0;

    virtual void transpose(const Tensor *out, const Tensor *x) const = 0;
};

} // namespace powerserve
