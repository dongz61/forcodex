// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "executor/executor.hpp"

#include "core/logger.hpp"

#include <cstdint>
#include <unordered_set>
#include <cstdint>
#include <array>
#include <string>
#include <fmt/core.h>


namespace powerserve {

// ziqian add: debug hook impl and other debug tools
// ===== Debug hook impl =====
static OpAfterExecHook g_after_exec_hook = nullptr;

void set_op_after_exec_hook(OpAfterExecHook hook) {
    g_after_exec_hook = std::move(hook);
}

OpAfterExecHook & get_op_after_exec_hook() {
    return g_after_exec_hook;
}

// 简单稳定 hash：FNV-1a 64-bit
static inline uint64_t fnv1a_u64(uint64_t h, uint64_t x) {
    constexpr uint64_t FNV_PRIME = 1099511628211ull;
    h ^= x;
    h *= FNV_PRIME;
    return h;
}

static inline uint64_t hash_u64(uint64_t h, uint64_t x) {
    return fnv1a_u64(h, x);
}

// 把 dtype/shape 编码进 hash
static inline uint64_t hash_tensor_meta(uint64_t h, const powerserve::Tensor *t) {
    if (!t) {
        return hash_u64(h, 0xdeadbeefull);
    }
    h = hash_u64(h, (uint64_t)t->m_dtype);

    // 形状固定 4 维
    auto sh = t->m_shape;
    h = hash_u64(h, (uint64_t)sh[0]);
    h = hash_u64(h, (uint64_t)sh[1]);
    h = hash_u64(h, (uint64_t)sh[2]);
    h = hash_u64(h, (uint64_t)sh[3]);

    // view / buffer-kind 也可以编码（可选）
    h = hash_u64(h, (uint64_t)(t->m_data ? 1 : 0));

    return h;
}

// 对整个 ops 列表做 signature hash
static uint64_t hash_ops_signature(const std::vector<std::shared_ptr<powerserve::OpNode>> &ops) {
    uint64_t h = 1469598103934665603ull; // FNV offset basis
    h = hash_u64(h, (uint64_t)ops.size());

    for (size_t i = 0; i < ops.size(); ++i) {
        const auto &op = ops[i];
        if (!op) {
            h = hash_u64(h, 0xabad1deau);
            continue;
        }

        h = hash_u64(h, (uint64_t)op->op);        // op type
        h = hash_u64(h, (uint64_t)op->prev.size());
        h = hash_u64(h, (uint64_t)op->next.size());

        // prev tensors meta
        for (auto &p : op->prev) {
            const powerserve::Tensor *t = p ? p->tensor() : nullptr;
            h = hash_tensor_meta(h, t);
        }

        // next tensors meta
        for (auto &n : op->next) {
            const powerserve::Tensor *t = n ? n->tensor() : nullptr;
            h = hash_tensor_meta(h, t);
        }
    }

    return h;
}
// ziqian: end

// ziqian：增加通过后端决定分配buffer类型
void Executor::allocate_buffers() {
    const bool use_opencl = m_platform.using_opencl(m_graph.m_model_id);

    powerserve::opencl::OpenCLBackend* cl_backend = nullptr;
    if (use_opencl) {
        cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            m_platform.get_backend(m_graph.m_model_id));
        POWERSERVE_ASSERT(cl_backend && "OpenCL backend is null or not OpenCLBackend");
    }

    // ------------------------------------------------------------
    // 0) Build a set of tensors that must stay on CPU (weights/params)
    // ------------------------------------------------------------
    std::unordered_set<Tensor*> skip_migrate;
    if (use_opencl) {
        for (auto &op : m_graph.ops) {
            switch (op->op) {
            case OpType::GET_EMBEDDING: {
                // embedding Phase1 expects weight on CPU
                Tensor* w = op->prev[0]->tensor();   // weight
                if (w) skip_migrate.insert(w);
            } break;
            case OpType::RMS_NORM: {
                // rmsnorm Phase1 expects weight on CPU (your backend does D2H -> CPU -> H2D)
                Tensor* w = op->prev[1]->tensor();   // weight
                if (w) skip_migrate.insert(w);
            } break;
            default:
                break;
            }
        }
    }

    for (auto &node : m_graph.tensors) {
        auto tensor = node->tensor();
        if (!tensor) continue;

        // ------------------------------------------------------------
        // Special handling for views: OpenCL must allocate sub-buffers
        // ------------------------------------------------------------
        if (use_opencl && node->type == NodeType::TENSOR_VIEW) {
            // If already OpenCLBuffer-backed, nothing to do.
            if (tensor->m_data) {
                auto &base = tensor->get<BaseBuffer>();
                if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&base)) {
                    continue;
                }
            }

            // Views should always be allocated as OpenCL sub-buffers from parent OpenCLBuffer.
            // If parent hasn't been migrated yet, it will be migrated in its own iteration.
            // After parent becomes OpenCLBuffer, this allocation will succeed.
            switch (tensor->m_dtype) {
            case DataType::FP32:
                create_opencl_buffer<float>(node);
                break;
            case DataType::FP16:
                create_opencl_buffer<uint16_t>(node);
                break;
            case DataType::INT32:
                create_opencl_buffer<int32_t>(node);
                break;
            default:
                POWERSERVE_ABORT("allocate_buffers(view): unsupported dtype");
            }
            continue;
        }

        // ----------------------------
        // Case 1: tensor already has buffer
        // ----------------------------
        if (tensor->m_data) {
            if (!use_opencl) {
                continue; // CPU backend no-op
            }

            // Do NOT migrate weights/params
            if (skip_migrate.count(tensor) > 0) {
                continue;
            }

            // already OpenCLBuffer -> skip
            {
                auto &base = tensor->get<BaseBuffer>();
                if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&base)) {
                    continue;
                }
            }

            // must be CPUBuffer if we reach here
            {
                auto &base = tensor->get<BaseBuffer>();
                auto *cpu_buf = dynamic_cast<powerserve::CPUBuffer*>(&base);
                if (!cpu_buf) {
                    POWERSERVE_ABORT("allocate_buffers: tensor has non-CPU, non-OpenCL buffer type");
                }

                // dtype whitelist for migration
                if (tensor->m_dtype != DataType::FP32 &&
                    tensor->m_dtype != DataType::FP16 &&
                    tensor->m_dtype != DataType::INT32) {
                    continue;
                }

                // 1) clone metadata into tmp tensor (non-view only)
                Tensor tmp = *tensor;
                tmp.m_data.reset();

                // 2) allocate OpenCL buffer for tmp
                switch (tensor->m_dtype) {
                case DataType::FP32:
                    create_opencl_buffer_for_tensor<float>(&tmp);
                    break;
                case DataType::FP16:
                    create_opencl_buffer_for_tensor<uint16_t>(&tmp);
                    break;
                case DataType::INT32:
                    create_opencl_buffer_for_tensor<int32_t>(&tmp);
                    break;
                default:
                    POWERSERVE_ABORT("allocate_buffers migrate: unsupported dtype={} shape=[{}, {}, {}, {}]",
                                     (int)tensor->m_dtype,
                                     tensor->m_shape[0], tensor->m_shape[1], tensor->m_shape[2], tensor->m_shape[3]);
                }

                // 3) H2D copy
                cl_backend->copy(&tmp, tensor);

                // 4) replace original CPU buffer
                tensor->m_data = std::move(tmp.m_data);

                continue;
            }
        }

        // ----------------------------
        // Case 2: tensor has no buffer (original logic)
        // ----------------------------
        switch (tensor->m_dtype) {
        case DataType::FP32:
            if (use_opencl) create_opencl_buffer<float>(node);
            else            create_cpu_buffer<float>(node);
            break;
        case DataType::FP16:
            if (use_opencl) create_opencl_buffer<uint16_t>(node);
            else            create_cpu_buffer<uint16_t>(node);
            break;
        case DataType::INT32:
            if (use_opencl) create_opencl_buffer<int32_t>(node);
            else            create_cpu_buffer<int32_t>(node);
            break;
        default:
            POWERSERVE_ABORT("allocate_buffers: unsupported dtype");
        }
    }
}

// ziqian：end

void Executor::plan() {
    // ziqian：增加通过后端决定使用什么plan方法
    auto *backend = m_platform.get_backend(m_graph.m_model_id);
    POWERSERVE_ASSERT(backend != nullptr);
    backend->plan(m_graph.ops);
    // ziqian：end
}

#ifdef POWERSERVE_DUMP_TENSORS
// Debug code: dump a tensor's data
void tensor_dump(Tensor* x, std::vector<size_t> max_show_elems, std::string name) {
    POWERSERVE_ASSERT(x->m_dtype == DataType::FP32);
    auto shape = x->m_shape;
    auto stride = x->get<CPUBuffer>().m_stride;
    printf("--------------------Dumping GGML tensor-------------------\n");
    printf("Tensor name: %s\n", name.c_str());
    printf("Tensor rank: 4\n");
    printf("Tensor shape: [%ld, %ld, %ld, %ld]\n", shape[3], shape[2], shape[1], shape[0]);
    printf("Tensor dtype: FP32\n");
    for (size_t i3 = 0; i3 < shape[3] && i3 < max_show_elems[3]; i3++) {
        for (size_t i2 = 0; i2 < shape[2] && i2 < max_show_elems[2]; i2++) {
            for (size_t i1 = 0; i1 < shape[1] && i1 < max_show_elems[1]; i1++) {
                printf("Dumping elements in dimension [%ld, %ld, %ld]:", i3, i2, i1);
                for (size_t i0 = 0; i0 < shape[0] && i0 < max_show_elems[0]; i0++) {
                    float *ptr = (float *)((char *)x->get<CPUBuffer>().m_data + i3 * stride[3] + i2 * stride[2] + i1 * stride[1] + i0 * stride[0]);
                    printf(" %.6f", (double)*ptr);
                }
                printf("\n");
            }
        }
    }
}
#endif //POWERSERVE_DUMP_TENSORS

void Executor::run() {
    // ziqian：增加通过后端决定执行哪个算子
    auto &model_id = m_graph.m_model_id;
    auto *backend = m_platform.get_backend(model_id);
    POWERSERVE_ASSERT(backend != nullptr);
    const bool use_opencl = m_platform.using_opencl(model_id);
    plan();

    // ziqian: compare whether two backends have same op order
    // uint64_t h = hash_ops_signature(m_graph.ops);
    // fmt::print("[OPSEQ HASH] model_id={} ops={} hash=0x{:016x}\n",
    //            m_graph.m_model_id,
    //            m_graph.ops.size(),
    //            h);

    // // 可选：打印前 20 个 op 概览，便于肉眼核对
    // int limit = std::min<int>(20, (int)m_graph.ops.size());
    // for (int i = 0; i < limit; ++i) {
    //     auto &op = m_graph.ops[i];
    //     fmt::print("  [op#{:03d}] type={} prev={} next={}\n",
    //                i, (int)op->op, op->prev.size(), op->next.size());
    // }
    

    int op_idx = 0;

    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::GET_EMBEDDING: {
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            backend->get_embedding(out, weight, tokens);
#ifdef POWERSERVE_DUMP_TENSORS
            std::vector<size_t> dump_embedding_dims={8, 6, 1, 1};
            tensor_dump(out, dump_embedding_dims, "Embedding");
#endif //POWERSERVE_DUMP_TENSORS
        } break;

        case OpType::ADD: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            backend->add(out, a, b);
        } break;

        case OpType::MAT_MUL: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            backend->matmul(out, a, b);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            auto [eps]  = op->get_params<RMSNormParams>();
            backend->rmsnorm(out, x, weight, eps);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            backend->silu_hadamard(out, gate, up);
        } break;

        case OpType::ROPE: {
            auto src             = op->prev[0]->tensor();
            auto out             = op->next[0]->tensor();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            backend->rope(out, src, pos, rope_cfg);
        } break;

        case OpType::SOFTMAX: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            backend->softmax(out, x);
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            backend->copy(dst, src);
        } break;

#if defined(POWERSERVE_WITH_QNN)
        case OpType::QNN_FORWARD: {
            auto x     = op->prev[0]->tensor();
            auto out   = op->output();
            auto pos   = op->get_params<QNNForwardParams>().pos;
            auto &mask = op->get_params<QNNForwardParams>().mask;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pos, mask);
#ifdef POWERSERVE_DUMP_TENSORS
            std::vector<size_t> dump_qnn_dims={8, 6, 1, 1};
            tensor_dump(out, dump_qnn_dims, "QNN");
#endif //POWERSERVE_DUMP_TENSORS
        } break;
        case OpType::QNN_FORWARD_VL: {
            auto x                  = op->prev[0]->tensor();
            auto out                = op->output();
            auto pos                = op->get_params<QNNForwardVLParams>().pos;
            auto &mask              = op->get_params<QNNForwardVLParams>().mask;
            auto &pixel_values_list = op->get_params<QNNForwardVLParams>().pixel_values_list;
            auto &img_infos         = op->get_params<QNNForwardVLParams>().img_infos;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pixel_values_list, img_infos, pos, mask);
            pixel_values_list.clear();
            img_infos.clear();
        } break;
#endif

        case OpType::PRINT: {
            auto x    = op->prev[0]->tensor();
            auto size = op->get_params<PrintParams>().size;
            backend->print(x, size);

        } break;

        case OpType::ADD_CACHE: {
            auto k                 = op->prev[0]->tensor();
            auto v                 = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            backend->add_cache(k, v, L, pos, head_id);
        } break;
        case OpType::PERMUTE: {
            auto x      = op->prev[0]->tensor();
            auto out    = op->output();
            auto [axes] = op->get_params<PermuteParams>();
            backend->permute(out, x, axes);
        } break;

        case OpType::CONT: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            backend->cont(out, x);
        } break;

        case OpType::VIEW: {
            auto out = op->output();
            auto *out_view_node = op->next[0]->tensor_view();
            POWERSERVE_ASSERT(out_view_node && "VIEW op output is not a TensorViewNode");
            Tensor *src = out_view_node->parent;
            POWERSERVE_ASSERT(src && "TensorViewNode parent is null");
            auto [stride, offset] = op->get_params<ViewParams>();

            if (use_opencl) {
                // OpenCL: materialize view as sub-buffer
                auto &parent = src->get<powerserve::opencl::OpenCLBuffer>();

                std::shared_ptr<powerserve::opencl::OpenCLBuffer> view_buf;
                switch (out->m_dtype) {
                    case DataType::FP32:
                        view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(
                            parent, out->m_shape, offset);
                        break;
                    case DataType::FP16:
                        view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<uint16_t>(
                            parent, out->m_shape, offset);
                        break;
                    case DataType::INT32:
                        view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<int32_t>(
                            parent, out->m_shape, offset);
                        break;
                    default:
                        POWERSERVE_ABORT("VIEW OpenCL unsupported dtype");
                }

                POWERSERVE_ASSERT(view_buf && "Failed to create OpenCL view buffer");
                out->m_data = std::static_pointer_cast<BaseBuffer>(view_buf);

                // Important: VIEW carries stride metadata from graph
                out->get<powerserve::opencl::OpenCLBuffer>().m_stride = stride;
                break;
            }

            // CPU behavior unchanged
            out->get<CPUBuffer>().m_stride = stride;
            out->get<CPUBuffer>().m_data   = (char *)out->get<CPUBuffer>().m_data + offset;
        } break;

        case OpType::SOFTMAX_EXT: {
            auto out               = op->output();
            auto x                 = op->prev[0]->tensor();
            auto mask              = op->prev[1]->tensor();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();

            backend->softmax_ext(out, x, mask, scale, max_bias);
        } break;

        case OpType::GET_MASK: {
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            auto n_kv        = out->m_shape[0];
            auto batch_size  = out->m_shape[1];

            POWERSERVE_ASSERT(out->m_dtype == DataType::FP32);

            if (!use_opencl) {
                // ===== CPU original path =====
                auto mask_buf = (float *)out->get<CPUBuffer>().m_data;
                for (size_t i = 0; i < batch_size; i++) {
                    size_t cur_pos = pos[i];
                    for (size_t j = 0; j < n_kv; j++) {
                        mask_buf[j + i * n_kv] = (j <= cur_pos) ? 0.f : -INFINITY;
                    }
                }
                break;
            }

            // ===== OpenCL bring-up fallback path =====
            // 1) create a temporary CPU tensor
            Tensor tmp_cpu(DataType::FP32, out->m_shape);
            tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

            // 2) fill mask on CPU (same logic)
            auto mask_buf = (float *)tmp_cpu.get<CPUBuffer>().m_data;
            for (size_t i = 0; i < batch_size; i++) {
                size_t cur_pos = pos[i];
                for (size_t j = 0; j < n_kv; j++) {
                    mask_buf[j + i * n_kv] = (j <= cur_pos) ? 0.f : -INFINITY;
                }
            }

            // 3) copy CPU -> OpenCL output tensor
            // backend is the current backend (OpenCLBackend when use_opencl)
            auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend *>(backend);
            POWERSERVE_ASSERT(cl_backend && "backend is not OpenCLBackend while use_opencl=true");
            cl_backend->copy(out, &tmp_cpu);

        } break;


        case OpType::TRANSPOSE: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            backend->transpose(out, x);
        } break;
        default:
            POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
        if (get_op_after_exec_hook()) {
            get_op_after_exec_hook()(op_idx, op.get());
        }

        op_idx++;
    }
    // ziqian：end
} 
}// namespace powerserve
