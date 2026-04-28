#include "executor/executor.hpp"

#include "backend/ggml/ggml.hpp"
#include "backend/ggml/ggml_cluster_manager.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/module/ggml_cluster_profile.hpp"
#include "model/module/ggml_cluster_runtime.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace powerserve {

namespace {

auto copy_tensor_f32_contiguous(const Tensor *tensor) -> std::vector<float> {
    POWERSERVE_ASSERT(tensor != nullptr);
    POWERSERVE_ASSERT(tensor->m_dtype == DataType::FP32);
    const auto &buffer = tensor->get<CPUBuffer>();
    const auto *base = static_cast<const char *>(buffer.m_data);
    std::vector<float> out;
    out.reserve(tensor->n_elements());
    for (size_t i3 = 0; i3 < tensor->m_shape[3]; ++i3) {
        for (size_t i2 = 0; i2 < tensor->m_shape[2]; ++i2) {
            for (size_t i1 = 0; i1 < tensor->m_shape[1]; ++i1) {
                for (size_t i0 = 0; i0 < tensor->m_shape[0]; ++i0) {
                    const auto *ptr = reinterpret_cast<const float *>(
                        base + i3 * buffer.m_stride[3] +
                        i2 * buffer.m_stride[2] +
                        i1 * buffer.m_stride[1] +
                        i0 * buffer.m_stride[0]
                    );
                    out.push_back(*ptr);
                }
            }
        }
    }
    return out;
}

} // namespace

static inline bool force_get_mask_cpu_fallback() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char *v = std::getenv("POWERSERVE_FORCE_GET_MASK_CPU_FALLBACK");
    cached = (v && (
        std::strcmp(v, "1") == 0 ||
        std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "TRUE") == 0 ||
        std::strcmp(v, "on") == 0 ||
        std::strcmp(v, "ON") == 0
    )) ? 1 : 0;
    return cached == 1;
}

static OpAfterExecHook g_after_exec_hook = nullptr;

void set_op_after_exec_hook(OpAfterExecHook hook) {
    g_after_exec_hook = std::move(hook);
}

OpAfterExecHook & get_op_after_exec_hook() {
    return g_after_exec_hook;
}

// ziqian：增加通过后端决定分配buffer类型
void Executor::allocate_buffers() {
    const bool use_opencl = m_platform.using_opencl(m_graph.m_model_id);

    powerserve::opencl::OpenCLBackend* cl_backend = nullptr;
    if (use_opencl) {
        cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend*>(
            m_platform.get_backend(m_graph.m_model_id));
        POWERSERVE_ASSERT(cl_backend && "OpenCL backend is null or not OpenCLBackend");
    }

    std::unordered_set<Tensor*> view_op_outputs;
    if (use_opencl) {
        for (auto &op : m_graph.ops) {
            if (op && op->op == OpType::VIEW) {
                Tensor* out = op->output();
                if (out) view_op_outputs.insert(out);
            }
        }
    }

    for (auto &node : m_graph.tensors) {
        auto tensor = node->tensor();
        if (!tensor) continue;

        if (use_opencl && node->type == NodeType::TENSOR_VIEW) {
            // 1) VIEW op 的输出：run() 里会按 (stride, offset) 物化 OpenCL view（避免重复）
            if (view_op_outputs.count(tensor) > 0) {
                continue;
            }

            // 2) 其他 view（比如 transpose/permute 产生的 TensorViewNode）：
            if (tensor->m_data) {
                auto &base = tensor->get<BaseBuffer>();
                if (dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&base)) {
                    continue;
                }
                // 如果这里不是 OpenCLBuffer，说明 view 在 OpenCL 模式下混入了 CPUBuffer（通常是你还没处理的迁移路径）
                POWERSERVE_ABORT("allocate_buffers(view): view tensor has non-OpenCL buffer under use_opencl");
            }

            auto *view_node = node->tensor_view();
            POWERSERVE_ASSERT(view_node && "allocate_buffers(view): tensor_view() is null");
            POWERSERVE_ASSERT(view_node->parent && "allocate_buffers(view): parent is null");
            POWERSERVE_ASSERT(view_node->parent->m_data && "allocate_buffers(view): parent has no buffer");

            auto &parent_cl = view_node->parent->get<powerserve::opencl::OpenCLBuffer>();

            std::shared_ptr<powerserve::opencl::OpenCLBuffer> view_buf;
            switch (tensor->m_dtype) {
            case DataType::FP32:
                view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(parent_cl, tensor->m_shape, /*offset=*/0);
                break;
            case DataType::FP16:
                view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<uint16_t>(parent_cl, tensor->m_shape, /*offset=*/0);
                break;
            case DataType::INT32:
                view_buf = powerserve::opencl::OpenCLBuffer::create_buffer_view<int32_t>(parent_cl, tensor->m_shape, /*offset=*/0);
                break;
            default:
                POWERSERVE_ABORT("allocate_buffers(view): unsupported dtype");
            }

            POWERSERVE_ASSERT(view_buf && "allocate_buffers(view): failed to create OpenCL view buffer");
            tensor->m_data = std::static_pointer_cast<BaseBuffer>(view_buf);
            continue;
        }

        // ----------------------------
        // Case 1: tensor already has buffer
        // ----------------------------
        if (tensor->m_data) {
            if (!use_opencl) {
                continue; // CPU backend no-op
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
                    tensor->m_dtype != DataType::INT32 &&
                    tensor->m_dtype != DataType::GGML_Q4_0 &&
                    tensor->m_dtype != DataType::GGML_Q8_0) {
                    continue;
                }

                auto resident = cl_backend->get_or_create_resident_buffer(tensor);
                if (!resident) {
                    POWERSERVE_ABORT("allocate_buffers migrate: failed to get resident OpenCL buffer for dtype={} shape=[{}, {}, {}, {}]",
                                     (int)tensor->m_dtype,
                                     tensor->m_shape[0], tensor->m_shape[1], tensor->m_shape[2], tensor->m_shape[3]);
                }
                tensor->m_data = std::static_pointer_cast<BaseBuffer>(resident);

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
        case DataType::GGML_Q4_0:
            if (use_opencl) create_opencl_buffer<uint8_t>(node);
            else            POWERSERVE_ABORT("allocate_buffers: quant dtype on CPU path requires preloaded ggml buffer");
            break;
        case DataType::GGML_Q8_0:
            if (use_opencl) create_opencl_buffer<uint8_t>(node);
            else            POWERSERVE_ABORT("allocate_buffers: quant dtype on CPU path requires preloaded ggml buffer");
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

    int op_idx = 0;

    const bool cluster_profile = ggml::cluster_profile_enabled();
    const bool dense_profile = ggml::dense_profile_enabled();
    const bool layer_profile = cluster_profile || dense_profile;
    for (auto op : m_graph.ops) {
        const int64_t op_start_ns = layer_profile ? timestamp_ns() : 0;
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

        case OpType::CLUSTER_UPDATE: {
            POWERSERVE_ASSERT(!use_opencl, "CLUSTER_UPDATE is currently only implemented on GGML/CPU");
            const auto &params = op->get_params<ClusterUpdateParams>();
            const auto runtime = powerserve::ggml::get_cluster_runtime(params.model_id);
            POWERSERVE_ASSERT(runtime.manager != nullptr, "CLUSTER_UPDATE missing cluster manager");
            auto *k = op->prev[0]->tensor();
            auto *v = op->prev[1]->tensor();
            const int64_t copy_start_ns = cluster_profile ? timestamp_ns() : 0;
            auto new_k = copy_tensor_f32_contiguous(k);
            auto new_v = copy_tensor_f32_contiguous(v);
            const int64_t copy_ns = cluster_profile ? timestamp_ns() - copy_start_ns : 0;
            const int64_t update_start_ns = cluster_profile ? timestamp_ns() : 0;
            runtime.manager->update_layer_after_decode(
                static_cast<size_t>(params.layer_id),
                params.token_position,
                new_k,
                new_v
            );
            if (cluster_profile) {
                ggml::cluster_profile_record_cluster_update(
                    static_cast<size_t>(params.layer_id),
                    copy_ns,
                    timestamp_ns() - update_start_ns
                );
            }
        } break;

        case OpType::CLUSTER_ATTN: {
            auto out = op->output();
            auto q = op->prev[0]->tensor();
            const auto &params = op->get_params<ClusterAttnParams>();

            auto *ggml_backend = dynamic_cast<powerserve::ggml::GGMLBackend *>(backend);
            POWERSERVE_ASSERT(
                ggml_backend != nullptr,
                "CLUSTER_ATTN is currently only implemented in GGML backend"
            );
            ggml_backend->cluster_attn(
                out,
                q,
                params.model_id,
                params.layer_id,
                params.scale,
                params.topk_clusters,
                params.n_heads,
                params.n_kv_heads,
                params.head_size
            );
        } break;

        case OpType::GET_MASK: {
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            auto n_kv        = out->m_shape[0];
            auto batch_size  = out->m_shape[1];
            (void)mask;

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

            auto *cl_backend = dynamic_cast<powerserve::opencl::OpenCLBackend *>(backend);
            POWERSERVE_ASSERT(cl_backend && "backend is not OpenCLBackend while use_opencl=true");

            if (force_get_mask_cpu_fallback()) {
                // ===== OpenCL CPU fallback path =====
                Tensor tmp_cpu(DataType::FP32, out->m_shape);
                tmp_cpu.m_data = powerserve::CPUBuffer::create_buffer<float>(out->m_shape);

                auto mask_buf = (float *)tmp_cpu.get<CPUBuffer>().m_data;
                for (size_t i = 0; i < batch_size; i++) {
                    size_t cur_pos = pos[i];
                    for (size_t j = 0; j < n_kv; j++) {
                        mask_buf[j + i * n_kv] = (j <= cur_pos) ? 0.f : -INFINITY;
                    }
                }

                cl_backend->copy(out, &tmp_cpu);
                break;
            }

            // ===== OpenCL GPU path =====
            cl_backend->get_mask(out, pos);

        } break;


        case OpType::TRANSPOSE: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            backend->transpose(out, x);
        } break;
        default:
            POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
        if (cluster_profile && op->profile_layer_id >= 0) {
            const int64_t op_ns = timestamp_ns() - op_start_ns;
            ggml::cluster_profile_record_layer_op(
                static_cast<size_t>(op->profile_layer_id),
                "layer",
                op_ns
            );
            if (op->profile_scope == "ffn") {
                ggml::cluster_profile_record_layer_op(
                    static_cast<size_t>(op->profile_layer_id),
                    "ffn",
                    op_ns
                );
            }
        }
        if (dense_profile && op->profile_layer_id >= 0) {
            const int64_t op_ns = timestamp_ns() - op_start_ns;
            ggml::dense_profile_record_layer_op(
                static_cast<size_t>(op->profile_layer_id),
                "layer",
                op_ns
            );
            if (op->profile_scope == "attn") {
                ggml::dense_profile_record_layer_op(
                    static_cast<size_t>(op->profile_layer_id),
                    "attn",
                    op_ns
                );
            } else if (op->profile_scope == "ffn") {
                ggml::dense_profile_record_layer_op(
                    static_cast<size_t>(op->profile_layer_id),
                    "ffn",
                    op_ns
                );
            }
        }
        if (get_op_after_exec_hook()) {
            get_op_after_exec_hook()(op_idx, op.get());
        }

        op_idx++;
    }
    // ziqian：end
} 
}// namespace powerserve
