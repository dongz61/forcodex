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
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <mutex>


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

static constexpr uintptr_t kTraceDev = (uintptr_t)0x5790aeb3f7a0;

// 打印 CPU tensor 前 N 个 FP32（按 stride0 取）
static inline void dump_cpu_f32_head(const Tensor *t, int n, const char *tag) {
    if (!t || !t->m_data || t->m_dtype != DataType::FP32) return;
    auto *cb = dynamic_cast<powerserve::CPUBuffer*>(t->m_data.get());
    if (!cb || !cb->m_data) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)t->m_shape[0]);
    size_t s0 = (size_t)cb->m_stride[0];

    float mn = 0.f, mx = 0.f;
    bool first = true;

    fmt::print("  [{}] ", tag);
    for (size_t i = 0; i < count; ++i) {
        float v = *(float*)((char*)cb->m_data + i * s0);
        if (first) { mn = mx = v; first = false; }
        else { mn = std::min(mn, v); mx = std::max(mx, v); }
        if (i < 8) fmt::print("{:.6f} ", v);
    }
    fmt::print("(min={:.6f} max={:.6f} n={})\n", mn, mx, count);
}

// 对 OpenCL tensor 回读前 N 个 FP32（通过一个“缩小 shape 的 view”回读）
static inline void dump_opencl_f32_head(
    powerserve::opencl::OpenCLBackend *cl_backend,
    const Tensor *t_cl,
    int n,
    const char *tag)
{
    if (!cl_backend || !t_cl || !t_cl->m_data || t_cl->m_dtype != DataType::FP32) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)t_cl->m_shape[0]);

    Tensor cl_view = *const_cast<Tensor*>(t_cl);
    cl_view.m_shape = Shape{count, 1, 1, 1};

    Tensor cpu_out(DataType::FP32, Shape{count, 1, 1, 1});
    cpu_out.m_data = powerserve::CPUBuffer::create_buffer<float>(cpu_out.m_shape);

    cl_backend->copy(&cpu_out, &cl_view);
    dump_cpu_f32_head(&cpu_out, (int)count, tag);
}

// 对比 CPU 源 vs CL 回读 head（max_abs + nz）
static inline void compare_cpu_vs_opencl_head(
    powerserve::opencl::OpenCLBackend *cl_backend,
    const Tensor *cpu_src,
    const Tensor *cl_dst,
    int n)
{
    if (!cl_backend || !cpu_src || !cl_dst) return;
    if (!cpu_src->m_data || !cl_dst->m_data) return;
    if (cpu_src->m_dtype != DataType::FP32 || cl_dst->m_dtype != DataType::FP32) return;

    auto *src = dynamic_cast<powerserve::CPUBuffer*>(cpu_src->m_data.get());
    if (!src || !src->m_data) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)cpu_src->m_shape[0]);
    size_t s0 = (size_t)src->m_stride[0];

    Tensor cl_view = *const_cast<Tensor*>(cl_dst);
    cl_view.m_shape = Shape{count, 1, 1, 1};

    Tensor cpu_out(DataType::FP32, Shape{count, 1, 1, 1});
    cpu_out.m_data = powerserve::CPUBuffer::create_buffer<float>(cpu_out.m_shape);

    cl_backend->copy(&cpu_out, &cl_view);
    auto *dst = dynamic_cast<powerserve::CPUBuffer*>(cpu_out.m_data.get());
    if (!dst || !dst->m_data) return;

    float max_abs = 0.f;
    size_t max_i = 0;
    size_t nz = 0;

    for (size_t i = 0; i < count; ++i) {
        float a = *(float*)((char*)src->m_data + i * s0);
        float b = *(float*)((char*)dst->m_data + i * (size_t)dst->m_stride[0]);
        float d = std::fabs(a - b);
        if (d != 0.f) nz++;
        if (d > max_abs) { max_abs = d; max_i = i; }
    }

    fmt::print("  [H2D-head-compare] n={} nz={} max_abs={:.6f} at i={}\n",
               count, nz, max_abs, max_i);
}

// ziqian: end

// ziqian: add logger
static FILE* g_cl_upload_fp = nullptr;
static std::once_flag g_cl_upload_once;

static inline FILE* cl_upload_fp() {
    std::call_once(g_cl_upload_once, []() {
        // ⚠️ fopen 不支持 "~" 展开，必须用绝对路径
        const char* path = "/home/intern/ziqian/PowerServe-opencl/HybridRAG/powerserve_cl_upload.log";

        g_cl_upload_fp = std::fopen(path, "w");
        if (!g_cl_upload_fp) {
            // 打印失败原因，然后直接 abort
            std::fprintf(stderr,
                         "[FATAL] failed to open log file: %s, errno=%d (%s)\n",
                         path, errno, std::strerror(errno));
            std::fflush(stderr);
            std::abort();
        }

        std::setvbuf(g_cl_upload_fp, nullptr, _IOLBF, 0); // 行缓冲
    });
    return g_cl_upload_fp;
}

template <typename... Args>
static inline void CL_UPLOAD_FLOG(const char* fmt_str, Args&&... args) {
    if (FILE* fp = cl_upload_fp()) {
        fmt::print(fp, "[INFO ] ");
        fmt::print(fp, fmt::runtime(fmt_str), std::forward<Args>(args)...);
        fmt::print(fp, "\n");
        std::fflush(fp);
    }
}

// 让 dump_* 也写进同一个文件：加一个 FILE* 参数版本
static inline void dump_cpu_f32_head_fp(FILE* fp, const Tensor *t, int n, const char *tag) {
    if (!fp || !t || !t->m_data || t->m_dtype != DataType::FP32) return;
    auto *cb = dynamic_cast<powerserve::CPUBuffer*>(t->m_data.get());
    if (!cb || !cb->m_data) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)t->m_shape[0]);
    size_t s0 = (size_t)cb->m_stride[0];

    float mn = 0.f, mx = 0.f;
    bool first = true;

    fmt::print(fp, "  [{}] ", tag);
    for (size_t i = 0; i < count; ++i) {
        float v = *(float*)((char*)cb->m_data + i * s0);
        if (first) { mn = mx = v; first = false; }
        else { mn = std::min(mn, v); mx = std::max(mx, v); }
        if (i < 8) fmt::print(fp, "{:.6f} ", v);
    }
    fmt::print(fp, "(min={:.6f} max={:.6f} n={})\n", mn, mx, count);
}

static inline void dump_opencl_f32_head_fp(
    FILE* fp,
    powerserve::opencl::OpenCLBackend *cl_backend,
    const Tensor *t_cl,
    int n,
    const char *tag)
{
    if (!fp || !cl_backend || !t_cl || !t_cl->m_data || t_cl->m_dtype != DataType::FP32) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)t_cl->m_shape[0]);

    Tensor cl_view = *const_cast<Tensor*>(t_cl);
    cl_view.m_shape = Shape{count, 1, 1, 1};

    Tensor cpu_out(DataType::FP32, Shape{count, 1, 1, 1});
    cpu_out.m_data = powerserve::CPUBuffer::create_buffer<float>(cpu_out.m_shape);

    cl_backend->copy(&cpu_out, &cl_view);
    dump_cpu_f32_head_fp(fp, &cpu_out, (int)count, tag);
}

static inline void compare_cpu_vs_opencl_head_fp(
    FILE* fp,
    powerserve::opencl::OpenCLBackend *cl_backend,
    const Tensor *cpu_src,
    const Tensor *cl_dst,
    int n)
{
    if (!fp || !cl_backend || !cpu_src || !cl_dst) return;
    if (!cpu_src->m_data || !cl_dst->m_data) return;
    if (cpu_src->m_dtype != DataType::FP32 || cl_dst->m_dtype != DataType::FP32) return;

    auto *src = dynamic_cast<powerserve::CPUBuffer*>(cpu_src->m_data.get());
    if (!src || !src->m_data) return;

    size_t count = std::min<size_t>((size_t)n, (size_t)cpu_src->m_shape[0]);
    size_t s0 = (size_t)src->m_stride[0];

    Tensor cl_view = *const_cast<Tensor*>(cl_dst);
    cl_view.m_shape = Shape{count, 1, 1, 1};

    Tensor cpu_out(DataType::FP32, Shape{count, 1, 1, 1});
    cpu_out.m_data = powerserve::CPUBuffer::create_buffer<float>(cpu_out.m_shape);

    cl_backend->copy(&cpu_out, &cl_view);
    auto *dst = dynamic_cast<powerserve::CPUBuffer*>(cpu_out.m_data.get());
    if (!dst || !dst->m_data) return;

    float max_abs = 0.f;
    size_t max_i = 0;
    size_t nz = 0;

    for (size_t i = 0; i < count; ++i) {
        float a = *(float*)((char*)src->m_data + i * s0);
        float b = *(float*)((char*)dst->m_data + i * (size_t)dst->m_stride[0]);
        float d = std::fabs(a - b);
        if (d != 0.f) nz++;
        if (d > max_abs) { max_abs = d; max_i = i; }
    }

    fmt::print(fp, "  [H2D-head-compare] n={} nz={} max_abs={:.6f} at i={}\n",
               count, nz, max_abs, max_i);
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

                {
                    auto *dst_cl = dynamic_cast<powerserve::opencl::OpenCLBuffer*>(&tmp.get<BaseBuffer>());
                    void *dst_dev = dst_cl ? (void*)dst_cl->get_device_buffer() : nullptr;
                    const size_t dst_size = dst_cl ? (size_t)dst_cl->m_size : 0;
                    const bool hit = (dst_size == 67108864);

                    if (hit) {
                        auto *src_cpu = dynamic_cast<powerserve::CPUBuffer*>(&tensor->get<BaseBuffer>());
                        auto *fp = cl_upload_fp();

                        CL_UPLOAD_FLOG(
                            "[CL-UPLOAD][BEFORE][TRACE_DEV] tensor_ptr={} dtype={} shape=[{}, {}, {}, {}] src_cpu={} "
                            "dst_dev={} dst_base_off={} dst_size={}",
                            (void*)tensor,
                            (int)tensor->m_dtype,
                            tensor->m_shape[0], tensor->m_shape[1], tensor->m_shape[2], tensor->m_shape[3],
                            (void*)(src_cpu ? src_cpu->m_data : nullptr),
                            dst_dev,
                            (size_t)(dst_cl ? dst_cl->get_base_offset() : 0),
                            (size_t)(dst_cl ? dst_cl->m_size : 0)
                        );

                        if (fp) dump_cpu_f32_head_fp(fp, tensor, 16, "H2D-src-head");
                    }

                    cl_backend->copy(&tmp, tensor);

                    if (hit) {
                        auto *fp = cl_upload_fp();

                        CL_UPLOAD_FLOG(
                            "[CL-UPLOAD][AFTER ][TRACE_DEV] tensor_ptr={} dst_dev={} dst_base_off={} dst_size={}",
                            (void*)tensor,
                            dst_dev,
                            (size_t)(dst_cl ? dst_cl->get_base_offset() : 0),
                            (size_t)(dst_cl ? dst_cl->m_size : 0)
                        );

                        if (fp) {
                            dump_opencl_f32_head_fp(fp, cl_backend, &tmp, 16, "H2D-dst-readback-head");
                            compare_cpu_vs_opencl_head_fp(fp, cl_backend, tensor, &tmp, 16);
                        }
                    }
                }
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
