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

#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"
#include "backend/opencl/opencl_buffer.hpp"  
#include "backend/opencl/opencl_backend.hpp"



namespace powerserve {

struct Executor {
public:
    Platform &m_platform;
    Graph &m_graph;

public:
    Executor(Platform &platform, Graph &graph) : m_platform(platform), m_graph(graph) {}

public:
    void allocate_buffers();
    void run();
    void plan();

private:
    template <typename T>
    void create_cpu_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data =
                CPUBuffer::create_buffer_view<T>(tensor->tensor_view()->parent->get<CPUBuffer>(), tensor->m_shape);
        } else {
            tensor->m_data = CPUBuffer::create_buffer<T>(tensor->m_shape);
        }
    }

    // ziqian：增加OpenCL buffer创建
    template <typename T>
    void create_opencl_buffer_for_tensor(Tensor* tensor) {
        POWERSERVE_ASSERT(tensor != nullptr);

        // If this tensor is a view, allocate as a sub-buffer view.
        // NOTE: In the graph codepath, view tensor allocation uses TensorNode metadata (parent pointer & shape).
        // For raw Tensor*, we cannot reliably reconstruct the parent relationship.
        // So we only support non-view tensors here.

        // base buffer must be created by OpenCLBackend (it owns memory_pool)
        auto* backend = m_platform.get_backend(m_graph.m_model_id);
        auto* ocl = dynamic_cast<opencl::OpenCLBackend*>(backend);
        if (!ocl) {
            POWERSERVE_ABORT("create_opencl_buffer_for_tensor: backend is not OpenCLBackend");
        }

        auto ocl_buf = ocl->create_buffer(tensor->m_shape, tensor->m_dtype);
        tensor->m_data = std::static_pointer_cast<BaseBuffer>(ocl_buf);
    }

    template <typename T>
    void create_opencl_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data =
                opencl::OpenCLBuffer::create_buffer_view<T>(
                    tensor->tensor_view()->parent->get<opencl::OpenCLBuffer>(),
                    tensor->m_shape);
        } else {
            create_opencl_buffer_for_tensor<T>(tensor.get());
        }
    }
    // ziqian：end
};

} // namespace powerserve
