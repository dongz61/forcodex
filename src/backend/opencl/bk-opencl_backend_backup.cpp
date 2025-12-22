// opencl_backend.cpp
#include "opencl_backend.hpp"
#include "core/logger.hpp"
#include "core/config.hpp"
#include "ggml.h"
#include <iostream>

#define CL_CHECK(call) \
    do { \
        cl_int err = (call); \
        if (err != CL_SUCCESS) { \
            POWERSERVE_LOG_ERROR("OpenCL error at {}:{} - {}: {}", \
                               __FILE__, __LINE__, #call, context->get_error_string(err)); \
            return; \
        } \
    } while (0)

namespace powerserve::opencl {

OpenCLBackend::OpenCLBackend(const ModelConfig::LLMConfig &config, const HyperParams &hparams)
    : num_threads(hparams.n_threads) {
    
    setup_default_config();
    // device_preference = hparams.opencl_device; // 假设HyperParams中有这个字段
    POWERSERVE_LOG_DEBUG("OpenCLBackend constructor called");
}

OpenCLBackend::~OpenCLBackend() {
    cleanup();
    POWERSERVE_LOG_DEBUG("OpenCLBackend destructor called");
}

void OpenCLBackend::setup_default_config() {
    thread_config.clear();
    for (int i = 0; i < num_threads; i++) {
        thread_config.emplace_back(ThreadConfig{});
    }
}

bool OpenCLBackend::initialize() {
    POWERSERVE_LOG_INFO("enter init");
    if (initialized) {
        return true;
    }
    
    // 1. 创建OpenCL上下文
    context = std::make_shared<OpenCLContext>();
    if (!context->initialize(device_preference)) {
        POWERSERVE_LOG_ERROR("Failed to initialize OpenCL context");
        return false;
    }
    
    // 2. 创建内存池
    memory_pool = std::make_shared<OpenCLMemoryPool>(context);
    
    // 3. 创建内核管理器
    kernel_manager = std::make_shared<OpenCLKernelManager>(context);
    
    // 4. 初始化内核管理器
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
    
       
    initialized = true;
    POWERSERVE_LOG_INFO("initialize end this={}, initialized={}", (void*)this, initialized);
    return true;
}

void OpenCLBackend::cleanup() {
    if (!initialized) return;
    
    std::cout << "[DEBUG] === CLEANUP TEST VERSION ===" << std::endl;
    
    // 方案1：先清理 OpenCL，再清理线程池
    std::cout << "[DEBUG] 1. Cleaning OpenCL context first..." << std::endl;
    if (context) {
        context.reset();  // 直接释放，不调用 cleanup()
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
    
    initialized = false;
    std::cout << "[DEBUG] === CLEANUP DONE ===" << std::endl;
}
// ==================== 算子实现 ====================

void OpenCLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    
    POWERSERVE_LOG_DEBUG("===== ADD OPERATOR START =====");
    
    try {
        // 获取 OpenCL 缓冲区
        auto& src0_buffer = src0->get<OpenCLBuffer>();
        auto& src1_buffer = src1->get<OpenCLBuffer>();
        auto& dst_buffer = dst->get<OpenCLBuffer>();
        
        // 获取形状和步长信息
        Shape src0_shape = src0->m_shape;
        Shape src1_shape = src1->m_shape;
        Shape dst_shape = dst->m_shape;
        
        Stride src0_stride = src0_buffer.get_stride();
        Stride src1_stride = src1_buffer.get_stride();
        Stride dst_stride = dst_buffer.get_stride();
        
        // 参数命名参考 llama.cpp
        const int ne00 = static_cast<int>(src0_shape[0]);
        const int ne01 = static_cast<int>(src0_shape[1]);
        const int ne02 = static_cast<int>(src0_shape[2]);
        const int ne03 = static_cast<int>(src0_shape[3]);
        
        const int ne10 = static_cast<int>(src1_shape[0]);
        const int ne11 = static_cast<int>(src1_shape[1]);
        const int ne12 = static_cast<int>(src1_shape[2]);
        const int ne13 = static_cast<int>(src1_shape[3]);
        
        const int ne0 = static_cast<int>(dst_shape[0]);
        const int ne1 = static_cast<int>(dst_shape[1]);
        const int ne2 = static_cast<int>(dst_shape[2]);
        const int ne3 = static_cast<int>(dst_shape[3]);
        
        const cl_ulong nb00 = static_cast<cl_ulong>(src0_stride[0]);
        const cl_ulong nb01 = static_cast<cl_ulong>(src0_stride[1]);
        const cl_ulong nb02 = static_cast<cl_ulong>(src0_stride[2]);
        const cl_ulong nb03 = static_cast<cl_ulong>(src0_stride[3]);
        
        const cl_ulong nb10 = static_cast<cl_ulong>(src1_stride[0]);
        const cl_ulong nb11 = static_cast<cl_ulong>(src1_stride[1]);
        const cl_ulong nb12 = static_cast<cl_ulong>(src1_stride[2]);
        const cl_ulong nb13 = static_cast<cl_ulong>(src1_stride[3]);
        
        const cl_ulong nb0 = static_cast<cl_ulong>(dst_stride[0]);
        const cl_ulong nb1 = static_cast<cl_ulong>(dst_stride[1]);
        const cl_ulong nb2 = static_cast<cl_ulong>(dst_stride[2]);
        const cl_ulong nb3 = static_cast<cl_ulong>(dst_stride[3]);
        
        // 获取设备缓冲区
        cl_mem src0_data = src0_buffer.get_device_buffer();
        cl_mem src1_data = src1_buffer.get_device_buffer();
        cl_mem dst_data = dst_buffer.get_device_buffer();
        
        if (!src0_data || !src1_data || !dst_data) {
            POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for add");
            return;
        }
        
        // 检查是否为行广播情况（简化版）
        // llama.cpp 的条件：src1元素数 == ne10 && src1连续 && ne00%4==0 && ne10%4==0
        bool bcast_row = false;
        
        // 首先检查元素数
        if (src1->n_elements() == static_cast<size_t>(ne10)) {
            // 简化：暂时不检查连续性，只检查4的倍数
            bool ne00_div4 = (ne00 % 4 == 0);
            bool ne10_div4 = (ne10 % 4 == 0);
            
            bcast_row = ne00_div4 && ne10_div4;
            
            POWERSERVE_LOG_DEBUG("Broadcast check: src1_elems={}, ne10={}, ne00_div4={}, ne10_div4={}, bcast_row={}",
                               src1->n_elements(), ne10, ne00_div4, ne10_div4, bcast_row);
        }
        
        // 选择正确的内核
        cl_kernel kernel = nullptr;
        std::string kernel_name;
        
        if (dst->m_dtype == DataType::FP32 && 
            src0->m_dtype == DataType::FP32 && 
            src1->m_dtype == DataType::FP32) {
            
            if (bcast_row) {
                kernel_name = "kernel_add_row";
                kernel = kernel_manager->get_kernel(kernel_name);
                POWERSERVE_LOG_DEBUG("Using kernel: kernel_add_row");
            } else {
                kernel_name = "kernel_add";
                kernel = kernel_manager->get_kernel(kernel_name);
                POWERSERVE_LOG_DEBUG("Using kernel: kernel_add");
            }
        } 
        // 可以后续添加 FP16 支持
        
        if (!kernel) {
            POWERSERVE_LOG_ERROR("Add kernel not found: {}", kernel_name);
            return;
        }
        
        POWERSERVE_LOG_DEBUG("Kernel found: {}", kernel_name);
        
        // 设置内核参数
        cl_int err;
        cl_uint arg_index = 0;
        
        // 所有版本的通用偏移（目前都设为0）
        cl_ulong offset0 = 0;
        cl_ulong offset1 = 0;
        cl_ulong offsetd = 0;
        
        if (bcast_row) {
            // 行广播版本（7个参数）
            // kernel_add_row(src0, offset0, src1, offset1, dst, offsetd, ne)
            
            // 参数0: src0 buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
            CL_CHECK(err);
            
            // 参数1: offset0
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
            CL_CHECK(err);
            
            // 参数2: src1 buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);
            CL_CHECK(err);
            
            // 参数3: offset1
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);
            CL_CHECK(err);
            
            // 参数4: dst buffer
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
            CL_CHECK(err);
            
            // 参数5: offsetd
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
            CL_CHECK(err);
            
            // 参数6: ne (元素数/4)
            int ne = ne00 / 4;
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne);
            CL_CHECK(err);
            
            POWERSERVE_LOG_DEBUG("Broadcast kernel params: ne00={}, ne={}", ne00, ne);
            
        } else {
            // 普通版本（30个参数，参考 llama.cpp）
            // kernel_add(src0, offset0, src1, offset1, dst, offsetd, 
            //            ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
            //            ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
            //            ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3)
            
            // 参数0-5: buffers 和 offsets
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);   // 0
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);   // 1
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);   // 2
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);   // 3
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);    // 4
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);   // 5
            CL_CHECK(err);
            
            // 参数6-9: ne00, ne01, ne02, ne03
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne00);  // 6
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne01);  // 7
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne02);  // 8
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne03);  // 9
            CL_CHECK(err);
            
            // 参数10-13: nb00, nb01, nb02, nb03
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb00);  // 10
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb01);  // 11
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb02);  // 12
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb03);  // 13
            CL_CHECK(err);
            
            // 参数14-17: ne10, ne11, ne12, ne13
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne10);  // 14
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne11);  // 15
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne12);  // 16
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne13);  // 17
            CL_CHECK(err);
            
            // 参数18-21: nb10, nb11, nb12, nb13
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb10);  // 18
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb11);  // 19
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb12);  // 20
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb13);  // 21
            CL_CHECK(err);
            
            // 参数22-25: ne0, ne1, ne2, ne3
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne0);  // 22
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne1);  // 23
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne2);  // 24
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne3);  // 25
            CL_CHECK(err);
            
            // 参数26-29: nb0, nb1, nb2, nb3
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb0);  // 26
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb1);  // 27
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb2);  // 28
            CL_CHECK(err);
            err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb3);  // 29
            CL_CHECK(err);
            
            POWERSERVE_LOG_DEBUG("Normal kernel params set");
        }
        
        // 计算工作组大小（严格按照 llama.cpp 的逻辑）
        if (bcast_row) {
            // 行广播版本
            int n = dst->n_elements() / 4;  // 使用 float4
            size_t global_work_size[] = {static_cast<size_t>(n), 1, 1};
            size_t local_work_size[] = {64, 1, 1};
            
            POWERSERVE_LOG_DEBUG("Enqueue bcast kernel: n={}, global_size={}, local_size={}",
                               n, n, 64);
            
            err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                         1, nullptr, global_work_size, local_work_size,
                                         0, nullptr, nullptr);
            CL_CHECK(err);
            
        } else {
            // 普通版本（严格按照 llama.cpp）
            // 注意：这里使用 ne0（dst的第一维），不是 ne00（src0的第一维）！
            unsigned int nth = std::min(64, ne0);  // 关键修正！
            
            size_t global_work_size[3] = {
                static_cast<size_t>(ne01) * nth,  // 第一维：ne01 * nth
                static_cast<size_t>(ne02),        // 第二维：ne02
                static_cast<size_t>(ne03)         // 第三维：ne03
            };
            
            size_t local_work_size[3] = {
                static_cast<size_t>(nth),  // 本地工作组第一维大小
                1,                         // 第二维
                1                          // 第三维
            };
            
            POWERSERVE_LOG_DEBUG("Enqueue normal kernel: ne0={}, ne01={}, ne02={}, ne03={}, nth={}",
                               ne0, ne01, ne02, ne03, nth);
            POWERSERVE_LOG_DEBUG("Global work size: [{}, {}, {}]",
                               global_work_size[0], global_work_size[1], global_work_size[2]);
            POWERSERVE_LOG_DEBUG("Local work size: [{}, {}, {}]",
                               local_work_size[0], local_work_size[1], local_work_size[2]);
            
            err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                         3, nullptr, global_work_size, local_work_size,
                                         0, nullptr, nullptr);
            CL_CHECK(err);
        }
        
        // 等待完成
        err = clFinish(context->get_queue());
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
            // 不返回，继续执行
        }
        
        POWERSERVE_LOG_DEBUG("OpenCL add completed successfully");
        
    } catch (const std::bad_cast& e) {
        POWERSERVE_LOG_ERROR("Invalid buffer type for add: {}", e.what());
    } catch (const std::exception& e) {
        POWERSERVE_LOG_ERROR("Exception in add: {}", e.what());
    }
    
    POWERSERVE_LOG_DEBUG("===== ADD OPERATOR END =====");
}

void OpenCLBackend::get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::get_embedding called (not implemented)");
    // TODO: 实现 get_embedding 算子
}

void OpenCLBackend::matmul(const Tensor *dst,
                           const Tensor *src0,
                           const Tensor *src1) const {
    // src0 = A (FP16) [K, M]
    // src1 = B (FP32) [N, K]  (注意：是 B^T)
    // dst  = C (FP32) [N, M]

    auto &A_buf = src0->get<OpenCLBuffer>();
    auto &B_buf = src1->get<OpenCLBuffer>();
    auto &C_buf = dst->get<OpenCLBuffer>();

    const int K = src0->m_shape[0];
    const int M = src0->m_shape[1];
    const int N = src1->m_shape[0];

    cl_kernel kernel = kernel_manager->get_kernel("kernel_mul_mat_f16_f32");

    cl_uint arg = 0;
    cl_ulong off = 0;

    cl_mem A_cl = A_buf.get_device_buffer();
    cl_mem B_cl = B_buf.get_device_buffer();
    cl_mem C_cl = C_buf.get_device_buffer();

    clSetKernelArg(kernel, arg++, sizeof(int), &M);
    clSetKernelArg(kernel, arg++, sizeof(int), &N);
    clSetKernelArg(kernel, arg++, sizeof(int), &K);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &A_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &B_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &C_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off);

    // === llama.cpp 固定配置 ===
    constexpr size_t WG_M = 16;
    constexpr size_t WG_N = 8;
    constexpr size_t OPWM = 64;
    constexpr size_t OPWN = 64;

    const size_t grid_x = (M + OPWM - 1) / OPWM;
    const size_t grid_y = (N + OPWN - 1) / OPWN;

    size_t local[2]  = { WG_M, WG_N };
    size_t global[2] = {
        grid_x * WG_M,
        grid_y * WG_N
    };

    cl_int err = clEnqueueNDRangeKernel(
        context->get_queue(),
        kernel,
        2,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("matmul enqueue failed: {}", err);
        return;
    }

    clFinish(context->get_queue());
}


void OpenCLBackend::rms_norm(
    Tensor * dst,
    const Tensor * src,
    float eps
) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }

    // ----------------------------
    // Shape
    // ----------------------------
    const auto & s = src->m_shape;

    const int ne00 = (int)s[0];
    const int ne01 = (int)s[1];
    const int ne02 = (int)s[2];
    const int ne03 = (int)s[3];

    // ----------------------------
    // Buffers
    // ----------------------------
    auto & src_buf = src->get<OpenCLBuffer>();
    auto & dst_buf = dst->get<OpenCLBuffer>();

    cl_mem src_mem = src_buf.get_device_buffer();
    cl_mem dst_mem = dst_buf.get_device_buffer();

    if (!src_mem || !dst_mem) {
        POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for rms_norm");
        return;
    }

    // ----------------------------
    // Kernel
    // ----------------------------
    cl_kernel kernel = kernel_manager->get_kernel("kernel_rms_norm");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("kernel_rms_norm not found");
        return;
    }

    // ----------------------------
    // Strides (bytes)
    // column-major layout
    // ----------------------------
    const cl_ulong nb01 = ne00 * sizeof(float);
    const cl_ulong nb02 = ne01 * nb01;
    const cl_ulong nb03 = ne02 * nb02;

    const cl_ulong offset0 = 0;
    const cl_ulong offsetd = 0;

    // ----------------------------
    // Kernel args
    // ----------------------------
    cl_uint arg = 0;

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_mem));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dst_mem));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offsetd));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne01));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne02));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne03));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &eps));

    // local float * sum (dynamic local memory)
    const size_t local_size_x = 128;  // <=256
    CL_CHECK(clSetKernelArg(
        kernel,
        arg++,
        local_size_x * sizeof(float),
        nullptr
    ));

    // ----------------------------
    // NDRange
    // ----------------------------
    const size_t local[3]  = { local_size_x, 1, 1 };
    const size_t global[3] = {
        local_size_x * (size_t)ne01,
        (size_t)ne02,
        (size_t)ne03
    };

    // ----------------------------
    // Launch
    // ----------------------------
    cl_int err = clEnqueueNDRangeKernel(
        context->get_queue(),
        kernel,
        3,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr
    );

    CL_CHECK(err);
}

void OpenCLBackend::diag_mask_inf(const Tensor *dst,
                                 const Tensor *src0,
                                 int n_past) const {
    auto &src_buf = src0->get<OpenCLBuffer>();
    auto &dst_buf = dst->get<OpenCLBuffer>();

    const int ne00 = src0->m_shape.size() > 0 ? src0->m_shape[0] : 1;
    const int ne01 = src0->m_shape.size() > 1 ? src0->m_shape[1] : 1;
    const int ne02 = src0->m_shape.size() > 2 ? src0->m_shape[2] : 1;

    cl_mem src_cl = src_buf.get_device_buffer();
    cl_mem dst_cl = dst_buf.get_device_buffer();

    cl_ulong off0 = 0;
    cl_ulong offd = 0;

    // ✅ 选择B：只在 ne01==1 且 ne00%8==0 时使用 _8 kernel（复刻 llama.cpp 假设）
    const bool can_use_vec8 = (ne01 == 1) && (ne00 % 8 == 0);

    cl_int err = CL_SUCCESS;

    if (can_use_vec8) {
        cl_kernel kernel = kernel_manager->get_kernel("kernel_diag_mask_inf_8");

        cl_uint arg = 0;
        err  = clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &src_cl);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off0);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &dst_cl);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offd);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &ne00);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &ne01);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &n_past);

        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("diag_mask_inf_8 set args failed: {}", err);
            return;
        }

        size_t global = (size_t)ne00 * (size_t)ne01 * (size_t)ne02 / 8; // ne01==1
        size_t local  = 64;

        // 你之前修的 local 保护：保留即可（避免 global<local 或非整数倍）
        const size_t *local_ptr = &local;
        if (global < local || (global % local) != 0) {
            local_ptr = nullptr;
        }

        err = clEnqueueNDRangeKernel(
            context->get_queue(),
            kernel,
            1,
            nullptr,
            &global,
            local_ptr,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("diag_mask_inf_8 enqueue failed: {}", err);
            return;
        }

    } else {
        cl_kernel kernel = kernel_manager->get_kernel("kernel_diag_mask_inf");

        cl_uint arg = 0;
        err  = clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &src_cl);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off0);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &dst_cl);
        err |= clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offd);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &ne00);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &ne01);
        err |= clSetKernelArg(kernel, arg++, sizeof(int),      &n_past);

        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("diag_mask_inf set args failed: {}", err);
            return;
        }

        size_t global[3] = { (size_t)ne00, (size_t)ne01, (size_t)ne02 };
        size_t local[3]  = { 64, 1, 1 };

        const size_t *local_ptr = local;
        if (ne00 % 64 != 0) {
            local_ptr = nullptr; // driver choose
        }

        err = clEnqueueNDRangeKernel(
            context->get_queue(),
            kernel,
            3,
            nullptr,
            global,
            local_ptr,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("diag_mask_inf enqueue failed: {}", err);
            return;
        }
    }

    clFinish(context->get_queue());
}


void OpenCLBackend::rope(const Tensor *dst,
                         const Tensor *src0,
                         const Tensor *src1,
                         const RopeParams &p,
                         const Tensor *src2 /*= nullptr*/) const {
    POWERSERVE_ASSERT(src0 && src1 && dst);
    POWERSERVE_ASSERT(src0->m_data && src1->m_data && dst->m_data);

    auto &X_buf = src0->get<OpenCLBuffer>();
    auto &P_buf = src1->get<OpenCLBuffer>();
    auto &Y_buf = dst->get<OpenCLBuffer>();

    cl_mem X_cl = X_buf.get_device_buffer();
    cl_mem P_cl = P_buf.get_device_buffer();
    cl_mem Y_cl = Y_buf.get_device_buffer();

    cl_mem S_cl = X_cl;
    if (src2) {
        POWERSERVE_ASSERT(src2->m_data);
        S_cl = src2->get<OpenCLBuffer>().get_device_buffer();
    }

    // 你们目前没有 view/offset，先按 0
    cl_ulong off0 = 0, off1 = 0, off2 = 0, offd = 0;

    const int ne00 = dim4(src0, 0);
    const int ne01 = dim4(src0, 1);
    const int ne02 = dim4(src0, 2);
    const int ne03 = dim4(src0, 3);

    const int ne0  = dim4(dst, 0);
    const int ne1  = dim4(dst, 1);
    const int ne2  = dim4(dst, 2);
    const int ne3  = dim4(dst, 3);

    // 连续 stride（byte）
    const cl_ulong nb00 = (cl_ulong)src0->element_size();
    const cl_ulong nb01 = nb00 * (cl_ulong)ne00;
    const cl_ulong nb02 = nb01 * (cl_ulong)ne01;
    const cl_ulong nb03 = nb02 * (cl_ulong)ne02;

    const cl_ulong nb0  = (cl_ulong)dst->element_size();
    const cl_ulong nb1  = nb0 * (cl_ulong)ne0;
    const cl_ulong nb2  = nb1 * (cl_ulong)ne1;
    const cl_ulong nb3  = nb2 * (cl_ulong)ne2;

    // mode 解析（对齐 llama.cpp）
    const bool is_neox   = (p.mode & 2) != 0;
    const bool is_mrope  = (p.mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_vision = (p.mode == GGML_ROPE_TYPE_VISION);
    const int  is_imrope = (p.mode == GGML_ROPE_TYPE_IMROPE) ? 1 : 0;

    // 选 kernel：名字你需要和你们 .cl 里保持一致
    // 建议命名：kernel_rope_norm_f16/f32, kernel_rope_neox_f16/f32, kernel_rope_multi_f16/f32, kernel_rope_vision_f16/f32
    cl_kernel kernel = nullptr;
    const bool is_f16 = (src0->m_dtype == DataType::FP16);
    const bool is_f32 = (src0->m_dtype == DataType::FP32);
    POWERSERVE_ASSERT(is_f16 || is_f32);

    if (is_neox) {
        kernel = kernel_manager->get_kernel(is_f16 ? "kernel_rope_neox_f16" : "kernel_rope_neox_f32");
    } else if (is_mrope && !is_vision) {
        kernel = kernel_manager->get_kernel(is_f16 ? "kernel_rope_multi_f16" : "kernel_rope_multi_f32");
    } else if (is_vision) {
        kernel = kernel_manager->get_kernel(is_f16 ? "kernel_rope_vision_f16" : "kernel_rope_vision_f32");
    } else {
        kernel = kernel_manager->get_kernel(is_f16 ? "kernel_rope_norm_f16" : "kernel_rope_norm_f32");
    }
    POWERSERVE_ASSERT(kernel);

    // set args（顺序对齐你给的 ggml_cl_rope）
    cl_uint arg = 0;
    clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &X_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off0);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &P_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off1);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &S_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &off2);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &Y_cl);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offd);

    clSetKernelArg(kernel, arg++, sizeof(int),      &ne00);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne01);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne02);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne03);

    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb00);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03);

    clSetKernelArg(kernel, arg++, sizeof(int),      &ne0);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne1);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne2);
    clSetKernelArg(kernel, arg++, sizeof(int),      &ne3);

    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb0);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb1);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb2);
    clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb3);

    clSetKernelArg(kernel, arg++, sizeof(int),      &p.n_past);
    clSetKernelArg(kernel, arg++, sizeof(int),      &p.n_dims);
    clSetKernelArg(kernel, arg++, sizeof(int),      &p.n_ctx_orig);

    clSetKernelArg(kernel, arg++, sizeof(float),    &p.freq_base);
    clSetKernelArg(kernel, arg++, sizeof(float),    &p.freq_scale);
    clSetKernelArg(kernel, arg++, sizeof(float),    &p.ext_factor);
    clSetKernelArg(kernel, arg++, sizeof(float),    &p.attn_factor);
    clSetKernelArg(kernel, arg++, sizeof(float),    &p.beta_fast);
    clSetKernelArg(kernel, arg++, sizeof(float),    &p.beta_slow);

    if (is_mrope || is_vision) {
        clSetKernelArg(kernel, arg++, sizeof(int32_t) * 4, p.sections);
    }
    if (is_mrope && !is_vision) {
        clSetKernelArg(kernel, arg++, sizeof(int), &is_imrope);
    }

    // launch（对齐 llama.cpp：global = {ne01*nth, ne02, ne03}, local = {nth,1,1}）
    const int nth = imin(64, ne00);
    size_t local[3]  = { (size_t)nth, 1, 1 };
    size_t global[3] = { (size_t)ne01 * (size_t)nth, (size_t)ne02, (size_t)ne03 };

    cl_int err = clEnqueueNDRangeKernel(
        context->get_queue(),
        kernel,
        3,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("rope enqueue failed: {}", err);
        return;
    }
    clFinish(context->get_queue());
}


void OpenCLBackend::softmax(const Tensor * dst,
                            const Tensor * src0,
                            const Tensor * src1) const {
    POWERSERVE_ASSERT(dst);
    POWERSERVE_ASSERT(src0);

    // 获取 OpenCL 缓冲区
    auto & buf0 = src0->get<OpenCLBuffer>();
    auto & bufd = dst->get<OpenCLBuffer>();
    OpenCLBuffer * buf1 = src1 ? &src1->get<OpenCLBuffer>() : nullptr;

    // 获取 Tensor 维度信息
    const int ne00 = static_cast<int>(src0->m_shape[0]);  // 向量长度
    const int ne01 = static_cast<int>(src0->m_shape[1]);  // batch 维度
    const int ne02 = static_cast<int>(src0->m_shape[2]);  // 高度
    const int ne03 = static_cast<int>(src0->m_shape[3]);  // 深度

    // src0 的 stride 计算
    const size_t es = src0->element_size();
    const cl_ulong nb01 = es * src0->m_shape[0];
    const cl_ulong nb02 = nb01 * src0->m_shape[1];
    const cl_ulong nb03 = nb02 * src0->m_shape[2];

    // dst 的 stride 计算
    const cl_ulong nb1 = es * dst->m_shape[0];
    const cl_ulong nb2 = nb1 * dst->m_shape[1];
    const cl_ulong nb3 = nb2 * dst->m_shape[2];

    // src1 的 stride 计算（如果存在）
    cl_ulong nb11 = 0, nb12 = 0, nb13 = 0;
    int ne12 = 1, ne13 = 1;
    if (src1) {
        nb11 = es * src1->m_shape[0];
        nb12 = nb11 * src1->m_shape[1];
        nb13 = nb12 * src1->m_shape[2];
        ne12 = static_cast<int>(src1->m_shape[2]);
        ne13 = static_cast<int>(src1->m_shape[3]);
    } else {
        // 如果没有 src1，使用 src0 的维度
        ne12 = ne02;
        ne13 = ne03;
    }

    // 获取 OpenCL 内存对象
    cl_mem mem_src0 = buf0.get_device_buffer();
    cl_mem mem_dst = bufd.get_device_buffer();
    cl_mem mem_src1 = buf1 ? buf1->get_device_buffer() : mem_src0;

    // 获取 OpenCL 队列
    cl_command_queue queue = context->get_queue();

    // 获取 kernel
    cl_kernel kernel = kernel_manager->get_kernel("kernel_soft_max");
    if (!kernel) {
        POWERSERVE_LOG_ERROR("Kernel 'kernel_soft_max' not found!");
        return;
    }

    // 获取 Softmax 算子的参数
    float scale = 1.0f;
    float max_bias = 0.0f;
    const float m0 = 1.0f;
    const float m1 = 1.0f;
    const int n_head_log2 = 0;

    // 设置内核参数
    int arg = 0;
    cl_ulong offset0 = 0;
    cl_ulong offset1 = 0;
    cl_ulong offset2 = 0;
    cl_ulong offsetd = 0;

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_src0));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_src1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_src0));  // src2 使用 src0
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mem_dst));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offsetd));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb03));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne12));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &ne13));

    // 添加 src1 的 stride 参数
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb13));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &nb3));

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &scale));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &max_bias));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &m0));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &m1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), &n_head_log2));

    // 设置本地内存参数（非常重要！）
    const size_t local_work_size = std::min<size_t>(32, ne00);
    const size_t local_mem_size = local_work_size * sizeof(float);
    CL_CHECK(clSetKernelArg(kernel, arg++, local_mem_size, nullptr));

    // 设置全局工作项和局部工作项大小
    const size_t global[] = { 
        static_cast<size_t>(ne01) * local_work_size, 
        static_cast<size_t>(ne02), 
        static_cast<size_t>(ne03) 
    };
    const size_t local[] = { local_work_size, 1, 1 };

    // 执行内核
    CL_CHECK(clEnqueueNDRangeKernel(
        queue, kernel,
        3, nullptr, global, local,
        0, nullptr, nullptr
    ));
}

void OpenCLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::permute called (not implemented)");
    // TODO: 实现 permute 算子
}

void OpenCLBackend::cont(const Tensor *out, const Tensor *x) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::cont called (not implemented)");
    // TODO: 实现 cont 算子
}

void OpenCLBackend::softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, 
                               float scale, float max_bias) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::softmax_ext called (not implemented)");
    // TODO: 实现 softmax_ext 算子
}

bool OpenCLBackend::is_contiguous(const Tensor *tensor, int n) const {
    if (!initialized) return false;
    // 简化实现：暂时假设所有张量都是连续的
    return true;
}

int OpenCLBackend::get_n_tasks(std::shared_ptr<OpNode> op) {
    if (!initialized) return 1;
    // 简化实现：对于 OpenCL，通常使用 GPU 并行，所以返回 1
    return 1;
}

enum ggml_type OpenCLBackend::get_vec_dot_type(const Tensor *tensor) {
    // 转换 DataType 到 ggml_type
    switch (tensor->m_dtype) {
        case DataType::FP32:
            return GGML_TYPE_F32;
        case DataType::FP16:
            return GGML_TYPE_F16;
        case DataType::GGML_Q4_0:
            return GGML_TYPE_Q4_0;
        case DataType::GGML_Q8_0:
            return GGML_TYPE_Q8_0;
        default:
            return GGML_TYPE_F32;
    }
}

void OpenCLBackend::silu(const Tensor* dst, const Tensor* src0) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    
    POWERSERVE_LOG_DEBUG("===== SILU OPERATOR START =====");
    
    try {
        // 获取缓冲区
        auto& src0_buffer = src0->get<OpenCLBuffer>();
        auto& dst_buffer = dst->get<OpenCLBuffer>();
        
        // 检查形状一致性
        Shape src0_shape = src0->m_shape;
        Shape dst_shape = dst->m_shape;
        
        if (src0_shape != dst_shape) {
            POWERSERVE_LOG_ERROR("SiLU: src0 and dst shapes must match");
            return;
        }
        
        // 获取设备缓冲区
        cl_mem src0_data = src0_buffer.get_device_buffer();
        cl_mem dst_data = dst_buffer.get_device_buffer();
        
        if (!src0_data || !dst_data) {
            POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for silu");
            return;
        }
        
        // 选择正确的内核（参考 llama.cpp 的逻辑）
        cl_kernel kernel = nullptr;
        int n = static_cast<int>(dst->n_elements());
        
        // 检查是否可以使用优化的4元素版本
        if (n % 4 == 0 && dst->m_dtype == DataType::FP32 && src0->m_dtype == DataType::FP32) {
            kernel = kernel_manager->get_kernel("kernel_silu_4");
            n /= 4;
            POWERSERVE_LOG_DEBUG("Using optimized silu_4 kernel, n={}", n);
        } else {
            kernel = kernel_manager->get_kernel("kernel_silu");
            POWERSERVE_LOG_DEBUG("Using standard silu kernel, n={}", n);
        }
        
        if (!kernel) {
            POWERSERVE_LOG_ERROR("SiLU kernel not found");
            return;
        }
        
        // 设置内核参数
        cl_int err;
        cl_uint arg_index = 0;
        
        // 参数 0: src0 buffer
        cl_ulong offset0 = 0;  // 暂时设为0，后续可支持视图偏移
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to set arg 0: {}", 
                               context->get_error_string(err));
            return;
        }
        
        // 参数 1: offset0
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
        
        // 参数 2: dst buffer
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
        
        // 参数 3: offsetd
        cl_ulong offsetd = 0;
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
        
        // 计算工作组大小（参考 llama.cpp 的逻辑）
        size_t global_work_size[] = {static_cast<size_t>(n), 1, 1};
        size_t local_work_size[] = {64, 1, 1};
        
        POWERSERVE_LOG_DEBUG("Global work size: {}, local work size: {}", 
                           global_work_size[0], local_work_size[0]);
        
        // 执行内核
        err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                     1,  // 1维工作组
                                     nullptr,
                                     global_work_size,
                                     local_work_size,
                                     0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to enqueue silu kernel: {}", 
                               context->get_error_string(err));
            return;
        }
        
        // 等待完成
        err = clFinish(context->get_queue());
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
        }
        
        POWERSERVE_LOG_DEBUG("OpenCL silu completed: shape=[{}, {}, {}, {}]", 
                           dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[3]);
        
    } catch (const std::bad_cast& e) {
        POWERSERVE_LOG_ERROR("Invalid buffer type for silu: {}", e.what());
    } catch (const std::exception& e) {
        POWERSERVE_LOG_ERROR("Exception in silu: {}", e.what());
    }
    
    POWERSERVE_LOG_DEBUG("===== SILU OPERATOR END =====");
}
void OpenCLBackend::gelu(const Tensor* dst, const Tensor* src0) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    
    POWERSERVE_LOG_DEBUG("===== GELU OPERATOR START =====");
    
    try {
        // 获取缓冲区
        auto& src0_buffer = src0->get<OpenCLBuffer>();
        auto& dst_buffer = dst->get<OpenCLBuffer>();
        
        // 检查形状一致性
        Shape src0_shape = src0->m_shape;
        Shape dst_shape = dst->m_shape;
        
        if (src0_shape != dst_shape) {
            POWERSERVE_LOG_ERROR("GELU: src0 and dst shapes must match");
            return;
        }
        
        // 获取设备缓冲区
        cl_mem src0_data = src0_buffer.get_device_buffer();
        cl_mem dst_data = dst_buffer.get_device_buffer();
        
        if (!src0_data || !dst_data) {
            POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for gelu");
            return;
        }
        
        // 选择正确的内核（与 silu 相同逻辑）
        cl_kernel kernel = nullptr;
        int n = static_cast<int>(dst->n_elements());
        
        // 检查是否可以使用优化的4元素版本
        if (n % 4 == 0 && dst->m_dtype == DataType::FP32 && src0->m_dtype == DataType::FP32) {
            kernel = kernel_manager->get_kernel("kernel_gelu_4");
            n /= 4;
            POWERSERVE_LOG_DEBUG("Using optimized gelu_4 kernel, n={}", n);
        } else {
            kernel = kernel_manager->get_kernel("kernel_gelu");
            POWERSERVE_LOG_DEBUG("Using standard gelu kernel, n={}", n);
        }
        
        if (!kernel) {
            POWERSERVE_LOG_ERROR("GELU kernel not found");
            return;
        }
        
        // 设置内核参数（与 silu 完全相同）
        cl_int err;
        cl_uint arg_index = 0;
        
        // 参数 0: src0 buffer
        cl_ulong offset0 = 0;
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to set arg 0: {}", 
                               context->get_error_string(err));
            return;
        }
        
        // 参数 1: offset0
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
        
        // 参数 2: dst buffer
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &dst_data);
        
        // 参数 3: offsetd
        cl_ulong offsetd = 0;
        err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offsetd);
        
        // 计算工作组大小（与 silu 完全相同）
        size_t global_work_size[] = {static_cast<size_t>(n), 1, 1};
        size_t local_work_size[] = {64, 1, 1};
        
        POWERSERVE_LOG_DEBUG("Global work size: {}, local work size: {}", 
                           global_work_size[0], local_work_size[0]);
        
        // 执行内核
        err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                     1,  // 1维工作组
                                     nullptr,
                                     global_work_size,
                                     local_work_size,
                                     0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_ERROR("Failed to enqueue gelu kernel: {}", 
                               context->get_error_string(err));
            return;
        }
        
        // 等待完成
        err = clFinish(context->get_queue());
        if (err != CL_SUCCESS) {
            POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
        }
        
        POWERSERVE_LOG_DEBUG("OpenCL gelu completed: shape=[{}, {}, {}, {}]", 
                           dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[3]);
        
    } catch (const std::bad_cast& e) {
        POWERSERVE_LOG_ERROR("Invalid buffer type for gelu: {}", e.what());
    } catch (const std::exception& e) {
        POWERSERVE_LOG_ERROR("Exception in gelu: {}", e.what());
    }
    
    POWERSERVE_LOG_DEBUG("===== GELU OPERATOR END =====");
}
void OpenCLBackend::copy(const Tensor *dst, const Tensor *src) const {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    
    POWERSERVE_LOG_DEBUG("=== Using OpenCL KERNEL for copy ===");

    // 在 llama.cpp 中，copy 是 src0 -> src1，而你的接口是 src -> dst
    // 所以：src0 = src, src1 = dst
    const Tensor* src0 = src;
    const Tensor* src1 = dst;
    
    // 获取形状信息
    const int ne00 = static_cast<int>(src0->m_shape[0]);
    const int ne01 = static_cast<int>(src0->m_shape[1]);
    const int ne02 = static_cast<int>(src0->m_shape[2]);
    const int ne03 = static_cast<int>(src0->m_shape[3]);
    
    const int ne10 = static_cast<int>(src1->m_shape[0]);
    const int ne11 = static_cast<int>(src1->m_shape[1]);
    const int ne12 = static_cast<int>(src1->m_shape[2]);
    const int ne13 = static_cast<int>(src1->m_shape[3]);
    
    // 获取 OpenCL 缓冲区
    auto& src0_buffer = src0->get<OpenCLBuffer>();
    auto& src1_buffer = src1->get<OpenCLBuffer>();
    
    // 获取步长信息
    const Stride src0_stride = src0_buffer.get_stride();
    const Stride src1_stride = src1_buffer.get_stride();
    
    const cl_ulong nb00 = static_cast<cl_ulong>(src0_stride[0]);
    const cl_ulong nb01 = static_cast<cl_ulong>(src0_stride[1]);
    const cl_ulong nb02 = static_cast<cl_ulong>(src0_stride[2]);
    const cl_ulong nb03 = static_cast<cl_ulong>(src0_stride[3]);
    
    const cl_ulong nb10 = static_cast<cl_ulong>(src1_stride[0]);
    const cl_ulong nb11 = static_cast<cl_ulong>(src1_stride[1]);
    const cl_ulong nb12 = static_cast<cl_ulong>(src1_stride[2]);
    const cl_ulong nb13 = static_cast<cl_ulong>(src1_stride[3]);
    
    // 获取设备缓冲区
    cl_mem src0_data = src0_buffer.get_device_buffer();
    cl_mem src1_data = src1_buffer.get_device_buffer();
    
    if (!src0_data || !src1_data) {
        POWERSERVE_LOG_ERROR("Invalid OpenCL buffers for copy");
        return;
    }
    
    // 选择正确的内核
    cl_kernel kernel = nullptr;
    
    // 根据数据类型选择内核
    if (src0->m_dtype == DataType::FP32 && src1->m_dtype == DataType::FP32) {
        kernel = kernel_manager->get_kernel("kernel_cpy_f32_f32");
    } else if (src0->m_dtype == DataType::FP16 && src1->m_dtype == DataType::FP16) {
        kernel = kernel_manager->get_kernel("kernel_cpy_f16_f16");
    } else if (src0->m_dtype == DataType::FP16 && src1->m_dtype == DataType::FP32) {
        kernel = kernel_manager->get_kernel("kernel_cpy_f16_f32");
    } else if (src0->m_dtype == DataType::FP32 && src1->m_dtype == DataType::FP16) {
        kernel = kernel_manager->get_kernel("kernel_cpy_f32_f16");
    } else {
        POWERSERVE_LOG_ERROR("Unsupported copy types: src={}, dst={}",
                           static_cast<int>(src0->m_dtype),
                           static_cast<int>(src1->m_dtype));
        return;
    }
    
    if (!kernel) {
        POWERSERVE_LOG_ERROR("Copy kernel not found");
        return;
    }
    
    // 设置内核参数
    cl_uint arg_index = 0;
    cl_int err;
    
    // 参数 0: src0 buffer
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src0_data);
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to set arg {}: {}", arg_index-1, 
                           context->get_error_string(err));
        return;
    }
    
    // 参数 1: offset0
    cl_ulong offset0 = 0;
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset0);
    
    // 参数 2: src1 buffer
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &src1_data);
    
    // 参数 3: offset1
    cl_ulong offset1 = 0;
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &offset1);
    
    // 参数 4-7: ne00, ne01, ne02, ne03
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne00);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne01);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne02);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne03);
    
    // 参数 8-11: nb00, nb01, nb02, nb03
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb00);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb01);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb02);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb03);
    
    // 参数 12-15: ne10, ne11, ne12, ne13
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne10);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne11);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne12);
    err = clSetKernelArg(kernel, arg_index++, sizeof(int), &ne13);
    
    // 参数 16-19: nb10, nb11, nb12, nb13
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb10);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb11);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb12);
    err = clSetKernelArg(kernel, arg_index++, sizeof(cl_ulong), &nb13);
    
    // 计算工作组大小（按照 llama.cpp 的方式）
    const int nth = std::min(64, ne00);
    
    size_t global_work_size[3] = {
        static_cast<size_t>(ne01 * nth),
        static_cast<size_t>(ne02),
        static_cast<size_t>(ne03)
    };
    
    size_t local_work_size[3] = {
        static_cast<size_t>(nth),
        1,
        1
    };
    
    // 执行内核
    err = clEnqueueNDRangeKernel(context->get_queue(), kernel,
                                 3,
                                 nullptr,
                                 global_work_size,
                                 local_work_size,
                                 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_ERROR("Failed to enqueue copy kernel: {}", 
                           context->get_error_string(err));
        return;
    }
    
    // 等待完成
    err = clFinish(context->get_queue());
    if (err != CL_SUCCESS) {
        POWERSERVE_LOG_WARN("clFinish failed: {}", context->get_error_string(err));
    }
    
    POWERSERVE_LOG_DEBUG("OpenCL copy completed: {} -> {}, shape=[{}, {}, {}, {}]",
                       static_cast<int>(src0->m_dtype),
                       static_cast<int>(src1->m_dtype),
                       ne00, ne01, ne02, ne03);
}

void OpenCLBackend::print(const Tensor *x, size_t size) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::print called (not implemented)");
    // TODO: 实现 print 算子
}

void OpenCLBackend::reset_kv_batch_size(const size_t batch_size) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::reset_kv_batch_size called (not implemented)");
    // TODO: 实现 reset_kv_batch_size
}

void OpenCLBackend::add_cache(const Tensor *k, const Tensor *v, size_t L, 
                             const std::vector<int> &pos, size_t head_id) {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::add_cache called (not implemented)");
    // TODO: 实现 add_cache 算子
}

void OpenCLBackend::transpose(const Tensor *out, const Tensor *x) const {
    if (!initialized) return;
    POWERSERVE_LOG_DEBUG("OpenCLBackend::transpose called (not implemented)");
    // TODO: 实现 transpose 算子
}

// ==================== 计划调度 ====================

void OpenCLBackend::plan(std::vector<std::shared_ptr<OpNode>> &ops) {
    if (!initialized) {
        POWERSERVE_LOG_ERROR("OpenCL backend not initialized");
        return;
    }
    
    // 简单的实现，后续可以优化
    size_t max_work_size = 0;
    for (auto op : ops) {
        // 估算工作内存大小
        size_t cur = 1024 * 1024; // 1MB 作为起始
        max_work_size = std::max(max_work_size, cur);
    }
    
    setup_work_data(max_work_size);
}

void OpenCLBackend::setup_work_data(size_t work_size) {
    if (work_size <= work_data.size) {
        return;
    }
    
    // 添加一些padding
    if (work_size > 0) {
        work_size += 64 * num_threads; // 缓存行对齐
    }
    
    work_data.buffer.resize(work_size);
    work_data.size = work_size;
    
    POWERSERVE_LOG_DEBUG("OpenCL work data size set to {} bytes", work_size);
}

// ==================== 内存管理 ====================

cl_mem OpenCLBackend::allocate_device_memory(size_t size) {
    if (!memory_pool) {
        POWERSERVE_LOG_ERROR("Memory pool not initialized");
        return nullptr;
    }
    
    cl_mem buffer = memory_pool->allocate(size);
    if (!buffer) {
        POWERSERVE_LOG_ERROR("Failed to allocate {} bytes", size);
    }
    
    return buffer;
}

void OpenCLBackend::free_device_memory(cl_mem buffer) {
    if (memory_pool && buffer) {
        memory_pool->free(buffer);
    }
}

bool OpenCLBackend::copy_to_device(cl_mem dst, const void* src, size_t size) {
    if (!memory_pool || !dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_to_device");
        return false;
    }
    
    return memory_pool->copy_host_to_device(dst, src, size);
}

bool OpenCLBackend::copy_to_host(void* dst, cl_mem src, size_t size) {
    if (!memory_pool || !dst || !src) {
        POWERSERVE_LOG_ERROR("Invalid arguments for copy_to_host");
        return false;
    }
    
    return memory_pool->copy_device_to_host(dst, src, size);
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

// ==================== 线程池管理 ====================

void OpenCLBackend::reset_threadpool() {
    if (thread_pool) {
        thread_pool.reset();
        POWERSERVE_LOG_DEBUG("OpenCL thread pool reset");
    }
}

// ==================== 设备信息 ====================

size_t OpenCLBackend::get_device_memory() const {
    if (!context) return 0;
    return context->get_device_info().global_mem_size;
}

// ==================== 私有辅助函数 ====================


cl_mem OpenCLBackend::get_cl_buffer(const Tensor* tensor) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // 检查是否已有缓存
    auto it = tensor_buffers_.find(tensor);
    if (it != tensor_buffers_.end()) {
        return it->second;
    }
    
    try {
        // 尝试获取 OpenCLBuffer
        auto& buffer = tensor->get<OpenCLBuffer>();
        cl_mem device_buffer = buffer.get_device_buffer();
        
        // 缓存
        tensor_buffers_[tensor] = device_buffer;
        return device_buffer;
        
    } catch (const std::bad_cast& e) {
        // 不是 OpenCLBuffer
        POWERSERVE_LOG_DEBUG("Tensor is not an OpenCLBuffer");
        return nullptr;
    }
}

cl_kernel OpenCLBackend::get_kernel_for_type(const std::string& base_name, DataType dtype) const {
    if (!kernel_manager) {
        return nullptr;
    }
    
    std::string kernel_name;
    switch (dtype) {
        case DataType::FP32:
            kernel_name = base_name + "_f32";
            break;
        case DataType::FP16:
            kernel_name = base_name + "_f16";
            break;
        default:
            POWERSERVE_LOG_ERROR("Unsupported dtype for kernel: {}", static_cast<int>(dtype));
            return nullptr;
    }
    
    return kernel_manager->get_kernel(kernel_name);
}

void OpenCLBackend::set_kernel_args_from_tensors(cl_kernel kernel, 
                                                 const std::vector<const Tensor*>& tensors) const {
    if (!kernel) {
        return;
    }
    
    cl_uint arg_index = 0;
    for (const auto* tensor : tensors) {
        cl_mem buffer = get_cl_buffer(tensor);
        if (buffer) {
            clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &buffer);
        }
    }
}

} // namespace powerserve::opencl