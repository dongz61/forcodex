// Auto-generated from rms_norm.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string rms_norm_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 移除所有 subgroup 扩展和宏定义
// 只需要最基本的 OpenCL 1.2 功能

//------------------------------------------------------------------------------
// rms_norm - 简化版本，不使用 subgroup
//------------------------------------------------------------------------------
kernel void kernel_rms_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;

    // 并行求和
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    
    // 使用共享内存进行 work-group 减少，而不是 subgroup
    local float local_sum[256]; // 假设最大工作组大小256
    int lid = get_local_id(0);
    local_sum[lid] = all_sum;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 树形减少
    for (uint stride = get_local_size(0)/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        // 处理不能被4整除的剩余元素
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            local_sum[0] += x_scalar[i] * x_scalar[i];
        }
        local_sum[0] /= ne00;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const float mean  = local_sum[0];
    const float scale = 1.0f/sqrt(mean + eps);
    
    global float4 * y = (global float4 *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float * y_scalar = (global float *) y;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = x[i00] * scale;
    }
    if (lid == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = x_scalar[i00] * scale;
        }
    }
}

//------------------------------------------------------------------------------
// rms_norm_mul - 简化版本
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;
    
    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);
    
    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);
    
    int lid = get_local_id(0);
    local float local_sum[256];
    float sumf = 0;
    
    // 并行求和
    for (int i00 = lid; i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    
    local_sum[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 树形减少
    for (uint stride = get_local_size(0)/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mean = local_sum[0] / ne00;
    float scale = 1.0f/sqrt(mean + eps);
    
    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = lid; i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
    }
}
)CLC";

} // namespace powerserve::opencl::embedded
