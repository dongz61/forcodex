// Auto-generated from rms_norm.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string rms_norm_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp64
typedef double rms_acc_t;
#else
typedef float rms_acc_t;
#endif

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
        local rms_acc_t * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    rms_acc_t sumf = 0;

    // 并行求和
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        float4 v = x[i00];
        sumf += (rms_acc_t)(v.s0 * v.s0 + v.s1 * v.s1 + v.s2 * v.s2 + v.s3 * v.s3);
    }

    // 使用共享内存进行 work-group 减少，而不是 subgroup
    int lid = get_local_id(0);
    sum[lid] = sumf;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 树形减少
    for (uint stride = get_local_size(0)/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum[lid] += sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        // 处理不能被4整除的剩余元素
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += (rms_acc_t)(x_scalar[i] * x_scalar[i]);
        }
        sum[0] /= (rms_acc_t)ne00;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const rms_acc_t mean  = sum[0];
    const float scale = (float)(1.0 / sqrt(mean + (rms_acc_t)eps));
    
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
        local rms_acc_t * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;
    
    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);
    
    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    global char * f_base = (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);
    global float4 * f = (global float4 *) f_base;
    global float * f_scalar = (global float *) f_base;
    
    int lid = get_local_id(0);
    rms_acc_t sumf = 0;
    
    // 并行求和
    for (int i00 = lid; i00 < ne00/4; i00 += get_local_size(0)) {
        float4 v = x[i00];
        sumf += (rms_acc_t)(v.s0 * v.s0 + v.s1 * v.s1 + v.s2 * v.s2 + v.s3 * v.s3);
    }
    
    sum[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 树形减少
    for (uint stride = get_local_size(0)/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum[lid] += sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += (rms_acc_t)(x_scalar[i] * x_scalar[i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rms_acc_t mean = sum[0] / (rms_acc_t)ne00;
    float scale = (float)(1.0 / sqrt(mean + (rms_acc_t)eps));
    
    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = lid; i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
    }
    if (lid == 0) {
        global float * y_scalar = (global float *) y;
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = (x_scalar[i00] * scale) * f_scalar[i00 % ne10];
        }
    }
}

)CLC";

} // namespace powerserve::opencl::embedded
