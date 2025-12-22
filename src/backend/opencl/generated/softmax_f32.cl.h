// Auto-generated from softmax_f32.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string softmax_f32_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_soft_max(
    global char * src0,
    ulong offset0,
    global char * src1,
    ulong offset1,
    global char * src2,
    ulong offset2,
    global char * dst,
    ulong offsetd,
    int ne00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne12,
    int ne13,
    ulong nb11,
    ulong nb12,
    ulong nb13,
    ulong nb1,
    ulong nb2,
    ulong nb3,
    float scale,
    float max_bias,
    float m0,
    float m1,
    int n_head_log2,
    local float * lbuf   // 👈 本地缓冲
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int lid = get_local_id(0);
    int lsz = get_local_size(0);

    global float * psrc0 =
        (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    global float * pmask =
        (src1 != src0)
        ? (global float *)(src1 + i01*nb11 + (i02%ne12)*nb12 + (i03%ne13)*nb13)
        : 0;

    global float * psrc2 =
        (src2 != src0) ? (global float *)src2 : 0;

    global float * pdst =
        (global float *)(dst + i01*nb1 + i02*nb2 + i03*nb3);

    /* ---------- ALiBi ---------- */
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        int h = i02;
        float base = h < n_head_log2 ? m0 : m1;
        int   expn = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
        slope = pow(base, expn);
    }

    /* ---------- max ---------- */
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i = lid; i < ne00; i += lsz) {
        float v = psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f);
        lmax = fmax(lmax, v);
    }

    lbuf[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s) {
            lbuf[lid] = fmax(lbuf[lid], lbuf[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float maxv = lbuf[0];

    /* ---------- sum ---------- */
    float lsum = 0.0f;
    for (int i = lid; i < ne00; i += lsz) {
        float e = exp(psrc0[i]*scale +
                      (pmask ? slope*pmask[i] : 0.0f) - maxv);
        pdst[i] = e;
        lsum += e;
    }

    lbuf[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s) {
            lbuf[lid] += lbuf[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float sum = lbuf[0];
    if (psrc2) {
        sum += exp(psrc2[i02] - maxv);
    }

    for (int i = lid; i < ne00; i += lsz) {
        pdst[i] /= sum;
    }
}

)CLC";

} // namespace powerserve::opencl::embedded
