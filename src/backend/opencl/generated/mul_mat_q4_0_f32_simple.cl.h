// Auto-generated from mul_mat_q4_0_f32_simple.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string mul_mat_q4_0_f32_simple_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK4_0 32
#define BS4_0 18   // sizeof(block_q4_0) = 2 + 16

// Read ggml block layout directly:
// block_q4_0: [half d][uint8 qs[16]]; each byte packs 2 q4 (low nibble first)
kernel void kernel_mul_mat_q4_0_f32_simple(
    global const uchar * w,     // ggml raw buffer
    ulong off_w,
    global const float * x,
    ulong off_x,
    global float * dst,
    ulong off_dst,
    int K, int N, int M,
    ulong nb_w1,
    ulong nb_x1,
    ulong nb_dst1
) {
    const int n = (int)get_global_id(0);
    const int m = (int)get_global_id(1);
    if (n >= N || m >= M) return;

    global const uchar * w_col = (global const uchar *)((global const char *)w + off_w + (ulong)n * nb_w1);
    global const float * x_col = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK4_0;

    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_col + (ulong)b * (ulong)BS4_0;

        const float d = vload_half(0, (global const half *)blk);

        float acc = 0.0f;
        const int k0 = b * QK4_0;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uchar byte = blk[2 + i];
            const int q0 = ((int)(byte & 0x0F)) - 8;        // k = k0 + 2*i
            const int q1 = ((int)((byte >> 4) & 0x0F)) - 8; // k = k0 + 2*i + 1

            const int kk = k0 + 2*i;
            acc += (float)q0 * x_col[kk] + (float)q1 * x_col[kk + 1];
        }
        sum += acc * d;
    }

    *out_ptr = sum;
}

)CLC";

} // namespace powerserve::opencl::embedded
