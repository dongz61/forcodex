// Auto-generated from mul_mat_q4_0_f32_simple.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string mul_mat_q4_0_f32_simple_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK4_0 32
#define BS4_0 18   // sizeof(block_q4_0) = 2 + 16

// roundf(x) semantics: ties-away-from-zero
inline int round_away_from_zero(float x) {
    float ax = fabs(x);
    int r = (int)floor(ax + 0.5f);
    return x < 0.0f ? -r : r;
}

// GGML-semantics Q4_0(weight) x F32(act):
// - act dynamically quantized per 32 elements to Q8_0 (dx stored as FP16)
// - dot(q4,q8) scaled by dw16*dx16
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

    global const uchar * w_row = (global const uchar *)((global const char *)w + off_w + (ulong)n * nb_w1);
    global const float * x_col = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK4_0;
    float sum = 0.0f;

    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_row + (ulong)b * (ulong)BS4_0;

        // weight scale dw stored as fp16 in ggml block_q4_0
        const float dw16 = vload_half(0, (global const half *)blk);

        const int k0 = b * QK4_0;

        // amax for x-block
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_0; ++i) {
            float ax = fabs(x_col[k0 + i]);
            amax = fmax(amax, ax);
        }
        if (amax == 0.0f) continue;

        // x quant scale d = amax/127, store as fp16 for dx16 (match ggml q8_0 quant)
        const float d  = amax * (1.0f / 127.0f);
        const float id = 1.0f / d;

        half dh_tmp;
        vstore_half(d, 0, &dh_tmp);
        const float dx16 = vload_half(0, &dh_tmp);

        // dot(q4,q8)
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uchar byte = blk[2 + i];
            const int qw0 = ((int)(byte & 0x0F)) - 8;        // [-8..7]
            const int qw1 = ((int)((byte >> 4) & 0x0F)) - 8; // [-8..7]

            const int kk = k0 + 2*i;

            int qx0 = round_away_from_zero(x_col[kk]     * id);
            int qx1 = round_away_from_zero(x_col[kk + 1] * id);
            qx0 = max(-127, min(127, qx0));
            qx1 = max(-127, min(127, qx1));

            sumi += qw0 * qx0 + qw1 * qx1;
        }

        sum += (float)sumi * (dw16 * dx16);
    }

    *out_ptr = sum;
}

)CLC";

} // namespace powerserve::opencl::embedded
