// Auto-generated from mul_mat_q4_0_f32_simple.cl
#pragma once

#include <string>

namespace powerserve::opencl::embedded {

const std::string mul_mat_q4_0_f32_simple_cl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK4_0 32
#define BS4_0 18   // sizeof(block_q4_0) = 2 + 16

static inline float fp16_roundtrip(float x) {
    half h;
    vstore_half(x, 0, &h);
    return vload_half(0, &h);
}

static inline int round_ties_away_from_zero(float x) {
    return (x >= 0.0f) ? (int)floor(x + 0.5f) : (int)ceil(x - 0.5f);
}

static inline char quantize_to_q8(float x, float id) {
    int q = (int)rint(x * id);
    q = clamp(q, -127, 127);
    return (char)q;
}

// Read ggml block layout directly:
// block_q4_0: [half d][uint8 qs[16]]; each byte packs 2 q4 (low nibble first)
// ggml vecdot path is effectively: q4(weight) dot q8(act) * (dw16 * dx16)
kernel void kernel_mul_mat_q4_0_f32_simple(
    global const uchar * w,
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
    global const float * x_row = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK4_0;

    float sumf = 0.0f;

    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_row + (ulong)b * (ulong)BS4_0;

        const float dw = vload_half(0, (global const half *)blk);
        const int k0 = b * QK4_0;

        // ---- dynamic quantize x block to q8_0 (ggml semantics) ----
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_0; ++i) {
            float xv = x_row[k0 + i];
            amax = fmax(amax, fabs(xv));
        }

        float id = 0.0f;
        float dx = 0.0f;
        if (amax > 0.0f) {
            id = 127.0f / amax;
            dx = fp16_roundtrip(amax / 127.0f);
        }

        int sumi = 0;

        // 16 bytes -> 32 q4 values
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uchar byte = blk[2 + i];

            const int qw0 = ((int)(byte & 0x0F)) - 8;
            const int qw1 = ((int)((byte >> 4) & 0x0F)) - 8;

            const int kk = k0 + (i * 2);

            const char qx0 = quantize_to_q8(x_row[kk + 0], id);
            const char qx1 = quantize_to_q8(x_row[kk + 1], id);

            sumi += (int)qx0 * qw0;
            sumi += (int)qx1 * qw1;
        }

        sumf += ((float)sumi) * dx * dw;
    }

    *out_ptr = sumf;
}

)CLC";

} // namespace powerserve::opencl::embedded
