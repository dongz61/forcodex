#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK8_0 32
#define BS8_0 34   // sizeof(block_q8_0) = 2 + 32

// roundf(x) semantics: ties-away-from-zero (match ggml quantize_row_q8_0_ref)
inline int round_away_from_zero(float x) {
    float ax = fabs(x);
    int r = (int)floor(ax + 0.5f);
    return x < 0.0f ? -r : r;
}

// GGML-semantics Q8_0(weight) x F32(act):
// - act dynamically quantized per 32 elements to Q8_0 (dx stored as FP16)
// - dot(int8,int8) scaled by dw16*dx16
kernel void kernel_mul_mat_q8_0_f32_simple(
    global const uchar * w,     // ggml raw buffer
    ulong off_w,                // byte offset
    global const float * x,
    ulong off_x,
    global float * dst,
    ulong off_dst,
    int K, int N, int M,
    ulong nb_w1,                // weight stride bytes for dim1
    ulong nb_x1,                // x stride bytes for dim1
    ulong nb_dst1               // dst stride bytes for dim1
) {
    const int n = (int)get_global_id(0); // [0..N)
    const int m = (int)get_global_id(1); // [0..M)
    if (n >= N || m >= M) return;

    global const uchar * w_row = (global const uchar *)((global const char *)w + off_w + (ulong)n * nb_w1);
    global const float * x_col = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK8_0;

    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_row + (ulong)b * (ulong)BS8_0;

        // dw stored as fp16 in ggml block
        const float dw16 = vload_half(0, (global const half *)blk);

        const int k0 = b * QK8_0;

        // amax = max(|x|)
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float ax = fabs(x_col[k0 + i]);
            amax = fmax(amax, ax);
        }
        if (amax == 0.0f) continue;

        // d = amax/127 in fp32 for id; but stored as fp16 for scaling (dx16)
        const float d  = amax * (1.0f / 127.0f);
        const float id = 1.0f / d;

        half dh_tmp;
        vstore_half(d, 0, &dh_tmp);
        const float dx16 = vload_half(0, &dh_tmp);

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            const char qw = (char)blk[2 + i];

            int qi = round_away_from_zero(x_col[k0 + i] * id);
            qi = max(-127, min(127, qi));

            sumi += ((int)qw) * qi;
        }

        sum += (float)sumi * (dw16 * dx16);
    }

    *out_ptr = sum;
}
