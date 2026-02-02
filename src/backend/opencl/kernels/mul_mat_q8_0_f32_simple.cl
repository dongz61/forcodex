#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK8_0 32
#define BS8_0 34   // sizeof(block_q8_0) = 2 + 32

// ---- helpers: match ggml quantize_row_q8_0_ref semantics ----

// ggml stores d as fp16; to match its scale semantics we do fp16 round-trip.
static inline float fp16_roundtrip(float x) {
    half h;
    vstore_half(x, 0, &h);
    return vload_half(0, &h);
}

// ggml uses roundf(x) which is ties-away-from-zero.
// Implement it explicitly (OpenCL round() is ties-to-even on many drivers).
static inline int round_ties_away_from_zero(float x) {
    return (x >= 0.0f) ? (int)floor(x + 0.5f) : (int)ceil(x - 0.5f);
}

static inline char quantize_to_q8(float x, float id) {
    // id = 127 / amax
    int q = (int)rint(x * id);
    q = clamp(q, -127, 127);
    return (char)q;
}

// Read ggml block layout directly:
// block_q8_0: [half d][int8 qs[32]]
kernel void kernel_mul_mat_q8_0_f32_simple(
    global const uchar * w,     // ggml raw buffer (Q8_0)
    ulong off_w,                // byte offset
    global const float * x,     // FP32 activations
    ulong off_x,
    global float * dst,         // FP32 out
    ulong off_dst,
    int K, int N, int M,
    ulong nb_w1,                // weight stride bytes for dim1 (row stride: one output row)
    ulong nb_x1,                // x stride bytes for dim1 (row stride: one batch column)
    ulong nb_dst1               // dst stride bytes for dim1 (row stride: one batch column)
) {
    const int n = (int)get_global_id(0); // [0..N)
    const int m = (int)get_global_id(1); // [0..M)
    if (n >= N || m >= M) return;

    // base pointers with byte offsets
    global const uchar * w_row = (global const uchar *)((global const char *)w + off_w + (ulong)n * nb_w1);
    global const float * x_row = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK8_0;

    float sumf = 0.0f;

    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_row + (ulong)b * (ulong)BS8_0;

        // dw is stored in fp16 in weight block
        const float dw = vload_half(0, (global const half *)blk);

        // ---- dynamic quantize x block to q8_0 (ggml semantics) ----
        const int k0 = b * QK8_0;

        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float xv = x_row[k0 + i];
            amax = fmax(amax, fabs(xv));
        }

        float id = 0.0f;
        float dx = 0.0f;
        if (amax > 0.0f) {
            // ggml: d = amax/127 (fp32), id = 1/d = 127/amax (fp32)
            id = 127.0f / amax;

            // ggml: store d into fp16 and later convert back to fp32 for scaling
            dx = fp16_roundtrip(amax / 127.0f);
        }

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            const char qw = (char)blk[2 + i];           // weight q8
            const char qx = quantize_to_q8(x_row[k0 + i], id); // act q8
            sumi += (int)qx * (int)qw;
        }

        // ggml_vec_dot_q8_0_q8_0: sumf += (dx * dw) * sumi
        sumf += ((float)sumi) * dx * dw;
    }

    *out_ptr = sumf;
}
