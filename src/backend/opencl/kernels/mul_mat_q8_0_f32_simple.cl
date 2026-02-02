#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define QK8_0 32
#define BS8_0 34   // sizeof(block_q8_0) = 2 + 32

// Read ggml block layout directly:
// block_q8_0: [half d][int8 qs[32]]
kernel void kernel_mul_mat_q8_0_f32_simple(
    global const uchar * w,     // ggml raw buffer
    ulong off_w,                // byte offset
    global const float * x,
    ulong off_x,
    global float * dst,
    ulong off_dst,
    int K, int N, int M,
    ulong nb_w1,                // weight stride bytes for dim1 (column stride)
    ulong nb_x1,                // x stride bytes for dim1
    ulong nb_dst1               // dst stride bytes for dim1
) {
    const int n = (int)get_global_id(0); // [0..N)
    const int m = (int)get_global_id(1); // [0..M)
    if (n >= N || m >= M) return;

    // base pointers with byte offsets
    global const uchar * w_col = (global const uchar *)((global const char *)w + off_w + (ulong)n * nb_w1);
    global const float * x_col = (global const float *)((global const char *)x + off_x + (ulong)m * nb_x1);
    global float * out_ptr     = (global float *)((global char *)dst + off_dst + (ulong)n * (ulong)sizeof(float) + (ulong)m * nb_dst1);

    const int blocks = K / QK8_0;

    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        global const uchar * blk = w_col + (ulong)b * (ulong)BS8_0;

        const float d = vload_half(0, (global const half *)blk);

        float acc = 0.0f;
        const int k0 = b * QK8_0;

        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            // qs are signed int8 in ggml layout
            const char q = (char)blk[2 + i];
            acc += (float)q * x_col[k0 + i];
        }
        sum += acc * d; // ggml-like scale per block
    }

    *out_ptr = sum;
}
