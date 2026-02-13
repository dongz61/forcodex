#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Match forcodex/libs ggml quantize_row_q8_0 fast paths:
// use round-to-nearest-even (rte), not ties-away-from-zero.
inline int round_ties_to_even_i32(float x) {
    return convert_int_rte(x);
}

inline float fp16_roundtrip_f32(float x) {
    half h = convert_half_rte(x);
    return convert_float(h);
}

// In-place-compatible q8 quantize-dequantize alignment for activations:
// each row is split into 32-wide blocks and transformed with ggml-like q8_0
// semantics (including fp16 scale roundtrip and ties-away rounding).
kernel void kernel_q8_align_x_f32(
    global const float * src,
    ulong                off_src,
    global float       * dst,
    ulong                off_dst,
    int                  K,
    int                  M
) {
    src = (global const float *)((global const char *)src + off_src);
    dst = (global float *)((global char *)dst + off_dst);

    const int lane = get_local_id(0);
    const int row  = get_global_id(1);
    const int blk  = get_group_id(0);
    const int k    = blk * 32 + lane;

    if (row >= M || k >= K) {
        return;
    }

    local float l_abs[32];
    float v = src[(size_t)row * (size_t)K + (size_t)k];
    l_abs[lane] = fabs(v);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) {
            l_abs[lane] = fmax(l_abs[lane], l_abs[lane + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float amax = l_abs[0];
    float d = 0.0f;
    float id = 0.0f;
    if (amax > 0.0f) {
        id = 127.0f / amax;
        d = fp16_roundtrip_f32(amax / 127.0f);
    }

    const int q = round_ties_to_even_i32(v * id);
    dst[(size_t)row * (size_t)K + (size_t)k] = d * (float)((char)q);
}

// Quantize activations x to q8 blocks (32 elems each) with ggml-like semantics:
// - d = fp16_roundtrip(amax / 127)
// - q = round_ties_to_even(x * (127/amax)), clamped to [-127, 127]
// Output layout:
// - dst_q: [row_lin, K] contiguous int8
// - dst_d: [row_lin, K/32] contiguous half scales
kernel void kernel_q8_quantize_x_f32(
    global const float * src,
    ulong                off_src,
    global char        * dst_q,
    ulong                off_q,
    global half        * dst_d,
    ulong                off_d,
    int                  K,
    int                  M,
    int                  ne12,
    int                  ne13,
    ulong                nb11,
    ulong                nb12,
    ulong                nb13
) {
    src   = (global const float *)((global const char *)src + off_src);
    dst_q = (global char *)((global char *)dst_q + off_q);
    dst_d = (global half *)((global char *)dst_d + off_d);

    const int lane = get_local_id(0);
    const int blk  = get_group_id(0);
    const int m    = get_global_id(1);
    const int im   = get_global_id(2);

    const int k = blk * 32 + lane;
    if (m >= M || im >= ne12 * ne13 || k >= K) {
        return;
    }

    const int i12 = im % ne12;
    const int i13 = im / ne12;
    const size_t row_src_off = (size_t)m * (size_t)(nb11 / sizeof(float)) +
                               (size_t)i12 * (size_t)(nb12 / sizeof(float)) +
                               (size_t)i13 * (size_t)(nb13 / sizeof(float));
    const int row_lin = im * M + m;
    const int nb = K / 32;

    local float l_abs[32];
    const float v = src[row_src_off + (size_t)k];
    l_abs[lane] = fabs(v);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) {
            l_abs[lane] = fmax(l_abs[lane], l_abs[lane + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float amax = l_abs[0];
    float id = 0.0f;
    float d = 0.0f;
    if (amax > 0.0f) {
        id = 127.0f / amax;
        d = fp16_roundtrip_f32(amax / 127.0f);
    }

    const int q = round_ties_to_even_i32(v * id);
    dst_q[(size_t)row_lin * (size_t)K + (size_t)k] = (char)q;

    if (lane == 0) {
        dst_d[(size_t)row_lin * (size_t)nb + (size_t)blk] = convert_half_rte(d);
    }
}
