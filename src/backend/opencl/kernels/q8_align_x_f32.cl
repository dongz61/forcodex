#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline int round_ties_away_from_zero_i32(float x) {
    return (x >= 0.0f) ? (int)floor(x + 0.5f) : (int)ceil(x - 0.5f);
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

    int q = round_ties_away_from_zero_i32(v * id);
    q = max(-127, min(127, q));
    dst[(size_t)row * (size_t)K + (size_t)k] = d * (float)q;
}
