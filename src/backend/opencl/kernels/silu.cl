#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// silu
//------------------------------------------------------------------------------
kernel void kernel_silu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

inline float sigmoid_f32(float x) {
    // numerically stable sigmoid
    if (x >= 0.0f) {
        float z = exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = exp(x);
        return z / (1.0f + z);
    }
}

__kernel void kernel_silu_hadamard_contig_f32(
    __global const float *hb,
    __global const float *hb2,
    __global float *out,
    int n
) {
    int gid = (int)get_global_id(0);
    if (gid < n) {
        float x = hb[gid];
        float s = sigmoid_f32(x);
        out[gid] = (x * s) * hb2[gid];
    }
}

