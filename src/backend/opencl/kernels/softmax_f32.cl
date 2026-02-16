#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define SOFTMAX_HAS_FP64 1
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define SOFTMAX_HAS_FP64 1
#endif

// Backup of the original strict scalar implementation.
// Kept for parity/debug fallback.
kernel void kernel_soft_max_strict_backup(
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
    local float * lbuf   
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;
    (void)lbuf;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int lid = get_local_id(0);

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

    // Original strict row-order scalar reduction.
    if (lid != 0) {
        return;
    }

    float maxv = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i = 0; i < ne00; ++i) {
        float v = psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f);
        maxv = fmax(maxv, v);
    }

#ifdef SOFTMAX_HAS_FP64
    double sum = 0.0;
    for (int i = 0; i < ne00; ++i) {
        float e = exp(psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f) - maxv);
        pdst[i] = e;
        sum += (double)e;
    }
    if (psrc2) {
        sum += (double)exp(psrc2[i02] - maxv);
    }
    float inv = (float)(1.0 / sum);
#else
    // No FP64 path: use compensated summation to reduce accumulation drift.
    float sum = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < ne00; ++i) {
        float e = exp(psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f) - maxv);
        pdst[i] = e;
        float y = e - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    if (psrc2) {
        float e2 = exp(psrc2[i02] - maxv);
        float y = e2 - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    float inv = 1.0f / sum;
#endif
    for (int i = 0; i < ne00; ++i) {
        pdst[i] *= inv;
    }
}

// Parallel work-group reduction implementation.
// Uses lbuf for both max and sum reductions.
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
    local float * lbuf
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

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

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        int h = i02;
        float base = h < n_head_log2 ? m0 : m1;
        int expn = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
        slope = pow(base, expn);
    }

    float tmax = -INFINITY;
    for (int i = lid; i < ne00; i += lsize) {
        float v = psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f);
        tmax = fmax(tmax, v);
    }
    if (psrc2 && lid == 0) {
        tmax = fmax(tmax, psrc2[i02]);
    }

    lbuf[lid] = tmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            lbuf[lid] = fmax(lbuf[lid], lbuf[lid + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maxv = lbuf[0];

    float tsum = 0.0f;
    for (int i = lid; i < ne00; i += lsize) {
        float e = exp(psrc0[i] * scale + (pmask ? slope * pmask[i] : 0.0f) - maxv);
        pdst[i] = e;
        tsum += e;
    }
    if (psrc2 && lid == 0) {
        tsum += exp(psrc2[i02] - maxv);
    }

    lbuf[lid] = tsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            lbuf[lid] += lbuf[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.0f / lbuf[0];

    for (int i = lid; i < ne00; i += lsize) {
        pdst[i] *= inv;
    }
}
