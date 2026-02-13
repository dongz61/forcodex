#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK8_0 32
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

#define NB_Q8_0 8

#ifdef INTEL_GPU
#define N_R0_Q8_0 4 // number of rows each subgroup works on
#define N_SG_Q8_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q8_0 4
#define N_SG_Q8_0 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif

kernel void kernel_mul_mv_q8_0_f32_flat(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows
    uint offset_src0_base = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global char * ax0, * ax1, * ax2, * ax3;
    global half * ad0, * ad1, * ad2, * ad3;
    uint offset_src0;

    offset_src0 = offset_src0_base + 0*nb01;
    offset_src0 = offset_src0/34;
    ax0 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad0 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 1*nb01;
    offset_src0 = offset_src0/34;
    ax1 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad1 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 2*nb01;
    offset_src0 = offset_src0/34;
    ax2 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad2 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 3*nb01;
    offset_src0 = offset_src0/34;
    ax3 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad3 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    global float * yb = y + ix*QK8_0 + il*NB_Q8_0;

    float8 yl;
    float8 qv;
    float4 sumf = 0.f;
    float  sumq = 0.f;
    global char * qs;

    // each thread handles NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        yl = vload8(0, yb);

        qs = ax0 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s0 += sumq*ad0[ib];

        qs = ax1 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s1 += sumq*ad1[ib];

        qs = ax2 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s2 += sumq*ad2[ib];

        qs = ax3 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s3 += sumq*ad3[ib];

        yb += N_SIMDWIDTH*NB_Q8_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst_f32[first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst_f32[first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst_f32[first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst_f32[first_row + 3] = tot.s3;
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_f32_intx_flat(
    global char * src0_q,
    global half * src0_d,
    global char * x_q,
    ulong         off_x_q,
    global half * x_d,
    ulong         off_x_d,
    global char * dst,
    ulong         offsetd,
    int           ne00, // K
    int           ne01, // N
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne11, // M
    int           ne12,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    x_q = (global char *)((global char *)x_q + off_x_q);
    x_d = (global half *)((global char *)x_d + off_x_d);
    dst = (global char *)((global char *)dst + offsetd);

    const int nb = ne00 / QK8_0;

    const int r0 = get_group_id(0);
    const int r1 = get_group_id(1); // m
    const int im = get_group_id(2); // i12 + i13 * ne12

    const int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;
    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const int lane = get_sub_group_local_id();
    const int ix = lane / 4;
    const int il = lane % 4;

    const int row_lin = im * ne11 + r1;
    global half * x_db = x_d + (size_t)row_lin * (size_t)nb;

    const ulong offset_src0_base = (ulong)first_row * nb01 +
                                   (ulong)(i12 / r2) * nb02 +
                                   (ulong)(i13 / r3) * nb03;

    ulong offset_src0 = offset_src0_base + 0 * nb01;
    offset_src0 = offset_src0 / 34;
    global char * ax0 = (global char *)((global char *)src0_q + offset_src0 * (ulong)QK8_0);
    global half * ad0 = (global half *)((global char *)src0_d + offset_src0 * sizeof(half));

    offset_src0 = offset_src0_base + 1 * nb01;
    offset_src0 = offset_src0 / 34;
    global char * ax1 = (global char *)((global char *)src0_q + offset_src0 * (ulong)QK8_0);
    global half * ad1 = (global half *)((global char *)src0_d + offset_src0 * sizeof(half));

    offset_src0 = offset_src0_base + 2 * nb01;
    offset_src0 = offset_src0 / 34;
    global char * ax2 = (global char *)((global char *)src0_q + offset_src0 * (ulong)QK8_0);
    global half * ad2 = (global half *)((global char *)src0_d + offset_src0 * sizeof(half));

    offset_src0 = offset_src0_base + 3 * nb01;
    offset_src0 = offset_src0 / 34;
    global char * ax3 = (global char *)((global char *)src0_q + offset_src0 * (ulong)QK8_0);
    global half * ad3 = (global half *)((global char *)src0_d + offset_src0 * sizeof(half));

    float4 sumf = 0.f;

    const int lanes_k = N_SIMDWIDTH / 4;
    // Keep the same block traversal pattern as kernel_mul_mv_q8_0_f32_flat
    // to reduce accumulation-order drift versus existing fast path.
    for (int ib = ix; ib < nb; ib += lanes_k) {
        global char * x_qb = x_q + (size_t)row_lin * (size_t)ne00 + (size_t)ib * (size_t)QK8_0 + (size_t)il * NB_Q8_0;
        char8 qx = vload8(0, x_qb);

        // row 0
        {
            global char * qs = ax0 + (size_t)ib * QK8_0 + (size_t)il * NB_Q8_0;
            char8 qw = vload8(0, qs);
            int s = 0;
            s += (int)qw.s0 * (int)qx.s0;
            s += (int)qw.s1 * (int)qx.s1;
            s += (int)qw.s2 * (int)qx.s2;
            s += (int)qw.s3 * (int)qx.s3;
            s += (int)qw.s4 * (int)qx.s4;
            s += (int)qw.s5 * (int)qx.s5;
            s += (int)qw.s6 * (int)qx.s6;
            s += (int)qw.s7 * (int)qx.s7;
            s += sub_group_shuffle_xor(s, 1);
            s += sub_group_shuffle_xor(s, 2);
            if (il == 0) {
                sumf.s0 += ((float)s) * (float)x_db[ib] * (float)ad0[ib];
            }
        }

        // row 1
        {
            global char * qs = ax1 + (size_t)ib * QK8_0 + (size_t)il * NB_Q8_0;
            char8 qw = vload8(0, qs);
            int s = 0;
            s += (int)qw.s0 * (int)qx.s0;
            s += (int)qw.s1 * (int)qx.s1;
            s += (int)qw.s2 * (int)qx.s2;
            s += (int)qw.s3 * (int)qx.s3;
            s += (int)qw.s4 * (int)qx.s4;
            s += (int)qw.s5 * (int)qx.s5;
            s += (int)qw.s6 * (int)qx.s6;
            s += (int)qw.s7 * (int)qx.s7;
            s += sub_group_shuffle_xor(s, 1);
            s += sub_group_shuffle_xor(s, 2);
            if (il == 0) {
                sumf.s1 += ((float)s) * (float)x_db[ib] * (float)ad1[ib];
            }
        }

        // row 2
        {
            global char * qs = ax2 + (size_t)ib * QK8_0 + (size_t)il * NB_Q8_0;
            char8 qw = vload8(0, qs);
            int s = 0;
            s += (int)qw.s0 * (int)qx.s0;
            s += (int)qw.s1 * (int)qx.s1;
            s += (int)qw.s2 * (int)qx.s2;
            s += (int)qw.s3 * (int)qx.s3;
            s += (int)qw.s4 * (int)qx.s4;
            s += (int)qw.s5 * (int)qx.s5;
            s += (int)qw.s6 * (int)qx.s6;
            s += (int)qw.s7 * (int)qx.s7;
            s += sub_group_shuffle_xor(s, 1);
            s += sub_group_shuffle_xor(s, 2);
            if (il == 0) {
                sumf.s2 += ((float)s) * (float)x_db[ib] * (float)ad2[ib];
            }
        }

        // row 3
        {
            global char * qs = ax3 + (size_t)ib * QK8_0 + (size_t)il * NB_Q8_0;
            char8 qw = vload8(0, qs);
            int s = 0;
            s += (int)qw.s0 * (int)qx.s0;
            s += (int)qw.s1 * (int)qx.s1;
            s += (int)qw.s2 * (int)qx.s2;
            s += (int)qw.s3 * (int)qx.s3;
            s += (int)qw.s4 * (int)qx.s4;
            s += (int)qw.s5 * (int)qx.s5;
            s += (int)qw.s6 * (int)qx.s6;
            s += (int)qw.s7 * (int)qx.s7;
            s += sub_group_shuffle_xor(s, 1);
            s += sub_group_shuffle_xor(s, 2);
            if (il == 0) {
                sumf.s3 += ((float)s) * (float)x_db[ib] * (float)ad3[ib];
            }
        }
    }

    global float * dst_f32 = (global float *) dst + (size_t)im * (size_t)ne0 * (size_t)ne1 + (size_t)r1 * (size_t)ne0;

    // Deterministic subgroup reduction over il==0 lanes only (ix lanes).
    // This avoids vendor-specific reduce trees in sub_group_reduce_add.
    float v0 = (il == 0) ? sumf.s0 : 0.0f;
    float v1 = (il == 0) ? sumf.s1 : 0.0f;
    float v2 = (il == 0) ? sumf.s2 : 0.0f;
    float v3 = (il == 0) ? sumf.s3 : 0.0f;

    for (int step = 4; step < N_SIMDWIDTH; step <<= 1) {
        v0 += sub_group_shuffle_xor(v0, step);
        v1 += sub_group_shuffle_xor(v1, step);
        v2 += sub_group_shuffle_xor(v2, step);
        v3 += sub_group_shuffle_xor(v3, step);
    }

    if (lane == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = v0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = v1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = v2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = v3;
    }
}

