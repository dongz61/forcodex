#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0 32
#define QK8_0 32
#define BS4_0 18   // sizeof(block_q4_0) = 2 + 16
#define BS8_0 34   // sizeof(block_q8_0) = 2 + 32

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}

kernel void kernel_get_rows_f32(
    global const float * w,
    ulong off_w,
    ulong nb_w0,
    ulong nb_w1,
    global const int * tokens,
    ulong off_t,
    global float * dst,
    ulong off_dst,
    ulong nb_dst0,
    ulong nb_dst1,
    int dim,
    int batch
) {
    const int d = (int)get_global_id(0);
    const int b = (int)get_global_id(1);
    if (d >= dim || b >= batch) return;

    global const int * t_ptr = (global const int *)((global const char *)tokens + off_t);
    const int tok = t_ptr[b];

    global const char * w_row = (global const char *)w + off_w + (ulong)tok * nb_w1;
    global const float * w_elem = (global const float *)(w_row + (ulong)d * nb_w0);

    global char * dst_row = (global char *)dst + off_dst + (ulong)b * nb_dst1;
    global float * out_elem = (global float *)(dst_row + (ulong)d * nb_dst0);
    *out_elem = *w_elem;
}

kernel void kernel_get_rows_q4_0(
    global const uchar * w,
    ulong off_w,
    ulong nb_w1,
    global const int * tokens,
    ulong off_t,
    global float * dst,
    ulong off_dst,
    ulong nb_dst1,
    int dim,
    int batch
) {
    const int ind = (int)get_global_id(0);
    const int b = (int)get_global_id(1);
    const int chunks = (dim + 15) / 16;
    if (ind >= chunks || b >= batch) return;

    global const int * t_ptr = (global const int *)((global const char *)tokens + off_t);
    const int tok = t_ptr[b];

    global const uchar * w_row = (global const uchar *)((global const char *)w + off_w + (ulong)tok * nb_w1);

    const int block_idx = ind / 2;
    const int il = ind % 2;
    global struct block_q4_0 * block = (global struct block_q4_0 *)(w_row + (ulong)block_idx * (ulong)BS4_0);

    float16 temp;
    dequantize_q4_0_f32(block, (short)il, &temp);

    const int base = ind * 16;
    global char * dst_row_bytes = (global char *)dst + off_dst + (ulong)b * nb_dst1;
    if ((base + 15) < dim) {
        global float16 * out_row = (global float16 *)dst_row_bytes;
        out_row[ind] = temp;
    } else {
        global float * out_row = (global float *)dst_row_bytes;
        if (base + 0 < dim) out_row[base + 0] = temp.s0;
        if (base + 1 < dim) out_row[base + 1] = temp.s1;
        if (base + 2 < dim) out_row[base + 2] = temp.s2;
        if (base + 3 < dim) out_row[base + 3] = temp.s3;
        if (base + 4 < dim) out_row[base + 4] = temp.s4;
        if (base + 5 < dim) out_row[base + 5] = temp.s5;
        if (base + 6 < dim) out_row[base + 6] = temp.s6;
        if (base + 7 < dim) out_row[base + 7] = temp.s7;
        if (base + 8 < dim) out_row[base + 8] = temp.s8;
        if (base + 9 < dim) out_row[base + 9] = temp.s9;
        if (base + 10 < dim) out_row[base + 10] = temp.sa;
        if (base + 11 < dim) out_row[base + 11] = temp.sb;
        if (base + 12 < dim) out_row[base + 12] = temp.sc;
        if (base + 13 < dim) out_row[base + 13] = temp.sd;
        if (base + 14 < dim) out_row[base + 14] = temp.se;
        if (base + 15 < dim) out_row[base + 15] = temp.sf;
    }
}

kernel void kernel_get_rows_q8_0(
    global const uchar * w,
    ulong off_w,
    ulong nb_w1,
    global const int * tokens,
    ulong off_t,
    global float * dst,
    ulong off_dst,
    ulong nb_dst1,
    int dim,
    int batch
) {
    const int d = (int)get_global_id(0);
    const int b = (int)get_global_id(1);
    if (d >= dim || b >= batch) return;

    global const int * t_ptr = (global const int *)((global const char *)tokens + off_t);
    const int tok = t_ptr[b];

    global const uchar * w_row = (global const uchar *)((global const char *)w + off_w + (ulong)tok * nb_w1);

    const int block_idx = d / QK8_0;
    const int in_block = d % QK8_0;
    global const uchar * block = w_row + (ulong)block_idx * (ulong)BS8_0;

    const float scale = vload_half(0, (global const half *)block);
    const char q = (char)block[2 + in_block];

    global float * out_row = (global float *)((global char *)dst + off_dst + (ulong)b * nb_dst1);
    out_row[d] = ((float)q) * scale;
}