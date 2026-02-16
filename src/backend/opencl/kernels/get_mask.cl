#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_get_mask_f32(
        global float *dst,
        ulong offsetd,
        global int *pos,
        ulong offsetp,
        int n_kv,
        int batch_size
) {
    dst = (global float *)((global char *)dst + offsetd);
    pos = (global int *)((global char *)pos + offsetp);

    const int j = get_global_id(0);
    const int b = get_global_id(1);

    if (j >= n_kv || b >= batch_size) {
        return;
    }

    const int p = pos[b];
    dst[b * n_kv + j] = (j <= p) ? 0.0f : -INFINITY;
}
