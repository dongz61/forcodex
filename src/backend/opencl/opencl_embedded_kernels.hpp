// src/backend/opencl/opencl_embedded_kernels.hpp
#pragma once

#ifdef POWERSERVE_OPENCL_EMBED_KERNELS

// 包含所有生成的嵌入式内核头文件

//modified 
#ifdef OPENCL_ADD_CL_AVAILABLE
#include "add.cl.h"
#endif

#ifdef OPENCL_MUL_CL_AVAILABLE
#include "mul.cl.h"
#endif

#ifdef OPENCL_SCALE_CL_AVAILABLE
#include "scale.cl.h"
#endif

#ifdef OPENCL_NORM_CL_AVAILABLE
#include "norm.cl.h"
#endif

//modified
#ifdef OPENCL_RMS_NORM_CL_AVAILABLE
#include "rms_norm.cl.h"
#endif

#ifdef OPENCL_SILU_CL_AVAILABLE
#include "silu.cl.h"
#endif

//modified
#ifdef OPENCL_SOFTMAX_CL_AVAILABLE
#include "softmax_f32.cl.h"
#endif

#ifdef OPENCL_DIAG_MASK_INF_CL_AVAILABLE
#include "diag_mask_inf.cl.h"
#endif

#ifdef OPENCL_ROPE_CL_AVAILABLE
#include "rope.cl.h"
#endif

#ifdef OPENCL_CPY_CL_AVAILABLE
#include "cpy.cl.h"
#endif

//modified
#ifdef OPENCL_MATMUL_CL_AVAILABLE
#include "mul_mat_f16_f32.cl.h"
#endif

#ifdef OPENCL_MUL_MV_Q4_0_F32_8X_FLAT_CL_AVAILABLE
#include "mul_mv_q4_0_f32_8x_flat.cl.h"
#endif

#ifdef OPENCL_MUL_MV_Q8_0_F32_FLAT_CL_AVAILABLE
#include "mul_mv_q8_0_f32_flat.cl.h"
#endif

#ifdef OPENCL_MUL_MM_Q8_0_F32_L4_LM_CL_AVAILABLE
#include "mul_mm_q8_0_f32_l4_lm.cl.h"
#endif

#ifdef OPENCL_MUL_MM_F16_F32_L4_LM_CL_AVAILABLE
#include "mul_mm_f16_f32_l4_lm.cl.h"
#endif

#ifdef OPENCL_MUL_MM_F32_F32_L4_LM_CL_AVAILABLE
#include "mul_mm_f32_f32_l4_lm.cl.h"
#endif

#ifdef OPENCL_MUL_MV_F16_F32_1ROW_CL_AVAILABLE
#include "mul_mv_f16_f32_1row.cl.h"
#endif

#ifdef OPENCL_MUL_MV_F16_F32_L4_CL_AVAILABLE
#include "mul_mv_f16_f32_l4.cl.h"
#endif

#ifdef OPENCL_MUL_MV_F32_F32_CL_AVAILABLE
#include "mul_mv_f32_f32.cl.h"
#endif

#ifdef OPENCL_Q8_ALIGN_X_F32_CL_AVAILABLE
#include "q8_align_x_f32.cl.h"
#endif

#ifdef OPENCL_GET_ROWS_CL_AVAILABLE
#include "get_rows.cl.h"
#endif

#ifdef OPENCL_GET_MASK_CL_AVAILABLE
#include "get_mask.cl.h"
#endif

#endif // POWERSERVE_OPENCL_EMBED_KERNELS
