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

#ifdef OPENCL_GELU_CL_AVAILABLE
#include "gelu.cl.h"
#endif

#ifdef OPENCL_RELU_CL_AVAILABLE
#include "relu.cl.h"
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

#endif // POWERSERVE_OPENCL_EMBED_KERNELS