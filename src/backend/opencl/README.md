# OpenCL Backend

This directory contains the OpenCL backend for `powerserve`.

The goal of this backend is simple: build with OpenCL enabled, set one runtime environment variable, and run the model with OpenCL kernels instead of the default backend where supported.

## Directory Layout

- `opencl_backend.*`: backend entry and lifecycle
- `opencl_context.*`: device discovery, context, command queue
- `opencl_memory.*`: buffer allocation and memory pool
- `opencl_kernel_manager.*`: kernel compilation and lookup
- `opencl_kernels_ops.cpp`: main operator implementations
- `opencl_tensor_ops.cpp`: tensor copy/view/transpose/permute helpers
- `opencl_kv_cache.*`, `opencl_kv_cache_ops.cpp`: KV cache storage and updates
- `kernels/*.cl`: OpenCL kernel sources

## Build

Enable the backend at configure time:

```bash
cmake -B build -S . -DPOWERSERVE_WITH_OPENCL=ON
cmake --build build
```

Useful build options:

- `POWERSERVE_WITH_OPENCL=ON`: enable OpenCL backend
- `POWERSERVE_OPENCL_EMBED_KERNELS=ON`: embed `.cl` kernels into the binary instead of loading them from files at runtime

Notes:

- On non-Android platforms, CMake will try `find_package(OpenCL)`.
- On Android, you may need to provide `POWERSERVE_ANDROID_OPENCL_LIB=/abs/path/to/libOpenCL.so`.

## Runtime Usage

OpenCL is selected at runtime with:

```bash
POWERSERVE_USE_OPENCL=1
```

Example:

```bash
POWERSERVE_USE_OPENCL=1 ./build/path/to/powerserve-run
```

If `POWERSERVE_USE_OPENCL` is not set, the normal backend selection path is used.

## Device Selection

The backend can select devices through these environment variables:

- `GGML_OPENCL_PLATFORM`
- `GGML_OPENCL_DEVICE`

They accept either an index or a substring match, depending on the value provided by the user.

If no device is specified, the backend tries to pick an available GPU first, then falls back to another available OpenCL device.

Example:

```bash
GGML_OPENCL_PLATFORM=0 GGML_OPENCL_DEVICE=0 POWERSERVE_USE_OPENCL=1 ./build/path/to/powerserve-run
```

## Kernel Loading

There are two kernel loading modes:

- Runtime file loading from `src/backend/opencl/kernels/*.cl`
- Embedded kernels generated at build time when `POWERSERVE_OPENCL_EMBED_KERNELS=ON`

If you are packaging binaries or running in an environment where shipping `.cl` files is inconvenient, embedded kernels are usually the simpler option.

## Current Support

The backend already covers the main path used by the model, including:

- tensor copy / contiguous conversion / transpose / permute
- add
- matmul for `FP16`, `FP32`, `Q4_0`, `Q8_0` weights with `FP32` activations/output
- get rows / embedding lookup
- RMSNorm
- RoPE
- mask generation
- `softmax_ext`
- KV cache updates

## Current Limitations

This backend is still under active development. A few practical limitations matter when using it:

- Not every backend API is fully implemented
- Some paths still fall back to GGML/CPU behavior
- Several operators are currently strict about dtype and layout
- `softmax()` is not implemented; the current attention path uses `softmax_ext()`
- The current KV cache storage is `FP32`
- The OpenCL KV cache supports batched updates, but its storage layout is still per-layer `{kv_dim, max_seq_len}` without a separate batch dimension

## Debugging

When OpenCL initialization fails, the first things to check are:

1. OpenCL runtime/library is installed and linkable
2. The target device is visible to the system
3. `GGML_OPENCL_PLATFORM` / `GGML_OPENCL_DEVICE` are pointing to the expected device
4. Kernel sources are available if embedded kernels are disabled

For backend quality comparison, see:

- `forcodex/tools/README_backend_eval.md`

## Quick Summary

For normal usage, the minimum setup is:

```bash
cmake -B build -S . -DPOWERSERVE_WITH_OPENCL=ON
cmake --build build
POWERSERVE_USE_OPENCL=1 ./build/path/to/powerserve-run
```
