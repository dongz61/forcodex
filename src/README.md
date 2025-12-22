# forcodex

This repository contains an extracted `src/` directory from the PowerServe project,
intended for focused code review and design discussion.

## Scope
- OpenCL backend implementation
- OpenCLBuffer ownership & lifetime
- KVCache (time-dimension state) design
- Attention correctness for multi-token inference

## Notes
- This is NOT a standalone buildable project
- Some dependencies are assumed to exist in the original repository
- The code is reviewed in isolation for correctness and design

## Focus Areas
- src/runtime/opencl
- src/include (PowerServe-style headers colocated in src)

