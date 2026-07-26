<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# Phase 5 · Expert

> **6–8 weeks · ⭐⭐⭐⭐⭐ · 14 exercises** — [Roadmap](../docs/ROADMAP.md) · [← Repository](../README.md)

## What this phase is for

Everything so far taught you to write kernels. This phase teaches you **when not
to**. A tuned cuBLAS `SGEMM` has years of architecture-specific work behind it; a
hand-written kernel that reaches 80% of it is an excellent result, and reaching
100% is a full-time job.

So the goal here is judgement: know what the ecosystem provides, know how to call
it correctly, know how to measure the gap, and know the few situations where
writing your own genuinely wins — usually **fusion**, where a custom kernel avoids
a round trip through DRAM that a library call cannot.

## Exercises

| # | Exercise | Difficulty | Core idea |
|---|---|---|---|
| 01 | thrust-basics | ⭐⭐ | STL-style algorithms on the device |
| 02 | cub-primitives | ⭐⭐⭐ | Block/device primitives vs your hand-written versions |
| 03 | cublas-gemm | ⭐⭐⭐ | The column-major trap; how far your tiled matmul really is |
| 04 | cublas-batched | ⭐⭐⭐ | Batched and strided-batched GEMM |
| 05 | curand-monte-carlo | ⭐⭐⭐ | Device-side RNG, and what "parallel random" means |
| 06 | cufft-convolution | ⭐⭐⭐ | Convolution via FFT |
| 07 | cusparse-spmv | ⭐⭐⭐⭐ | CSR SpMV — irregular, memory bound, unavoidable |
| 08 | mixed-precision | ⭐⭐⭐⭐ | FP16/BF16/TF32, `half2` vectorisation, accuracy vs speed |
| 09 | tensor-core-wmma | ⭐⭐⭐⭐⭐ | The WMMA API (needs sm_70+) |
| 10 | cutlass-gemm | ⭐⭐⭐⭐⭐ | Templated GEMM, epilogue fusion *(optional dependency)* |
| 11 | pytorch-extension | ⭐⭐⭐⭐ | A custom CUDA op callable from Python *(optional dependency)* |
| 12 | nvrtc-jit | ⭐⭐⭐⭐ | Runtime compilation and the driver API |
| 13 | cuda-opengl-interop | ⭐⭐⭐⭐ | Zero-copy visualisation *(optional dependency)* |
| 14 | kernel-fusion | ⭐⭐⭐⭐ | Fewer passes over memory — the highest-leverage optimisation left |

Exercises marked *optional dependency* build only with:

```bash
./scripts/build.sh --optional
```

Exercise 09 requires compute capability ≥ 7.0 and skips cleanly on older cards.

## Build and run

```bash
./scripts/build.sh
ctest --test-dir build -L p5 --output-on-failure
```

## Checklist

- [ ] I reach for Thrust/CUB before writing a scan or a sort by hand
- [ ] I know cuBLAS is column-major and can call it on row-major data correctly
- [ ] I know what percentage of cuBLAS my own GEMM achieves
- [ ] I can use FP16 without silently destroying accuracy, and can say why
- [ ] I have run a Tensor Core kernel and measured the speedup over FP32
- [ ] I can expose a CUDA kernel to Python
- [ ] I can identify a chain of kernels worth fusing, and measure the traffic saved

## Checkpoint

Given a new problem, you can say within minutes which library covers it, call that
library correctly, and quantify what a custom kernel would have to beat.
