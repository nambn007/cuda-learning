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
| 01 ✅ | [thrust-and-cub](exercises/01-thrust-and-cub/) | ⭐⭐ | Calibration: your Phase 3 reduction matches CUB exactly |
| 02 ⚠️ | [cublas-gemm](exercises/02-cublas-gemm/) | ⭐⭐⭐ | The column-major trap, and how far your GEMM really is |
| 03 ✅ | [tensor-core-wmma](exercises/03-tensor-core-wmma/) | ⭐⭐⭐⭐⭐ | 8192 FLOPs in one instruction — and what it costs in precision |

✅ verified on an RTX 3060 · ⚠️ code complete, not yet run on the reference machine

### Deferred

Specified in [docs/ROADMAP.md](../docs/ROADMAP.md) but not written yet:
**cublas-batched**, **curand-monte-carlo**, **cufft-convolution**, **cusparse-spmv**,
**mixed-precision**, **cutlass-gemm**, **pytorch-extension**, **nvrtc-jit**,
**cuda-opengl-interop**, **kernel-fusion**.

Exercise 03 requires compute capability ≥ 7.0 and skips cleanly on older cards.

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
