<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 03 - Tensor Cores via WMMA

> **Phase 5 · Expert** | Difficulty: ⭐⭐⭐⭐⭐ | Time: ~3 h | Prerequisites: [02](../02-cublas-gemm/) | **Requires sm_70+** | ✅ verified on an RTX 3060

## Goal

Use the hardware unit that makes modern GPUs fast at deep learning — and measure
what it costs in precision, rather than pretending it is free.

## Background

A Tensor Core computes a small matrix multiply-accumulate in **one instruction**:

```
D = A × B + C     A, B: 16×16 half     C, D: 16×16 float
```

That is 16·16·16·2 = **8192 FLOPs per instruction**, which is why a card with
Tensor Cores quotes an FP16 number many times its FP32 one.

The **WMMA API** (`nvcuda::wmma`) exposes them at **warp** level:

```cuda
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16,16,16, float>             c;
wmma::fill_fragment(c, 0.0f);
wmma::load_matrix_sync(a, ptrA, lda);
wmma::mma_sync(c, a, b, c);
wmma::store_matrix_sync(ptrC, c, ldc, wmma::mem_row_major);
```

**Three rules the API imposes:**

1. The **whole warp** cooperates on one tile. Every thread must call every `wmma`
   function, uniformly — a divergent call is undefined behaviour.
2. **Never index into a fragment.** Its register layout is deliberately
   unspecified; you only load, multiply and store.
3. Pointers need **256-bit alignment**, and the leading dimension must be a
   multiple of 16 for `half`.

**Precision is a real trade.** Inputs are FP16 (~3 decimal digits); the accumulator
is FP32. For a GEMM that is usually fine because accumulation dominates the error —
but this exercise *measures* it rather than waving at it. BF16 (sm_80+) trades
mantissa bits for FP32's exponent range, which makes it far safer for training;
TF32 is automatic on Ampere for FP32 GEMMs unless you opt out.

## Your task

Write `gemmFp32` (the baseline) and `gemmWmma`. Launch WMMA with `blockDim(32, 4)`
— 32 threads in x is exactly one warp.

Verify with a ~2% relative tolerance. **If you find yourself widening it further,
stop and think about whether FP16 suits your data**, rather than loosening the test
until it passes.

## Build and run

```bash
cmake --build build --target p5_03_tensor_core_wmma -j
./build/bin/p5/p5_03_tensor_core_wmma
```

## Expected output

RTX 3060 (sm_86), n = 1024:

```
FP32, one thread per element          2.656 ms      808.45 GFLOP/s     1.00x
FP16 Tensor Core (WMMA)               0.378 ms     5683.34 GFLOP/s     7.03x
```

**7×** — and note that this is a *teaching* kernel. It reads A and B straight from
global memory with no shared-memory staging, so most of the Tensor Cores'
throughput goes on memory stalls. Stage the tiles as in
[Phase 3/04](../../../phase-3-intermediate/exercises/04-tiled-matmul/) and it goes
considerably further. That is what CUTLASS does, in about twenty layers.

## Key takeaways

- **One instruction, 8192 FLOPs.** That is the whole story of why FP16 throughput
  numbers look the way they do.
- **WMMA is warp-level and uniform.** Fragments are opaque by design.
- **The speedup is not free** — you changed the input precision. Measure the error
  on *your* data, not on random uniform values.
- **Compute units do not help if memory cannot feed them.** A naive WMMA kernel
  leaves most of the hardware idle.

## Going further

- Add shared-memory staging around the WMMA loop and re-measure.
- Try `__nv_bfloat16` instead of `half` and compare the error.
- Compare against `cublasGemmEx` with `CUBLAS_COMPUTE_16F`.
- Read the [CUTLASS](https://github.com/NVIDIA/cutlass) GEMM structure and count
  the levels of tiling.
