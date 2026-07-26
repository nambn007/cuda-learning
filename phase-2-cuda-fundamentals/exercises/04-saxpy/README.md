<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 04 - SAXPY and vectorised loads

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~1 h | Prerequisites: [03](../03-vector-add/) | **Requires a GPU**

## Goal

Learn the one metric that decides whether a memory-bound kernel is any good —
**percentage of peak bandwidth** — and meet `float4`, the cheapest optimisation in
CUDA.

## Background

`y[i] = a * x[i] + y[i]`. Reads 8 bytes, writes 4, does one FMA. Arithmetic
intensity 2/12 = 0.167 FLOP/byte, so like everything in Phase 2 it is firmly
memory bound.

Exercise 03 established that such kernels are limited by DRAM. The follow-up
question is: **given that we are limited by bandwidth, are we using all of it?**

**Reporting the right number.** For a memory-bound kernel, milliseconds are
meaningless without a size, and GB/s is meaningless without the hardware peak.
The useful figure is the ratio:

| % of peak | Verdict |
|---|---|
| > 80% | The kernel is finished. Optimise something else. |
| 50–80% | Reasonable for a kernel doing real work. |
| < 50% | Something is wrong — almost always the access pattern (Phase 3/01). |

**Vectorised loads.** `float4` is a built-in struct of four floats with 16-byte
alignment. Loading one issues a **single 16-byte memory instruction** instead of
four 4-byte ones. The bytes moved are identical; the number of *instructions*
drops by 4×. On a kernel with nothing else to do while it waits, that keeps more
data in flight per warp.

Expect a **few percent**, not a transformation. `float4` is worth reaching for
only once the access pattern is already coalesced — it makes a good kernel
slightly better, and does nothing at all for a bad one.

**Alignment is a hard requirement.** `reinterpret_cast<float4*>` is legal only on
a 16-byte aligned pointer. `cudaMalloc` returns at least 256-byte alignment, so a
base pointer is safe — but `d_x + 1` is not, and the failure mode is a
misaligned-address error at run time.

**One more detail:** `a * x[i] + y[i]` compiles to a single FMA — one multiply-add
with *one* rounding step. That is slightly more accurate than a separate multiply
and add, and slightly different from the CPU reference. Hence `rtol = 1e-5`.

## Your task

Open `main.cu`:

1. **`saxpy()`** — the scalar grid-stride version.
2. **`saxpyVectorised()`** — the same thing over `float4`, plus a scalar tail for
   `n % 4`.
3. **Report percentage of peak** using `theoreticalBandwidthGBs()` from
   `cuda_helper.h`.

Remember to reset `y` before every timed run — SAXPY accumulates into it.

## Build and run

```bash
cmake --build build --target p2_04_saxpy -j
./build/bin/p2/p2_04_saxpy
```

## Expected output

RTX 3060, `n = 32M`:

```
--- Correctness ---
  [PASS] scalar saxpy (n=33554432, max abs err 2.384e-07)
  [PASS] vectorised saxpy (float4) (n=33554432, max abs err 2.384e-07)

--- Performance ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
CPU (single thread)                  66.243      6.08       1.01     1.00x
GPU scalar                            1.242    324.13      54.02    53.33x
GPU float4                            1.206    333.80      55.63    54.92x

--- Percentage of peak bandwidth ---
  Theoretical peak      : 360.0 GB/s
  Scalar kernel         : 324.1 GB/s  (90% of peak)
  float4 kernel         : 333.8 GB/s  (93% of peak)
```

90% of peak from the *naive* version. That is what a coalesced memory-bound kernel
looks like, and it is the bar Phase 3 will hold every kernel to.

## Key takeaways

- **Percentage of peak is the metric** for memory-bound kernels. Milliseconds
  alone tell you nothing.
- **A simple coalesced kernel already reaches ~90%.** When you see 20%, the
  problem is the access pattern, not the arithmetic.
- **`float4` buys a few percent** by cutting instruction count, not bytes. It is a
  finishing touch, not a fix.
- **Alignment is a correctness requirement**, not a performance hint.
- FMA changes the last bits of your result. Always compare floats with a
  tolerance.

## Going further

- Try `float2` as well. Is the benefit proportional to the width?
- Read the SASS: `cuobjdump -sass build/bin/p2/p2_04_saxpy_sol | grep LDG`. Count
  `LDG.E` versus `LDG.E.128` in the two kernels.
- Sweep the block size from 64 to 1024. How flat is the curve? (Phase 3/15.)
- Make `x` deliberately misaligned (`d_x + 1`) and watch it fail. Then read the
  error under `compute-sanitizer`.
