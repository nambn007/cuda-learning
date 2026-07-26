<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 06 - SIMD intrinsics: the CPU's warp

> **Phase 1 · Foundation** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [05](../05-cache-and-bandwidth/)

## Goal

Write AVX2 code by hand and discover, before you ever launch a kernel, the two
rules that govern GPU performance: **work happens in fixed-width groups**, and
**wider execution only helps when you are not memory bound**.

## Background

An AVX2 register is 256 bits — **8 floats** — and one instruction operates on all
8 lanes. A CUDA **warp** is **32 threads** issuing one instruction together.

| | SIMD (AVX2) | SIMT (CUDA warp) |
|---|---|---|
| Lanes | 8 | 32 |
| Who writes the vector code | you, with intrinsics | nobody — you write scalar code |
| Leftover work | tail loop `n % 8` | partially filled warp / bounds check |
| Divergence | manual masks (`_mm256_blendv_ps`) | hardware predication; both sides still execute |
| Reduction | `hadd` tree across lanes | `__shfl_down_sync` tree across lanes |

Three kernels make the point:

- **SAXPY** (`y = a*x + y`) — 2 FLOPs per 12 bytes. Deeply memory bound. AVX2 will
  give you **well under 8×**, because 8-wide registers do not make DRAM faster.
- **Degree-15 polynomial** (Horner) — 30 FLOPs per 8 bytes, intensity ~3.75.
  Compute bound, so AVX2 gets close to the full 8×.
- **Dot product** — memory bound *and* it needs a horizontal reduction, the
  log₂(lanes) tree you will rewrite with `__shfl_down_sync` in Phase 3.

One non-obvious detail in the dot product: use **four independent accumulators**,
not one. An FMA has ~4 cycles of latency but issues every cycle. A single
accumulator serialises on its own dependency chain and reaches a quarter of peak.
The same instruction-level-parallelism trick reappears in the GPU reduction.

## Your task

Open `main.cpp`. The intrinsics you need:

| Intrinsic | Meaning |
|---|---|
| `__m256` | 8 floats |
| `_mm256_set1_ps(a)` | broadcast one float to all lanes |
| `_mm256_loadu_ps(p)` | load 8 floats (`u` = unaligned, always safe) |
| `_mm256_fmadd_ps(a,b,c)` | `a*b + c`, one instruction, one rounding |
| `_mm256_storeu_ps(p,v)` | store 8 floats |

1. **`saxpyAVX2()`** — main loop 8 at a time, then a scalar tail for `n % 8`.
2. **`dotAVX2()`** — vector accumulator plus a horizontal reduction
   (`_mm256_extractf128_ps` → `_mm_add_ps` → `_mm_hadd_ps` ×2 → `_mm_cvtss_f32`).
3. **`polyAVX2()`** — the Horner loop, 8 elements at a time.
4. **Benchmark** all three against the scalar versions.

No special build flags are needed: the functions carry
`__attribute__((target("avx2,fma")))`, and `main()` checks
`__builtin_cpu_supports("avx2")` before calling them.

## Build and run

```bash
cmake --build build --target p1_06_simd_intrinsics -j
./build/bin/p1/p1_06_simd_intrinsics
```

## Expected output

```
--- Memory-bound kernel: SAXPY ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
scalar                                 ~6.0      ~8.4       ~1.4      1.00x
AVX2 (8 lanes)                         ~5.6      ~9.0       ~1.5     ~1.07x     <- barely moves

--- Compute-bound kernel: degree-15 polynomial ---
scalar                                ~14.0       ...      ~9.0      1.00x
AVX2 (8 lanes)                         ~2.0       ...     ~63.0      ~7x        <- near-perfect
```

If your scalar polynomial is already fast, `-O3` auto-vectorised it for you.
Check with `g++ -O3 -fopt-info-vec`.

## Key takeaways

- **SIMD width only helps compute-bound code.** For memory-bound kernels the bus
  is the limit and 8 lanes change nothing. This is the single most common reason
  a "parallelised" kernel does not get faster — on a CPU *or* a GPU.
- **The tail loop is wasted hardware.** A warp with 3 active threads out of 32
  costs the same as a full one.
- **Reductions need a tree.** The `hadd` cascade here is structurally identical to
  `__shfl_down_sync`.
- **Latency needs multiple accumulators.** Throughput ≠ latency; keep several
  independent chains in flight.

## Going further

- Add an AVX-512 path (`__m512`, 16 floats) with a second `target` attribute and
  dispatch at run time. Does 16 lanes help SAXPY at all?
- Implement a masked tail with `_mm256_maskload_ps` instead of the scalar loop.
- Compare against `#pragma omp simd` — how close does the compiler get?
- Read the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).
