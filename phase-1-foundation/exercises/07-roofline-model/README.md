<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 07 - The roofline model

> **Phase 1 · Foundation** | Difficulty: ⭐⭐⭐ | Time: ~2 h | Prerequisites: [05](../05-cache-and-bandwidth/), [06](../06-simd-intrinsics/)

## Goal

Build the roofline of your own machine from measurements, and learn to answer the
question that must come **before** any optimisation: *is this kernel limited by
memory or by arithmetic?* Guessing wrong means optimising the wrong thing, and
this single model prevents most wasted effort in the rest of the curriculum.

## Background

```
attainable GFLOP/s = min( peak GFLOP/s , arithmetic intensity × peak GB/s )
```

**Arithmetic intensity (AI)** = FLOPs performed per byte moved. Plot AI on the
x-axis and attainable performance on the y-axis and you get two lines: a sloped
one (bandwidth-limited) and a flat one (compute-limited). Where they meet is the
**ridge point**.

```
 GFLOP/s
   ^
   |            flat roof = peak compute
   |        ____________________________
   |       /
   |      /   <- sloped roof = AI × peak bandwidth
   |     /
   |    /
   +---+------------------------------------> arithmetic intensity
       ridge point
```

- **AI below the ridge → memory bound.** The only way up is to move fewer bytes:
  better layout, coalescing, tiling, kernel fusion. Extra arithmetic is *free*.
- **AI above the ridge → compute bound.** Now instructions matter: SIMD, FMA,
  fast math, Tensor Cores.

Typical ridge points: a desktop CPU ~10 FLOP/byte, an RTX 3060 ~37 FLOP/byte, an
A100 50+. **The ridge point rises with every hardware generation**, because
arithmetic gets cheaper faster than memory does. That is why almost every kernel
you will ever profile turns out to be memory bound — and why Phase 3 of this
curriculum is mostly about memory.

## Your task

Open `main.cpp`:

1. **`intensityKernel<TILE>(in, out, n, fmaCount)`** — read one float, do
   `fmaCount` FMAs, write one float. Traffic is fixed at 8 bytes/element while
   arithmetic scales, so sweeping `fmaCount` sweeps the AI.
   Structure it as a **register tile**: hold `TILE` elements with `TILE`
   independent accumulators. One accumulator waits on its own previous result and
   reaches a quarter of peak; many keep the FMA pipeline full. `TILE` is a
   template parameter so the compiler can fully unroll and keep the tile in
   registers.
2. **`triad()`** — `a[i] = b[i] + s*c[i]` for the bandwidth roof.
3. **Bandwidth roof** — triad on the 64 MB array.
4. **Compute roof** — run `fmaCount = 256` on a cache-resident array for
   `TILE` = 4, 8, 16, 32, 64, 128 and take the best. Watch throughput climb, then
   fall off when the tile stops fitting in registers.
5. **Sweep** `kIntensitySweep` (using the winning tile) and compare achieved
   against `rooflineBound()`.

## Build and run

```bash
cmake --build build --target p1_07_roofline_model -j
./build/bin/p1/p1_07_roofline_model
```

## Expected output

Measured single-threaded on a desktop CPU:

```
--- Roof 2 - FMA throughput (and the register cliff) ---
  tile width            GFLOP/s   independent FMA chains
  --------------------------------------------------------------
  4                         9.9   4
  8                        17.2   8
  16                       31.1   16
  32                       64.2   32
  64                      121.8   64
  128                     118.3   128

  Best: tile 64 at 121.8 GFLOP/s.

--- The roofline ---
  peak bandwidth               17.043 GB/s
  peak throughput             121.842 GFLOP/s
  ridge point                   7.149 FLOP/byte

--- Intensity sweep ---
  FMAs         AI     achieved     roofline  of roof  attainable performance
  --------------------------------------------------------------------------------
  1           0.2          3.6          4.3      84%  |            | memory bound
  4           1.0         14.0         17.0      82%  |###         | memory bound
  16          4.0         48.0         68.0      70%  |#####       | memory bound
  64         16.0        118.0        121.8      97%  |###########| compute bound
  256        64.0        119.0        121.8      98%  |###########| compute bound
```

Two things to notice. The tile sweep doubles throughput four times over — that is
pure **instruction-level parallelism**, no extra work done. And the bar chart in
the intensity sweep *is* the roofline: rising while memory bound, flat afterwards.

> **Both roofs here are single-threaded.** A whole chip's compute roof scales with
> core count while its bandwidth roof does not (all cores share one memory
> controller), so the full-chip ridge point is several times higher than the ~7
> you will measure. An RTX 3060 sits near **37** FLOP/byte.

## Key takeaways

- **Measure the roofs, do not trust the spec sheet.** Achievable bandwidth is
  typically 60–80% of theoretical.
- **Classify before optimising.** A memory-bound kernel will not get faster from
  better instructions, and a compute-bound one will not get faster from better
  layout.
- **Efficiency is relative to the roof, not to the peak.** A kernel at 4 GFLOP/s
  can be at 98% of its ceiling — and then it is *done*.
- **Latency needs independent work.** One dependency chain reaches a quarter of
  peak; the fix is more chains, until you run out of registers.
- The ridge point rises every generation, so **the fraction of kernels that are
  memory bound keeps growing**.

## Going further

- Add a third roof for L2-resident data: rerun the sweep on an array that fits in
  cache. You get a *hierarchy* of rooflines, which is how Nsight Compute presents
  it for a GPU.
- Compute the AI of the matmul variants from exercise 02 and place them on your
  chart. Does blocking move a kernel to the right?
- Read Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
  Model* (CACM 2009).
- In Phase 3 exercise 16, Nsight Compute will draw this chart for your GPU kernels
  automatically.
