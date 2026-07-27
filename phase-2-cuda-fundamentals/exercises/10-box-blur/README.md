<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 10 - Box blur: stencils, halos and separability

> **Phase 2 · CUDA Fundamentals** | Difficulty: ⭐⭐ | Time: ~2 h | Prerequisites: [09](../09-rgb-to-grayscale/) | **Requires a GPU**

## Goal

Write your first **stencil** — the pattern behind convolution, image filters, PDE
solvers and cellular automata — and learn that the biggest win available is not a
GPU technique at all. It is algebra.

## Background

A box blur replaces each pixel by the average of the `(2R+1)²` pixels around it.
Every stencil shares three problems:

**1. Reuse.** Neighbouring outputs read overlapping input. At radius 4 the naive
kernel reads each input pixel up to **81 times**. Caches recover much of that, but
it is still the dominant cost.

**2. Boundaries.** Edge pixels have no neighbours on one side. Clamping (repeat the
border pixel) is the standard answer, and it is not free — this exercise measures
exactly how much it costs, and the number is larger than most people guess.

**3. Separability.** A box filter is the *product* of a horizontal box and a
vertical box, so blurring rows then columns gives **exactly** the same answer —
not an approximation — while reading `2(2R+1)` pixels instead of `(2R+1)²`:

| Radius | Naive taps | Separable taps | Ratio |
|---|---|---|---|
| 2 | 25 | 10 | 2.5× |
| 4 | 81 | 18 | **4.5×** |
| 8 | 289 | 34 | 8.5× |

This is a saving from **algebra**, not from any GPU trick. It applies on a CPU
too, it composes with every other optimisation, and it grows with the radius.
**Always look for separability before reaching for shared memory.**

**A note on precision:** everything here is integer arithmetic, so host and device
agree bit for bit and you can use `checkArrayExact()` — unlike the float luma in
exercise 09. The intermediate buffer must be wider than a byte, though: a
horizontal sum of nine pixels reaches 2295.

## Your task

Open `main.cu`:

1. **`blur2D()`** — the naive `(2R+1)²` stencil with `clampDev()` at the borders.
   Round with `(sum + window/2) / window` to match the CPU exactly.
2. **`blurHorizontal()` / `blurVertical()`** — the two separable passes. Work out
   why `tmp` is `unsigned short`.
3. **`blur2DNoBoundary()`** — interior only, no clamping. It is deliberately wrong
   near the edges; it exists to measure what the clamping cost.
4. **Report reads-per-pixel alongside the measured times**, so you can see how much
   of the predicted 4.5× actually materialises.

## Build and run

```bash
cmake --build build --target p2_10_box_blur -j
cd build/testrun && ../bin/p2/p2_10_box_blur
```

Writes `output_blur.pgm` — open it. The checkerboard should be soft and the circle
should have a fuzzy edge. If the borders look wrong, the clamping is wrong.

## Expected output

RTX 3060, 2048×2048, radius 4:

```
--- Correctness ---
  [PASS] naive 2D stencil (n=8388608, exact match)
  [PASS] separable (two 1D passes) (n=8388608, exact match)

--- Performance ---
Variant                           Time (ms)      GB/s   Speedup
----------------------------------------------------------------
CPU naive 2D                        263.170      2.61     1.00x
GPU naive 2D                          1.888    364.29   139.37x
GPU 2D, no boundary handling          0.918    748.93   286.53x
GPU separable (2 passes)              0.588    285.44   447.74x

  Reads per output pixel
    naive 2D    : 81       separable : 18       ratio : 4.5x fewer
    measured    : 3.21x faster
```

Two results worth sitting with:

- **Separability delivers 3.21× of a predicted 4.5×.** The gap is the second
  pass's extra kernel launch and its round trip through the intermediate buffer.
- **Boundary handling costs 2.06×.** Not from warp divergence — only the border
  warps diverge — but from the sheer count: **162 integer min/max pairs per output
  pixel**, all in the innermost loop of a kernel that is otherwise almost pure
  loads.

## Key takeaways

- **Look for algebraic structure first.** Separability beat every micro-optimisation
  available at this stage, and it would be worth 8.5× at radius 8.
- **Boundary handling is a real cost, not a detail.** The production fix is to pad
  the input by `radius` pixels so no clamping is needed at all, or to split the
  launch into an interior kernel and a thin border kernel.
- **Integer arithmetic gives bit-exact agreement** with a host reference. Use it
  when you can.
- Caches hide a lot of redundant reading — which is why the naive version is 139×
  faster than the CPU despite reading everything 81 times.

## Going further

- Rerun at radius 2, 8 and 16. Does the separable speedup track `(2R+1)/2`?
- A box blur has a **running-sum** formulation that is O(1) per pixel regardless of
  radius. Derive it, implement it, and compare.
- Three box blurs in a row approximate a Gaussian. Try it and look at the result.
- **Next:** [Phase 3/14 conv-2d-shared](../../../phase-3-intermediate/exercises/14-conv-2d-shared/)
  stages a tile *plus its halo* in `__shared__` memory so each input pixel is read
  from global memory exactly once, and
  [Phase 3/13](../../../phase-3-intermediate/exercises/13-conv-1d-constant/) puts
  the filter weights in `__constant__` memory.
