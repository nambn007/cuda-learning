<!-- Language: **English** | [Tiếng Việt](README.vi.md) -->

# 01 - Thrust and CUB

> **Phase 5 · Expert** | Difficulty: ⭐⭐ | Time: ~2 h | Prerequisites: [Phase 3/06](../../../phase-3-intermediate/exercises/06-reduction-variants/) | **Requires a GPU** | ✅ verified on an RTX 3060

## Goal

Calibration. Find out how far your hand-written Phase 3 kernels actually are from
the libraries, so that in future you know when writing your own is justified.

## Background

**Thrust** is an STL-style layer — `sort`, `reduce`, `transform`, `inclusive_scan`
over device vectors, with iterators and functors. Header-only, ships with the
toolkit.

**CUB** is the layer underneath, exposing the same algorithms at **warp, block and
device** level, so you can drop `cub::BlockReduce` into the middle of your own
kernel instead of calling a separate launch. Thrust is built on it.

**CUB's two-call convention** — it never allocates for you:

```cuda
cub::DeviceReduce::Sum(nullptr, tempBytes, ...);   // ask for the size
cudaMalloc(&d_temp, tempBytes);                    // allocate
cub::DeviceReduce::Sum(d_temp, tempBytes, ...);    // run
```

Deliberate: one scratch buffer gets reused across many calls instead of allocating
in a hot loop.

## Your task

Bring your **v7** reduction across from Phase 3/06, then compare it against
`cub::DeviceReduce::Sum` and `thrust::reduce`. Report **percentage of peak
bandwidth**, not milliseconds. Then run `cub::DeviceScan` and `thrust::sort`.

**Predict where your kernel lands before you look.**

## Build and run

```bash
cmake --build build --target p5_01_thrust_and_cub -j
./build/bin/p5/p5_01_thrust_and_cub
```

## Expected output

RTX 3060, 16M floats:

```
hand-written (Phase 3/06 v7)          0.202    332.67 GB/s     1.00x
cub::DeviceReduce::Sum                0.203    331.25 GB/s     1.00x
thrust::reduce                        0.220    304.77 GB/s     0.92x

  As a fraction of peak bandwidth:
    hand-written : 92%     CUB : 92%     Thrust : 85%

cub::DeviceScan                       0.413    325.24 GB/s
thrust::inclusive_scan                0.423    317.29 GB/s

thrust::sort : 3132 million keys per second
```

**Your Phase 3 kernel matches CUB exactly.** That is the payoff for having written
it seven times — you now know what "as good as the library" looks like, and you can
recognise it in your own code.

Thrust's 85% is the price of its generality; CUB is what you use when that 7%
matters.

## Key takeaways

- **A well-written hand kernel can match CUB** for a simple memory-bound
  operation — and you now know that from measurement, not faith.
- **Scan is not sequential.** CUB's single-pass decoupled look-back reaches near
  copy bandwidth; writing that yourself is a multi-week project.
- **Sort is where hand-writing is clearly wrong.** One line versus a serious project.
- **Thrust for clarity, CUB for control, your own only after measuring both.**

## Going further

- Use `cub::BlockReduce` *inside* one of your Phase 3 kernels.
- Sort key-value pairs with `thrust::sort_by_key`.
- Try `thrust::transform_reduce` with a custom functor — one pass instead of two.
- Compare `cub::DeviceRadixSort` against `thrust::sort` directly.
