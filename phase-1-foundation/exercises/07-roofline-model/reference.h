#pragma once
// ============================================================
// reference.h - the roofline model
// ============================================================
// The roofline answers the only question that matters before you
// optimise anything: IS THIS KERNEL LIMITED BY MEMORY OR BY MATH?
//
//   attainable GFLOP/s = min( peak GFLOP/s,
//                             arithmetic intensity * peak GB/s )
//
// Arithmetic intensity (AI) is FLOPs performed per byte moved. Plot
// AI on the x-axis and attainable performance on the y-axis and you
// get two straight lines: a sloped one (bandwidth limited) and a
// flat one (compute limited). Where they meet is the RIDGE POINT.
//
//   AI < ridge  -> memory bound. Optimise data movement: better
//                  layout, coalescing, tiling, fusion. Adding
//                  arithmetic is free here.
//   AI > ridge  -> compute bound. Optimise instructions: SIMD,
//                  FMA, fast math, Tensor Cores.
//
// Typical ridge points:
//   desktop CPU   ~10 FLOP/byte
//   RTX 3060      ~37 FLOP/byte  (13100 GFLOP/s / 360 GB/s)
//   A100          ~50+ FLOP/byte
//
// The ridge point rises with every hardware generation because
// arithmetic gets cheaper faster than memory does. That is why
// almost every kernel you will ever profile turns out to be memory
// bound, and why Phase 3 is mostly about memory.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// Small enough to live in L1/L2, so the compute measurement is not
// contaminated by DRAM traffic.
inline constexpr size_t kComputeElements = 8192;

// Far larger than any cache, so the bandwidth measurement really
// does hit DRAM.
inline constexpr size_t kMemoryElements = 16u * 1024u * 1024u;

// FMA counts to sweep. Each step doubles the arithmetic intensity.
inline const std::vector<int> kIntensitySweep = {1, 2, 4, 8, 16, 32, 64, 128, 256};

// Per element the kernel reads 4 bytes and writes 4 bytes, and
// performs 2 FLOPs per FMA.
inline double intensityOf(int fmaCount) {
    return (2.0 * fmaCount) / 8.0;
}

inline double intensityFlops(size_t n, int fmaCount) {
    return static_cast<double>(n) * 2.0 * fmaCount;
}

inline double intensityBytes(size_t n) {
    return static_cast<double>(n) * 2.0 * sizeof(float);
}

// The roofline itself.
inline double rooflineBound(double ai, double peakGflops, double peakGBs) {
    double memoryBound = ai * peakGBs;
    return memoryBound < peakGflops ? memoryBound : peakGflops;
}

inline double ridgePoint(double peakGflops, double peakGBs) {
    return peakGBs > 0.0 ? peakGflops / peakGBs : 0.0;
}
