#pragma once
// ============================================================
// reference.h - parallel reduction, seven ways
// ============================================================
// Reduction turns n values into one. It looks trivial and it is
// the best optimisation exercise on a GPU, because every version
// below is correct and each one is faster than the last for a
// different reason.
//
// The kernel reads n floats and writes almost nothing, so it is
// purely memory bound. Its ceiling is the DRAM bandwidth you
// measured in exercise 01 - roughly 330 GB/s on an RTX 3060, so
// about 0.39 ms for 128 MB. Every variant is scored against that.
//
// The progression, and what each step fixes:
//
//   v1  interleaved addressing, `tid % (2*s) == 0`
//         -> the branch DIVERGES: in a warp of 32, most threads do
//            nothing while the rest work, and both paths cost time
//   v2  interleaved addressing, index = 2*s*tid
//         -> no divergence, but the shared-memory stride is a power
//            of two, so it hits BANK CONFLICTS (exercise 03)
//   v3  sequential addressing, stride halves each step
//         -> no divergence, no conflicts, but HALF the threads are
//            idle from the very first iteration
//   v4  first add during load
//         -> each thread sums two elements before the tree starts,
//            so the grid is half the size and no thread is wasted
//   v5  unroll the last warp with __shfl_down_sync
//         -> the final 5 iterations need no shared memory and no
//            barriers at all
//   v6  grid-stride loop + full warp-shuffle reduction
//         -> the grid is sized to the GPU, each thread sums many
//            elements, and shared memory is used only once
//
// v1 to v6 is typically 5-10x, on a kernel that never changes what
// it computes.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 32u * 1024u * 1024u;  // 128 MB
inline constexpr int kBlockSize = 256;

// Accumulate in double so the reference is not itself limited by
// float rounding over 32M terms.
inline double sumCPU(const float* v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(v[i]);
    return s;
}

// One read per element; the writes are negligible.
inline double reductionBytes(size_t n) {
    return static_cast<double>(n) * sizeof(float);
}
