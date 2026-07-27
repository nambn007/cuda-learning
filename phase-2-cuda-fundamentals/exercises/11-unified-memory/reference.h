#pragma once
// ============================================================
// reference.h - unified (managed) memory
// ============================================================
// cudaMallocManaged returns ONE pointer that is valid on both the
// host and the device. No cudaMemcpy, no separate h_ and d_
// variables, no forgetting which one you are holding.
//
// The price is that the driver has to move the pages for you, and
// it only learns where they are needed by trapping a PAGE FAULT.
// On Pascal and later:
//
//   host writes  -> pages migrate to host memory
//   kernel reads -> every first touch faults, the driver migrates
//                   the page, the warp stalls
//   host reads   -> they all migrate back
//
// Fault-driven migration at 4 KB granularity is far slower than one
// big DMA transfer. The fix is to tell the driver what you already
// know:
//
//   cudaMemPrefetchAsync(p, bytes, device)  move it now, in bulk
//   cudaMemAdvise(..., SetReadMostly)       replicate read-only data
//   cudaMemAdvise(..., SetPreferredLocation) pin the home node
//
// With a prefetch, managed memory costs about the same as an
// explicit copy. Without one it can be several times slower.
//
// The feature that has no equivalent in the explicit API is
// OVERSUBSCRIPTION: you can allocate more managed memory than the
// GPU physically has, and the driver pages it in and out. That is
// what makes it possible to run a model larger than VRAM.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 256 MB per array - large enough that migration cost is obvious.
inline constexpr size_t kElements = 64u * 1024u * 1024u;

inline void axpyCPU(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

inline double axpyBytes(size_t n) {
    return static_cast<double>(n) * 3.0 * sizeof(float);
}
