#pragma once
// ============================================================
// reference.h - pinned memory and stream overlap
// ============================================================
// Phase 2 exercise 03 ended on an uncomfortable result: for simple
// element-wise work the GPU loses, because the PCIe round trip
// costs more than the arithmetic saves. This exercise is the first
// real answer to that.
//
// Two independent fixes:
//
// 1. PINNED (page-locked) HOST MEMORY. Ordinary host allocations
//    can be swapped out, so the driver cannot hand their addresses
//    to the DMA engine: it copies through an internal staging
//    buffer first. cudaMallocHost gives memory the OS cannot move,
//    which the DMA engine reads directly - typically 1.5-2x faster.
//
//    It is also a HARD REQUIREMENT for asynchrony:
//    cudaMemcpyAsync on pageable memory silently behaves
//    synchronously, so a "pipeline" built on pageable memory
//    overlaps nothing at all and looks correct while doing so.
//
// 2. STREAMS. A stream is an ordered queue of GPU work. Operations
//    in different streams may run concurrently, and a GPU has
//    separate copy engines for each direction plus the SMs, so
//    with enough streams you can have H2D, compute and D2H all in
//    flight at once.
//
//    Split the data into chunks and the total time goes from
//        H2D + kernel + D2H
//    towards
//        max(H2D, kernel, D2H) + one chunk of latency
//
// The trap: the DEFAULT STREAM synchronises with every other stream
// (unless compiled with --default-stream per-thread). One stray
// launch or cudaMemcpy without a stream argument in the middle of a
// pipeline serialises the whole thing, and nothing warns you - the
// answer stays correct and the timeline quietly collapses.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 64u * 1024u * 1024u;  // 256 MB
inline constexpr int kStreams = 4;
inline constexpr int kChunks = 16;
inline constexpr float kAlpha = 2.0f;

// Deliberately more arithmetic than a plain saxpy, so the kernel
// takes long enough to be worth overlapping with. A kernel that
// finishes instantly has nothing to hide behind.
inline void workloadCPU(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        for (int k = 0; k < 8; ++k) v = v * a + 1.0f;
        y[i] = v;
    }
}

inline double transferBytes(size_t n) {
    return static_cast<double>(n) * 2.0 * sizeof(float);  // in and out
}
