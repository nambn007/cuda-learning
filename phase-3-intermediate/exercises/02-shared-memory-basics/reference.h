#pragma once
// ============================================================
// reference.h - shared memory and __syncthreads
// ============================================================
// Shared memory is a small block of on-chip storage, private to a
// thread block, that you manage by hand. It is roughly as fast as
// L1 - about 100x lower latency than global memory - and there is
// very little of it: 48 KB per block by default, 100 KB per SM on
// an RTX 3060, shared between every block resident there.
//
// It exists to solve two problems, and every later exercise in
// Phase 3 is one of them:
//
//   REUSE      several threads need the same value. Read it from
//              global memory once, then serve it from shared.
//              (tiled matmul, convolution, reduction)
//
//   REORDERING one of your accesses cannot be coalesced. Stage the
//              data in shared memory, where the coalescing rules do
//              not apply, and rearrange it there.
//              (transpose, histogram)
//
// The cost of a block-private scratchpad is that you must
// synchronise by hand. __syncthreads() is a barrier: no thread in
// the block passes it until every thread has arrived, and all
// shared-memory writes issued before it are visible to all threads
// after it.
//
// Two rules that catch people:
//
// 1. __syncthreads() must be reached by EVERY thread in the block.
//    Putting one inside `if (tid < 100)` in a 256-thread block
//    hangs the block, or worse, silently misbehaves.
//
// 2. Since Volta, threads within a warp are NOT implicitly
//    synchronised. Code that omitted __syncthreads() and "worked"
//    on Kepler because a warp moved in lockstep is broken on every
//    modern GPU. Use __syncwarp() when warp-level sync is what you
//    actually mean.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kBlockSize = 256;
inline constexpr size_t kElements = 1u << 22;  // 4M floats

// Reverse each block-sized chunk independently.
inline void reverseBlocksCPU(const float* in, float* out, size_t n, int blockSize) {
    for (size_t base = 0; base < n; base += static_cast<size_t>(blockSize)) {
        const size_t count =
            (base + blockSize <= n) ? static_cast<size_t>(blockSize) : (n - base);
        for (size_t i = 0; i < count; ++i) out[base + i] = in[base + count - 1 - i];
    }
}

// Every thread in a block needs the sum of that block's elements -
// a small all-to-all, and the simplest possible demonstration of
// reuse. The naive version reads blockSize values from global
// memory per thread; the shared version reads one.
inline void blockBroadcastCPU(const float* in, float* out, size_t n, int blockSize) {
    for (size_t base = 0; base < n; base += static_cast<size_t>(blockSize)) {
        const size_t count =
            (base + blockSize <= n) ? static_cast<size_t>(blockSize) : (n - base);
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) sum += in[base + i];
        for (size_t i = 0; i < count; ++i) out[base + i] = in[base + i] * sum;
    }
}

// Global reads performed per output element.
inline double naiveReadsPerElement(int blockSize) {
    return static_cast<double>(blockSize) + 1.0;
}
inline double sharedReadsPerElement() { return 1.0; }
