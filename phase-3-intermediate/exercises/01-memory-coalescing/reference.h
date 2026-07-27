#pragma once
// ============================================================
// reference.h - memory coalescing, measured
// ============================================================
// This is the single most important performance fact about GPUs,
// and it is the same fact you measured on the CPU in Phase 1
// exercise 05:
//
//   MEMORY IS NOT MOVED IN BYTES, IT IS MOVED IN BLOCKS.
//
// An NVIDIA GPU services global memory in 32-byte SECTORS. When a
// warp issues a load, the hardware works out how many distinct
// sectors the 32 addresses fall into and fetches all of them.
//
//   32 consecutive floats  = 128 bytes = 4 sectors, all bytes used
//   32 floats, stride 8    = 32 distinct sectors, 4 of 32 bytes used
//
// The second case moves 8x the data for the same result. Nothing in
// the source code looks different, nothing warns you, and the
// answer is identical.
//
// Two separate effects are measured here:
//
// 1. STRIDE. As the gap between neighbouring threads' addresses
//    grows, the sectors per warp grow with it, until every thread
//    has its own sector and the curve flattens - you cannot waste
//    more than a whole sector per access.
//
// 2. ALIGNMENT. Even a perfectly contiguous access pays for one
//    extra sector per warp if it does not start on a sector
//    boundary. cudaMalloc returns 256-byte-aligned memory, so this
//    only bites when you offset into an array.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 256 MB: far larger than the 2.25 MB L2 on an RTX 3060, so the
// measurement really reaches DRAM rather than cache.
inline constexpr size_t kElements = 64u * 1024u * 1024u;

inline const std::vector<int> kStrides = {1, 2, 4, 8, 16, 32, 64, 128};
inline const std::vector<int> kOffsets = {0, 1, 2, 4, 8, 16, 31, 32};

inline constexpr int kSectorBytes = 32;
inline constexpr int kWarpSize = 32;

// Bytes the program asked for.
inline double usefulBytes(size_t touched) {
    return static_cast<double>(touched) * 2.0 * sizeof(float);  // read + write
}

// Sectors a warp must fetch for a strided read of 32 floats.
// With stride s (in floats), consecutive threads are 4s bytes
// apart. Once 4s >= 32 every thread needs its own sector.
inline double sectorsPerWarp(int strideElements) {
    const double bytesApart = 4.0 * strideElements;
    if (bytesApart >= kSectorBytes) return kWarpSize;
    const double perSector = kSectorBytes / bytesApart;
    return kWarpSize / perSector;
}

// Bytes the hardware actually has to move for the read side.
inline double predictedReadBytes(size_t touched, int strideElements) {
    const double warps = static_cast<double>(touched) / kWarpSize;
    return warps * sectorsPerWarp(strideElements) * kSectorBytes;
}
