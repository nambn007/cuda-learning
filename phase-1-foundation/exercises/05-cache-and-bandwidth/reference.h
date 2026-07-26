#pragma once
// ============================================================
// reference.h - measuring the memory hierarchy
// ============================================================
// This exercise is the CPU rehearsal for Phase 3 exercise 01
// (memory coalescing). The physical fact being demonstrated is the
// same on both processors:
//
//   MEMORY IS NOT MOVED IN BYTES, IT IS MOVED IN BLOCKS.
//
// x86 moves 64-byte cache lines. An NVIDIA GPU moves 32-byte
// sectors grouped into 128-byte transactions. In both cases, if
// you use 4 bytes out of every block you fetch, you are wasting
// 15/16 of your memory bandwidth - and bandwidth, not arithmetic,
// is what limits almost every real kernel.
// ============================================================

#include <cstddef>
#include <cstdint>
#include <vector>

#include "verify.h"

inline constexpr size_t kCacheLineBytes = 64;

// Big enough that it cannot sit in any level of cache, so every
// measurement really does reach DRAM.
inline constexpr size_t kStreamElements = 16u * 1024u * 1024u;  // 64 MB of float

// Strides to sweep, in units of float (4 bytes).
// stride 16 * 4 B = 64 B = exactly one cache line, which is where
// the useful-bandwidth curve should flatten out.
inline const std::vector<int> kStrides = {1, 2, 4, 8, 16, 32, 64, 128};

// Working-set sizes for the cache-level sweep, in KB.
inline const std::vector<size_t> kWorkingSetsKB = {4,    16,    64,    256,
                                                   1024, 4096,  16384, 65536};

// Useful bytes: what the program asked for.
inline double usefulBytes(size_t touchedElements) {
    return static_cast<double>(touchedElements) * sizeof(float);
}

// Bytes the hardware actually moved: one whole cache line per
// distinct line touched. With stride s (in floats), consecutive
// touches land on the same line only while s < 16.
inline double dramBytes(size_t touchedElements, int strideElements) {
    double elementsPerLine = static_cast<double>(kCacheLineBytes) / sizeof(float);
    double touchesPerLine = elementsPerLine / strideElements;
    if (touchesPerLine < 1.0) touchesPerLine = 1.0;
    return static_cast<double>(touchedElements) / touchesPerLine * kCacheLineBytes;
}
