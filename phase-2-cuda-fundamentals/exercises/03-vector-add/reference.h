#pragma once
// ============================================================
// reference.h - the whole point of this exercise
// ============================================================
// Vector addition is the "hello world" of GPU computing, and it is
// also the clearest demonstration of when a GPU is the WRONG tool.
//
// Per element the kernel reads 8 bytes, writes 4 and performs 1
// addition. Arithmetic intensity is 1/12 = 0.083 FLOP per byte.
// Your GPU's ridge point (Phase 1 exercise 08) is around 37. This
// kernel is therefore off the left edge of the roofline: it is
// pure memory traffic with a rounding error of arithmetic attached.
//
// And before the kernel can run at all, the inputs must cross PCIe
// at roughly 12 GB/s - about 30 times slower than the GPU's own
// memory. Count that transfer and the GPU can easily come out
// SLOWER end to end than the CPU it was supposed to accelerate.
//
// The lesson is not "do not use GPUs". It is: keep data resident on
// the device, and fuse work so that one transfer feeds many
// operations. Phase 5 exercise 14 makes that concrete.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 16M floats = 64 MB per array, 192 MB of traffic in total. Large
// enough that the measurement is not dominated by launch overhead,
// small enough to fit on any modern card.
inline constexpr size_t kElements = 1u << 24;

inline void vectorAddCPU(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// Two reads and one write per element.
inline double vectorAddBytes(size_t n) {
    return static_cast<double>(n) * 3.0 * sizeof(float);
}

// One addition per element.
inline double vectorAddFlops(size_t n) { return static_cast<double>(n); }

// Bytes that have to cross PCIe: two arrays in, one array out.
inline double transferBytes(size_t n) {
    return static_cast<double>(n) * 3.0 * sizeof(float);
}

struct VectorProblem {
    size_t n;
    std::vector<float> a, b, golden;

    explicit VectorProblem(size_t size) : n(size), a(size), b(size), golden(size) {
        fillRandom(a.data(), n, -1.0f, 1.0f, 31);
        fillRandom(b.data(), n, -1.0f, 1.0f, 32);
        vectorAddCPU(a.data(), b.data(), golden.data(), n);
    }
};
