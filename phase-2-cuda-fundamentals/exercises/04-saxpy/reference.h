#pragma once
// ============================================================
// reference.h - SAXPY and vectorised loads
// ============================================================
// SAXPY (Single-precision A times X Plus Y) is the canonical
// memory-bound kernel: y[i] = a * x[i] + y[i]. It reads 8 bytes,
// writes 4, and does one FMA. Intensity is 2/12 = 0.167 FLOP/byte.
//
// Exercise 03 established that such a kernel is limited by DRAM.
// This one asks the follow-up question: given that we are limited
// by bandwidth, are we USING all of it? The answer turns out to
// depend on how many bytes each thread requests at once.
//
// A single float load issues a 4-byte request. A float4 load issues
// one 16-byte request instead of four 4-byte ones, which cuts the
// number of memory instructions by 4 and often measurably improves
// throughput on a kernel that has nothing else to do.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr size_t kElements = 1u << 25;  // 32M floats = 128 MB
inline constexpr float kAlpha = 2.5f;

inline void saxpyCPU(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// Read x, read y, write y.
inline double saxpyBytes(size_t n) {
    return static_cast<double>(n) * 3.0 * sizeof(float);
}

// One multiply and one add per element.
inline double saxpyFlops(size_t n) { return static_cast<double>(n) * 2.0; }

struct SaxpyProblem {
    size_t n;
    std::vector<float> x, y0, golden;

    explicit SaxpyProblem(size_t size) : n(size), x(size), y0(size), golden(size) {
        fillRandom(x.data(), n, -1.0f, 1.0f, 41);
        fillRandom(y0.data(), n, -1.0f, 1.0f, 42);
        golden = y0;
        saxpyCPU(kAlpha, x.data(), golden.data(), n);
    }
};
