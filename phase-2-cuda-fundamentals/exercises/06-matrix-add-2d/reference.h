#pragma once
// ============================================================
// reference.h - 2D launches, and why the axis order matters
// ============================================================
// Adding two matrices is arithmetically identical to adding two
// vectors: the 2D shape exists only in your head, since row-major
// storage is one contiguous array either way.
//
// So why launch a 2D grid at all? Convenience, mostly: a stencil,
// a convolution or a transpose all need the (row, column) of the
// element, and computing it from a flat index costs a division.
//
// The part that is NOT cosmetic is which axis you map to x. Threads
// in a warp differ in threadIdx.x first, so
//
//   x = column  -> a warp touches 32 consecutive addresses -> fast
//   x = row     -> a warp touches 32 addresses `width` apart -> slow
//
// Both give the right answer. One of them is roughly an order of
// magnitude slower, and nothing warns you. This exercise measures
// the difference; Phase 3 exercise 01 explains the hardware reason.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 4096 x 4096 floats = 64 MB per matrix, 192 MB of traffic.
inline constexpr int kRows = 4096;
inline constexpr int kCols = 4096;

inline void matrixAddCPU(const float* a, const float* b, float* c, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int col = 0; col < cols; ++col) {
            size_t i = static_cast<size_t>(r) * cols + col;
            c[i] = a[i] + b[i];
        }
}

inline double matrixAddBytes(int rows, int cols) {
    return static_cast<double>(rows) * cols * 3.0 * sizeof(float);
}

struct MatrixProblem {
    int rows, cols;
    std::vector<float> a, b, golden;

    MatrixProblem(int r, int c)
        : rows(r),
          cols(c),
          a(static_cast<size_t>(r) * c),
          b(static_cast<size_t>(r) * c),
          golden(static_cast<size_t>(r) * c) {
        fillRandom(a.data(), a.size(), -1.0f, 1.0f, 51);
        fillRandom(b.data(), b.size(), -1.0f, 1.0f, 52);
        matrixAddCPU(a.data(), b.data(), golden.data(), rows, cols);
    }

    size_t elements() const { return static_cast<size_t>(rows) * cols; }
    size_t bytes() const { return elements() * sizeof(float); }
};
