#pragma once
// ============================================================
// reference.h - mapping data onto threads
// ============================================================
// A CUDA launch produces a 1D, 2D or 3D grid of blocks, each
// containing a 1D, 2D or 3D block of threads. Choosing that shape
// is the first design decision of every kernel, and getting the
// index arithmetic wrong is the most common beginner bug.
//
// Rule of thumb: match the shape of the grid to the shape of the
// DATA, then make sure that adjacent THREADS touch adjacent
// ADDRESSES. The second half of that rule is what makes memory
// accesses coalesce, and it is why x is the fastest-varying
// dimension in every example here.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 1D problem
inline constexpr int kLength = 100000;

// 2D problem (an image)
inline constexpr int kWidth = 1920;
inline constexpr int kHeight = 1080;

// 3D problem (a simulation volume)
inline constexpr int kDimX = 128;
inline constexpr int kDimY = 64;
inline constexpr int kDimZ = 32;

// Row-major flattening, identical to the CPU convention from
// Phase 1. Note that `x` is the fastest-varying index, so threads
// with consecutive threadIdx.x land on consecutive addresses.
inline int flatten2D(int x, int y, int width) { return y * width + x; }
inline int flatten3D(int x, int y, int z, int width, int height) {
    return (z * height + y) * width + x;
}

// Golden results the kernels must reproduce.
inline std::vector<int> golden1D(int n) {
    std::vector<int> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = i;
    return v;
}

inline std::vector<int> golden2D(int width, int height) {
    std::vector<int> v(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) v[static_cast<size_t>(flatten2D(x, y, width))] = x + y;
    return v;
}

inline std::vector<int> golden3D(int dx, int dy, int dz) {
    std::vector<int> v(static_cast<size_t>(dx) * dy * dz);
    for (int z = 0; z < dz; ++z)
        for (int y = 0; y < dy; ++y)
            for (int x = 0; x < dx; ++x)
                v[static_cast<size_t>(flatten3D(x, y, z, dx, dy))] = x + y + z;
    return v;
}
