#pragma once
// ============================================================
// reference.h - shared-memory tiling, the payoff exercise
// ============================================================
// Phase 2 exercise 07 measured a naive GPU matmul: perfectly
// coalesced, and still only ~6% of the card's FP32 peak, because
// every thread re-read its whole row of A and column of B from
// global memory.
//
//   intensity = 2N FLOPs / (2N * 4 bytes) = 0.25 FLOP per byte
//
// and that number does not improve with N. Against a ridge point
// near 37, it is hopeless.
//
// Tiling fixes the number itself. Load a TILE x TILE block of A and
// of B into shared memory, and every value loaded is used by TILE
// threads instead of 1:
//
//   intensity = 2 * TILE / (2 * 4 bytes) = TILE / 4 FLOP per byte
//
// TILE = 32 gives 8 FLOP/byte - a 32x improvement, and now within
// reach of the ridge. This is Phase 1 exercise 02's cache blocking,
// except that on a GPU you place the data by hand and the win is
// real rather than marginal.
//
// The second step doubles it again. If each thread computes TM
// outputs instead of 1, each value it reads from shared memory
// serves TM multiply-adds. Registers become a third level of tiling
// below shared memory - which is exactly how cuBLAS is structured,
// only with far more levels.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// Same size as Phase 2 exercise 07, so the numbers are comparable.
inline constexpr int kSize = 1024;
inline constexpr int kTile = 32;
inline constexpr int kThreadTile = 4;  // outputs per thread in the register-tiled version

inline void matmulCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        float* crow = C + static_cast<size_t>(i) * n;
        for (int j = 0; j < n; ++j) crow[j] = 0.0f;
        for (int k = 0; k < n; ++k) {
            const float a = A[static_cast<size_t>(i) * n + k];
            const float* brow = B + static_cast<size_t>(k) * n;
            for (int j = 0; j < n; ++j) crow[j] += a * brow[j];
        }
    }
}

inline double matmulFlops(int n) {
    double N = static_cast<double>(n);
    return 2.0 * N * N * N;
}

// Global loads issued per output element.
inline double naiveLoadsPerOutput(int n) { return 2.0 * n; }
inline double tiledLoadsPerOutput(int n, int tile) {
    return 2.0 * n / static_cast<double>(tile);
}

// FLOPs per byte loaded from global memory.
inline double naiveIntensity() { return 2.0 / (2.0 * sizeof(float)); }
inline double tiledIntensity(int tile) {
    return 2.0 * tile / (2.0 * sizeof(float));
}

struct MatmulProblem {
    int n;
    std::vector<float> A, B, golden;

    explicit MatmulProblem(int size)
        : n(size),
          A(static_cast<size_t>(size) * size),
          B(static_cast<size_t>(size) * size),
          golden(static_cast<size_t>(size) * size) {
        fillRandom(A.data(), A.size(), -1.0f, 1.0f, 91);
        fillRandom(B.data(), B.size(), -1.0f, 1.0f, 92);
        matmulCPU(A.data(), B.data(), golden.data(), n);
    }

    size_t elements() const { return static_cast<size_t>(n) * n; }
    size_t bytes() const { return elements() * sizeof(float); }
};
