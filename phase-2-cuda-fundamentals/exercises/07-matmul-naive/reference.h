#pragma once
// ============================================================
// reference.h - the GPU matmul baseline
// ============================================================
// Phase 1 exercise 01 wrote this on the CPU and got about 1 GFLOP/s.
// Now the same triple loop runs with one thread per output element.
//
// The interesting part is what the arithmetic intensity actually is.
// Naively you might say: 2*N^3 FLOPs over 3*N^2 floats of data, so
// intensity grows with N and the kernel is compute bound. That is
// true of the ALGORITHM but false of THIS KERNEL, because every
// thread re-reads its whole row of A and column of B from global
// memory:
//
//   traffic  = 2 * N   floats per output element
//   FLOPs    = 2 * N   per output element
//   intensity = 2N / (2N * 4 bytes) = 0.25 FLOP/byte
//
// Against a ridge point near 37, that is firmly memory bound - and
// it stays 0.25 no matter how large N gets. The whole of Phase 3
// exercise 04 is about fixing exactly this number by keeping tiles
// in shared memory so that each loaded value is used many times.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// 1024^3 = 2.1 GFLOP of work. Small enough that the CPU reference
// finishes in a few seconds, large enough to saturate the GPU.
inline constexpr int kSize = 1024;

inline void matmulCPU(const float* A, const float* B, float* C, int n) {
    // ikj order (Phase 1 exercise 02) so the reference is not
    // needlessly slow.
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

// What the NAIVE kernel actually moves: every thread reads a full
// row of A and a full column of B.
inline double naiveTrafficBytes(int n) {
    double N = static_cast<double>(n);
    return N * N * 2.0 * N * sizeof(float);
}

// The theoretical minimum: read A and B once, write C once.
inline double minimumTrafficBytes(int n) {
    double N = static_cast<double>(n);
    return 3.0 * N * N * sizeof(float);
}

struct MatmulProblem {
    int n;
    std::vector<float> A, B, golden;

    explicit MatmulProblem(int size)
        : n(size),
          A(static_cast<size_t>(size) * size),
          B(static_cast<size_t>(size) * size),
          golden(static_cast<size_t>(size) * size) {
        fillRandom(A.data(), A.size(), -1.0f, 1.0f, 61);
        fillRandom(B.data(), B.size(), -1.0f, 1.0f, 62);
        matmulCPU(A.data(), B.data(), golden.data(), n);
    }

    size_t elements() const { return static_cast<size_t>(n) * n; }
    size_t bytes() const { return elements() * sizeof(float); }
};
