#pragma once
// ============================================================
// reference.h - problem setup for the cache-blocking exercise
// ============================================================
// The golden result here is the naive ijk loop you wrote in
// exercise 01. Your job now is to compute the SAME numbers much
// faster, purely by changing the order in which memory is touched.
// ============================================================

#include <algorithm>
#include <cstddef>
#include <vector>

#include "verify.h"

inline int idx(int row, int col, int n) { return row * n + col; }

// Exercise 01's loop order, kept as the correctness reference.
inline void matmulGolden(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc += A[idx(i, k, n)] * B[idx(k, j, n)];
            C[idx(i, j, n)] = acc;
        }
    }
}

inline double matmulFlops(int n) {
    double N = static_cast<double>(n);
    return 2.0 * N * N * N;
}

// ------------------------------------------------------------
// How much data does each variant actually pull from DRAM?
// ------------------------------------------------------------
// Naive ijk: for every one of the N^2 output elements it streams a
// whole column of B. B does not fit in cache, so essentially every
// B access is a miss and drags in a full 64-byte line for 4 useful
// bytes. Traffic is on the order of N^3 * 64 bytes.
inline double naiveEstimatedBytes(int n) {
    double N = static_cast<double>(n);
    return N * N * N * 64.0;
}

// Blocked: each T x T tile of C is computed from N/T pairs of tiles
// of A and B, and every tile is small enough to sit in L1/L2. Each
// element of A and B is read N/T times instead of N times.
inline double blockedEstimatedBytes(int n, int tile) {
    double N = static_cast<double>(n);
    double T = static_cast<double>(tile);
    return 2.0 * (N * N) * (N / T) * sizeof(float) + N * N * sizeof(float);
}

struct MatmulProblem {
    int n;
    std::vector<float> A, B, golden;

    explicit MatmulProblem(int size) : n(size) {
        size_t total = static_cast<size_t>(n) * n;
        A.resize(total);
        B.resize(total);
        golden.resize(total);
        fillRandom(A.data(), total, -1.0f, 1.0f, 11);
        fillRandom(B.data(), total, -1.0f, 1.0f, 22);
        matmulGolden(A.data(), B.data(), golden.data(), n);
    }

    size_t elements() const { return static_cast<size_t>(n) * n; }
};
