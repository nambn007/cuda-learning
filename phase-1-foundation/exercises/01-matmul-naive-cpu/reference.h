#pragma once
// ============================================================
// reference.h - shared problem definition and golden result
// ============================================================
// Both main.cpp (your version) and solution.cpp (the reference)
// include this file, so both are checked against exactly the same
// golden result and measured with the same FLOP count.
//
// Note for this first exercise: the golden implementation below IS
// the algorithm you are asked to write. That is deliberate - the
// point of exercise 01 is not the maths, it is learning how to
// MEASURE. Try to write it without looking first.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

// ------------------------------------------------------------
// Layout
// ------------------------------------------------------------
// All matrices are square, N x N, and stored ROW-MAJOR: the
// element at row r and column c lives at index r * N + c. Row
// major means consecutive elements of a ROW are adjacent in
// memory; consecutive elements of a COLUMN are N floats apart.
// That single fact drives the whole of exercise 02.
inline int idx(int row, int col, int n) { return row * n + col; }

// ------------------------------------------------------------
// Golden C = A * B
// ------------------------------------------------------------
inline void matmulGolden(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) {
                acc += A[idx(i, k, n)] * B[idx(k, j, n)];
            }
            C[idx(i, j, n)] = acc;
        }
    }
}

// ------------------------------------------------------------
// Cost model
// ------------------------------------------------------------
// The inner loop runs N^3 times and does one multiply plus one
// add, so 2 * N^3 floating point operations in total. Dividing by
// the elapsed time gives FLOP/s, the number to compare against
// your CPU's peak - and later against the GPU.
inline double matmulFlops(int n) {
    double N = static_cast<double>(n);
    return 2.0 * N * N * N;
}

// Minimum traffic if every matrix were read/written exactly once.
// The naive algorithm moves far more than this because it re-reads
// B from DRAM over and over - which is exactly what exercise 02
// fixes.
inline double matmulMinBytes(int n) {
    double N = static_cast<double>(n);
    return 3.0 * N * N * sizeof(float);
}

// ------------------------------------------------------------
// Test data
// ------------------------------------------------------------
struct MatmulProblem {
    int n;
    std::vector<float> A, B, golden;

    explicit MatmulProblem(int size) : n(size) {
        size_t total = static_cast<size_t>(n) * n;
        A.resize(total);
        B.resize(total);
        golden.resize(total);

        // Values in [-1, 1] keep the accumulated sums small enough
        // that float rounding stays well inside our tolerance.
        fillRandom(A.data(), total, -1.0f, 1.0f, /*seed=*/11);
        fillRandom(B.data(), total, -1.0f, 1.0f, /*seed=*/22);

        matmulGolden(A.data(), B.data(), golden.data(), n);
    }

    size_t elements() const { return static_cast<size_t>(n) * n; }
};
