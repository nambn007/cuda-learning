#pragma once
// ============================================================
// reference.h - cuBLAS, and the gap to it
// ============================================================
// Phase 3 exercise 04 got a tiled matmul to about 19% of the card's
// FP32 peak. This exercise measures the rest of the distance.
//
// THE COLUMN-MAJOR TRAP. cuBLAS inherits Fortran's convention:
// matrices are COLUMN-major, so element (i, j) lives at
// A[i + j * ld] rather than A[i * ld + j]. Passing a row-major
// matrix to cublasSgemm computes the transpose of what you wanted,
// silently.
//
// The standard trick avoids any actual transposition. For
// row-major A, B, C:
//
//     C = A * B          (row-major)
//   is the same memory as
//     C^T = B^T * A^T    (column-major)
//
// so you call cublasSgemm with the operands SWAPPED and tell it
// nothing is transposed. No data moves; only the argument order
// changes. Getting used to that is most of the work of using
// cuBLAS from C++.
//
// What the remaining gap is made of, roughly in order:
//   - wider register tiles (8x8 per thread, not 4x1)
//   - vectorised float4 loads into shared memory
//   - double buffering, so the next tile loads while this one
//     computes
//   - a second level of shared-memory staging
//   - per-architecture tuning of every tile size
//   - Tensor Cores, where the data type allows (exercise 03)
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kSize = 2048;

inline double gemmFlops(int n) {
    double N = static_cast<double>(n);
    return 2.0 * N * N * N;
}
