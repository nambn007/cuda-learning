#pragma once
// ============================================================
// reference.h - Tensor Cores via the WMMA API
// ============================================================
// A Tensor Core is a hardware unit that computes a small matrix
// multiply-accumulate in one instruction:
//
//     D = A * B + C      with A, B 16x16 half, C, D 16x16 float
//
// One instruction, 16*16*16 * 2 = 8192 FLOPs. That is why a card
// with Tensor Cores quotes an FP16 number many times its FP32 one.
//
// The WMMA API (nvcuda::wmma) exposes them at WARP level - the
// whole warp cooperates on one 16x16x16 tile, and the fragment
// layout inside the registers is deliberately unspecified:
//
//     wmma::fragment<matrix_a, 16,16,16, half, row_major> a;
//     wmma::fragment<matrix_b, 16,16,16, half, col_major> b;
//     wmma::fragment<accumulator, 16,16,16, float>        c;
//
//     wmma::fill_fragment(c, 0.0f);
//     wmma::load_matrix_sync(a, ptrA, lda);
//     wmma::load_matrix_sync(b, ptrB, ldb);
//     wmma::mma_sync(c, a, b, c);
//     wmma::store_matrix_sync(ptrC, c, ldc, wmma::mem_row_major);
//
// Three constraints worth knowing before you start:
//   - requires compute capability 7.0 or newer
//   - the whole warp must call every wmma function, uniformly
//   - pointers must be 256-bit aligned and the leading dimension a
//     multiple of 16 for half
//
// PRECISION. Inputs are FP16 (about 3 decimal digits), the
// accumulator is FP32. For a GEMM that is usually fine because the
// accumulation dominates the error - but it is a real change, not a
// free speedup, and this exercise measures the error rather than
// waving at it.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kSize = 1024;  // multiple of 16
inline constexpr int kWmmaTile = 16;

inline double gemmFlops(int n) {
    double N = static_cast<double>(n);
    return 2.0 * N * N * N;
}
