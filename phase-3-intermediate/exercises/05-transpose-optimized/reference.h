#pragma once
// ============================================================
// reference.h - transpose with a shared-memory staging tile
// ============================================================
// Phase 2 exercise 08 established that a transpose cannot be
// coalesced by rearranging indices: out[col][row] = in[row][col]
// means whichever index a warp varies, one side is contiguous and
// the other is scattered. The scatter IS the transpose.
//
// The fix is the second use of shared memory (exercise 02):
// REORDERING. Shared memory has no coalescing requirement, so:
//
//   1. read a TILE x TILE block from global memory, COALESCED
//   2. write it into a shared tile
//   3. barrier
//   4. read the shared tile TRANSPOSED - free, no coalescing rules
//   5. write it out to global memory, COALESCED
//
// Both global accesses are now contiguous. The transpose happens
// entirely on chip.
//
// Except that step 4 walks a column of the shared tile, which is
// exactly the 32-way bank conflict from exercise 03. Without one
// padding column the shared-memory penalty replaces the global one
// and you have gained very little. With it, the transpose gets
// close to copy speed.
//
// This exercise is where exercises 02, 03 and Phase 2/08 meet.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kRows = 4096;
inline constexpr int kCols = 4096;
inline constexpr int kTile = 32;
inline constexpr int kBlockRows = 8;  // 32 x 8 = 256 threads, each doing 4 rows

inline void transposeCPU(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
}

// One read and one write per element - identical for copy and for
// transpose, which is what makes the comparison fair.
inline double transposeBytes(int rows, int cols) {
    return static_cast<double>(rows) * cols * 2.0 * sizeof(float);
}

struct TransposeProblem {
    int rows, cols;
    std::vector<float> in, golden;

    TransposeProblem(int r, int c)
        : rows(r), cols(c), in(static_cast<size_t>(r) * c), golden(static_cast<size_t>(r) * c) {
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                in[static_cast<size_t>(row) * cols + col] =
                    static_cast<float>(row) * 10000.0f + static_cast<float>(col);
        transposeCPU(in.data(), golden.data(), rows, cols);
    }

    size_t elements() const { return static_cast<size_t>(rows) * cols; }
    size_t bytes() const { return elements() * sizeof(float); }
};
