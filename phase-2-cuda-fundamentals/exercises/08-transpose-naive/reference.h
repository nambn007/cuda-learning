#pragma once
// ============================================================
// reference.h - transpose: the kernel you cannot coalesce
// ============================================================
// Transpose does no arithmetic at all. It reads N^2 floats and
// writes N^2 floats, and that is the entire kernel. So it should
// run at exactly the speed of a memory copy.
//
// It does not, and the reason is structural:
//
//   out[col][row] = in[row][col]
//
// A warp varies one index. If it varies `col`, the READ is
// contiguous but the WRITE lands on addresses `rows` apart. If it
// varies `row`, the write is contiguous and the read is strided.
// One of the two sides is always scattered, and no amount of
// index rearranging changes that - the transpose IS the scatter.
//
// This is the first kernel in the curriculum where being careful is
// not enough. The fix needs a staging area that both sides can be
// contiguous against: __shared__ memory, in Phase 3 exercise 05.
//
// The useful metric here is not GB/s in the abstract but GB/s
// RELATIVE TO A PLAIN COPY of the same data. Copy is the speed of
// light for this problem; transpose is measured as a fraction of it.
// ============================================================

#include <cstddef>
#include <vector>

#include "verify.h"

inline constexpr int kRows = 4096;
inline constexpr int kCols = 4096;

inline void transposeCPU(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
}

// One read and one write per element - the same for copy and for
// transpose, which is exactly why the comparison is fair.
inline double transposeBytes(int rows, int cols) {
    return static_cast<double>(rows) * cols * 2.0 * sizeof(float);
}

struct TransposeProblem {
    int rows, cols;
    std::vector<float> in, golden;

    TransposeProblem(int r, int c)
        : rows(r),
          cols(c),
          in(static_cast<size_t>(r) * c),
          golden(static_cast<size_t>(r) * c) {
        // A recognisable pattern: element (r, c) holds r * 10000 + c,
        // so a wrong transpose is obvious when printed.
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                in[static_cast<size_t>(row) * cols + col] =
                    static_cast<float>(row) * 10000.0f + static_cast<float>(col);
        transposeCPU(in.data(), golden.data(), rows, cols);
    }

    size_t elements() const { return static_cast<size_t>(rows) * cols; }
    size_t bytes() const { return elements() * sizeof(float); }
};
