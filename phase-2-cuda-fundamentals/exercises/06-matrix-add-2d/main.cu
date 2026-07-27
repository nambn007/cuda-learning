// ============================================================
// 06 - 2D matrix addition, and the axis order  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: the sensible mapping - threadIdx.x indexes the COLUMN
// ------------------------------------------------------------
//     col = blockIdx.x * blockDim.x + threadIdx.x
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     i   = row * cols + col
// Check both bounds.
__global__ void matrixAddRowMajor(const float* a, const float* b, float* c, int rows,
                                  int cols) {
    (void)a;
    (void)b;
    (void)c;
    (void)rows;
    (void)cols;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the same kernel with x and y swapped
// ------------------------------------------------------------
//     row = blockIdx.x * blockDim.x + threadIdx.x     <- swapped
//     col = blockIdx.y * blockDim.y + threadIdx.y     <- swapped
// Everything else identical. It will produce exactly the same
// numbers. Your job is to find out what it costs.
__global__ void matrixAddColumnMajor(const float* a, const float* b, float* c, int rows,
                                     int cols) {
    (void)a;
    (void)b;
    (void)c;
    (void)rows;
    (void)cols;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: a flat 1D grid-stride version
// ------------------------------------------------------------
// Row-major storage is one contiguous array, so the 2D shape is
// optional. Write the flat version and compare it with the good
// 2D one - the result may surprise you.
__global__ void matrixAddFlat(const float* a, const float* b, float* c, size_t n) {
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 2 / 06 - 2D matrix addition (starter)");
    requireCudaDevice();

    const int rows = kRows, cols = kCols;
    MatrixProblem problem(rows, cols);
    printf("  Matrix: %d x %d (%.0f MB each)\n", rows, cols,
           problem.bytes() / (1024.0 * 1024.0));

    // --------------------------------------------------------
    // TODO 4: allocate, copy, launch all three, verify each
    // --------------------------------------------------------
    // Suggested launch:
    //     dim3 block(16, 16);
    //     dim3 grid(ceilDiv(cols, block.x), ceilDiv(rows, block.y));
    //
    // Careful: the SWAPPED kernel needs its grid swapped too, or it
    // will not cover the whole matrix.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: time all three and explain the gap
    // --------------------------------------------------------
    // Report GB/s for each using matrixAddBytes(rows, cols), and as
    // a percentage of theoreticalBandwidthGBs().
    //
    // Before you run it, predict: how much slower will the swapped
    // version be? Write your guess down, then measure.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the three kernels and the measurements in main.cu");
    return verifySummary();
}
