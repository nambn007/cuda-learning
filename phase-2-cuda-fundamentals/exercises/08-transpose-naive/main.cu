// ============================================================
// 08 - Naive transpose: the kernel you cannot coalesce  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: a plain copy - the speed of light for this problem
// ------------------------------------------------------------
// A copy moves exactly the same bytes as a transpose: one read and
// one write per element, no arithmetic. Everything else in this
// exercise is measured as a fraction of this kernel.
__global__ void copyKernel(const float* in, float* out, int rows, int cols) {
    (void)in;
    (void)out;
    (void)rows;
    (void)cols;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: transpose with the READ coalesced
// ------------------------------------------------------------
//     col = blockIdx.x * blockDim.x + threadIdx.x
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     out[col * rows + row] = in[row * cols + col];
//
// Work out, before you run it, what each side looks like across a
// warp (threads varying `col`).
__global__ void transposeReadCoalesced(const float* in, float* out, int rows, int cols) {
    (void)in;
    (void)out;
    (void)rows;
    (void)cols;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: transpose with the WRITE coalesced
// ------------------------------------------------------------
// The mirror image: map threadIdx.x to `row` instead. The body is
// character-for-character identical; only the two index lines and
// the grid shape change.
__global__ void transposeWriteCoalesced(const float* in, float* out, int rows, int cols) {
    (void)in;
    (void)out;
    (void)rows;
    (void)cols;
    // TODO
}

int main() {
    printBanner("Phase 2 / 08 - Naive transpose (starter)");
    requireCudaDevice();

    const int rows = kRows, cols = kCols;
    TransposeProblem problem(rows, cols);
    printf("  Matrix: %d x %d (%.0f MB)\n", rows, cols,
           problem.bytes() / (1024.0 * 1024.0));

    // --------------------------------------------------------
    // TODO 4: allocate, copy in, run all three, verify the two
    //         transposes against problem.golden
    // --------------------------------------------------------
    // Suggested block: dim3(32, 8) = 256 threads. 32 wide so that a
    // warp spans exactly one row of the block.
    //
    // Careful: the two transposes need DIFFERENT grid shapes,
    // because they map x to different axes.
    //
    // The data is exact integers stored in floats, so compare with
    // rtol = atol = 0.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: measure all three, and report each transpose as a
    //         PERCENTAGE OF COPY BANDWIDTH
    // --------------------------------------------------------
    // GB/s on its own is not the interesting number here. Copy is
    // the best this problem could possibly go; the question is how
    // close a transpose gets.
    //
    // Then answer: can you rearrange the indices so that BOTH the
    // read and the write are coalesced? Try. Convince yourself of
    // the answer before reading the solution.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the three kernels and the comparison in main.cu");
    return verifySummary();
}
