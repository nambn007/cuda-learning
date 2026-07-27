// ============================================================
// 05 - Transpose with a shared-memory tile  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: a plain copy - the speed of light for this problem
// ------------------------------------------------------------
// Block is kTile x kBlockRows (32 x 8 = 256 threads), and each
// thread handles kTile/kBlockRows = 4 rows:
//     for (int j = 0; j < kTile; j += kBlockRows) { ... y + j ... }
__global__ void copyKernel(const float* in, float* out, int width, int height) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the naive transpose (Phase 2 exercise 08)
// ------------------------------------------------------------
__global__ void transposeNaive(const float* in, float* out, int width, int height) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: stage the tile in shared memory
// ------------------------------------------------------------
//   1. read a tile from global, COALESCED, into shared
//        tile[threadIdx.y + j][threadIdx.x] = in[(y+j)*width + x]
//   2. __syncthreads()
//   3. recompute x and y for the TRANSPOSED block position:
//        x = blockIdx.y * kTile + threadIdx.x;
//        y = blockIdx.x * kTile + threadIdx.y;
//   4. write out COALESCED, reading the tile transposed:
//        out[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y + j]
//
// Step 3 is the part people get wrong. The output tile lives at the
// transposed position in the GRID, not just inside the tile.
//
// Before writing it: work out the bank of tile[threadIdx.x][ty + j]
// across a warp. What conflict degree do you get?
__global__ void transposeSharedNoPad(const float* in, float* out, int width, int height) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: the same, with one padding column
// ------------------------------------------------------------
//     __shared__ float tile[kTile][kTile + 1];
//
// Predict how much this is worth BEFORE measuring. In exercise 03
// the identical 32-way conflict cost 15x. Do you expect the same
// here? If not, what is different about this kernel?
__global__ void transposeSharedPadded(const float* in, float* out, int width, int height) {
    (void)in;
    (void)out;
    (void)width;
    (void)height;
    // TODO
}

int main() {
    printBanner("Phase 3 / 05 - Transpose with shared memory (starter)");
    requireCudaDevice();

    TransposeProblem problem(kRows, kCols);
    printf("  Matrix: %d x %d (%.0f MB), tile %dx%d, block %dx%d\n", kRows, kCols,
           problem.bytes() / (1024.0 * 1024.0), kTile, kTile, kTile, kBlockRows);

    // --------------------------------------------------------
    // TODO 5: allocate, run all four, verify (rtol = atol = 0)
    // --------------------------------------------------------
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 6: measure everything as a PERCENTAGE OF COPY SPEED
    // --------------------------------------------------------
    // Then separate the two effects:
    //     naive -> shared            (what did coalescing buy?)
    //     shared -> shared padded    (what did the padding buy?)
    //
    // Explain any gap between the padding's cost here and its cost
    // in exercise 03. The answer is about what each kernel is
    // BOUND BY, and it is the most transferable thing in this
    // exercise.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the four kernels and the comparison");
    return verifySummary();
}
