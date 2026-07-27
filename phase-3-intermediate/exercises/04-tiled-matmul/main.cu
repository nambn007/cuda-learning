// ============================================================
// 04 - Tiled matrix multiplication  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: the naive baseline (from Phase 2 exercise 07)
// ------------------------------------------------------------
__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: tiled, one output per thread
// ------------------------------------------------------------
// This is Phase 1 exercise 02's cache blocking, with __shared__
// playing the part of the cache - except you place the data.
//
//   __shared__ float As[TILE][TILE], Bs[TILE][TILE];
//   for t in 0 .. n/TILE:
//       As[ty][tx] = A[row * n + t*TILE + tx];
//       Bs[ty][tx] = B[(t*TILE + ty) * n + col];
//       __syncthreads();
//       for k in 0..TILE:  acc += As[ty][k] * Bs[k][tx];
//       __syncthreads();          <- do NOT omit this one
//   C[row * n + col] = acc;
//
// BOTH barriers are necessary. Without the second, fast threads
// start loading the next tile while slow threads are still reading
// the current one. It usually still gives the right answer on small
// inputs, which is what makes it dangerous.
//
// Before writing it, check the bank behaviour of the inner loop for
// a warp (tx varies, ty fixed): what does As[ty][k] look like
// across the 32 threads? What about Bs[k][tx]? Does this kernel
// need the padding trick from exercise 03?
template <int TILE>
__global__ void matmulTiled(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: tiled + register tiling, TM outputs per thread
// ------------------------------------------------------------
// Each thread keeps TM accumulators in registers and computes TM
// output rows:
//
//   float acc[TM] = {0};
//   ...
//   for k in 0..TILE:
//       float b = Bs[k][tx];             // read once
//       for i in 0..TM: acc[i] += As[ty*TM + i][k] * b;   // used TM times
//
// The block becomes TILE x (TILE/TM) threads, and each thread loads
// TM elements of each tile so they still get filled exactly once.
//
// Predict before measuring: plain tiling cuts global loads per
// output by 32x. How much speedup do you expect from it? And how
// much from register tiling, which does NOT reduce global traffic
// at all?
//
// One of those two predictions is probably badly wrong. Finding out
// which is the exercise.
template <int TILE, int TM>
__global__ void matmulTiledRegister(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 3 / 04 - Tiled matrix multiplication (starter)");
    requireCudaDevice();

    const int n = kSize;
    printf("  C = A * B, n = %d (%.2f GFLOP)\n", n, matmulFlops(n) / 1e9);
    printf("  Building the CPU reference...\n");

    MatmulProblem problem(n);

    // --------------------------------------------------------
    // TODO 4: allocate, run all three, verify with rtol 1e-3
    // --------------------------------------------------------
    // Suggested launches:
    //   naive    : block(16,16),  grid(n/16, n/16)
    //   tiled    : block(32,32),  grid(n/32, n/32)     1024 threads
    //   register : block(32,8),   grid(n/32, n/32)      256 threads
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: measure, and report each variant's arithmetic
    //         intensity and percentage of the GPU's FP32 peak
    // --------------------------------------------------------
    // reference.h has naiveLoadsPerOutput(), tiledLoadsPerOutput(),
    // naiveIntensity() and tiledIntensity(). Peak FP32 comes from
    // coresPerSM() x multiProcessorCount x clockRate x 2.
    //
    // Then explain the gap between what the intensity numbers
    // predict and what you measured.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the three kernels and the analysis");
    return verifySummary();
}
