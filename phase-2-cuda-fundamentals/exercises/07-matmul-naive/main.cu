// ============================================================
// 07 - Naive matrix multiplication on the GPU  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: one thread per output element
// ------------------------------------------------------------
//     col = blockIdx.x * blockDim.x + threadIdx.x
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     if (row < n && col < n):
//         acc = 0
//         for k in 0..n:  acc += A[row*n + k] * B[k*n + col]
//         C[row*n + col] = acc
//
// Accumulate into a local `float acc`, exactly as on the CPU in
// Phase 1 exercise 01 - writing to C inside the k loop would cost
// a global store per iteration.
//
// Before you write it, work out the access pattern within a warp
// (threads differing in `col`):
//   - what does A[row*n + k] look like across the 32 threads?
//   - what does B[k*n + col] look like?
// One is a broadcast, the other is coalesced. Neither is the
// problem with this kernel - that will become clear from the
// numbers.
__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the same kernel with row and col swapped
// ------------------------------------------------------------
// Only the first two lines change. Time both.
__global__ void matmulNaiveSwapped(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 2 / 07 - Naive matrix multiplication (starter)");
    requireCudaDevice();

    const int n = kSize;
    printf("  C = A * B with n = %d (%.2f GFLOP of work)\n", n, matmulFlops(n) / 1e9);
    printf("  Building the CPU reference (a few seconds)...\n");

    MatmulProblem problem(n);
    std::vector<float> result(problem.elements());

    // --------------------------------------------------------
    // TODO 3: allocate, copy, launch, verify
    // --------------------------------------------------------
    // Suggested launch: dim3 block(16, 16), grid covering n x n.
    // Use a tolerance of 1e-4: summing 1024 terms with FMA drifts
    // measurably from the CPU's separate multiply and add.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: measure, then work out WHY it is slow
    // --------------------------------------------------------
    // Compute and print:
    //   - achieved GFLOP/s
    //   - naiveTrafficBytes(n): what this kernel actually moves
    //   - minimumTrafficBytes(n): what the algorithm needs
    //   - the ratio between them
    //   - effective bandwidth, and its percentage of peak
    //
    // Then answer: this kernel is coalesced, and it still reaches
    // only a small fraction of the GPU's peak GFLOP/s. Why?
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement matmulNaive(), the plumbing, and the analysis");
    return verifySummary();
}
