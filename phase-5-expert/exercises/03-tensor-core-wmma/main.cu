// ============================================================
// 03 - Tensor Cores via WMMA  [STARTER]
// ============================================================

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

using namespace nvcuda;

// ------------------------------------------------------------
// TODO 1: an FP32 baseline
// ------------------------------------------------------------
// One thread per output element, as in Phase 2 exercise 07.
__global__ void gemmFp32(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the Tensor Core version
// ------------------------------------------------------------
// One WARP computes one 16x16 output tile:
//
//   wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
//   wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b;
//   wmma::fragment<wmma::accumulator, 16,16,16, float>              c;
//
//   wmma::fill_fragment(c, 0.0f);
//   for (int k = 0; k < n; k += 16) {
//       wmma::load_matrix_sync(a, A + row*n + k, n);
//       wmma::load_matrix_sync(b, B + k*n + col, n);
//       wmma::mma_sync(c, a, b, c);
//   }
//   wmma::store_matrix_sync(C + row*n + col, c, n, wmma::mem_row_major);
//
// Three rules:
//   - the WHOLE WARP must call every wmma function, uniformly.
//     A divergent call is undefined behaviour.
//   - never index into a fragment. Its register layout is
//     deliberately unspecified.
//   - pointers need 256-bit alignment, and the leading dimension
//     must be a multiple of 16 for half.
//
// Launch with blockDim (32, 4): 32 threads in x is exactly one
// warp, and you get four warps per block.
__global__ void gemmWmma(const half* A, const half* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 5 / 03 - Tensor Cores via WMMA (starter)");
    requireCudaDevice();

    // Tensor Cores need sm_70+. Skip cleanly rather than failing -
    // a GTX 16xx card has none at all.
    if (!hasComputeCapability(7, 0)) {
        return skipExercise("Tensor Cores need compute capability 7.0 or newer");
    }

    printf("  n = %d, %.1f GFLOP per GEMM\n", kSize, gemmFlops(kSize) / 1e9);

    // --------------------------------------------------------
    // TODO 3: build both, convert the inputs to half, verify
    // --------------------------------------------------------
    // Use __float2half() for the conversion - and note that the
    // conversion ITSELF is lossy: FP16 holds about 3 decimal digits.
    //
    // Compare with a RELATIVE tolerance around 2%. If you find
    // yourself widening it further, stop and think about whether
    // FP16 is appropriate for your data rather than loosening the
    // test until it passes.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: measure, and report GFLOP/s for both
    // --------------------------------------------------------
    // Predict first: one mma_sync does 16*16*16*2 = 8192 FLOPs in a
    // single instruction. How much faster should that be?
    //
    // Whatever you measure, this kernel is a teaching version: it
    // reads A and B straight from global memory with no shared
    // staging, so most of the Tensor Cores' throughput goes on
    // memory stalls. Stage the tiles as in Phase 3 exercise 04 and
    // see how much further it goes.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the FP32 baseline and the WMMA kernel");
    return verifySummary();
}
