// ============================================================
// 02 - cuBLAS GEMM, and the gap to it  [STARTER]
// ============================================================

#include <cublas_v2.h>

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

#define CUBLAS_CHECK(call)                                                         \
    do {                                                                           \
        cublasStatus_t st_ = (call);                                               \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS error at %s:%d -> %d\n", __FILE__, __LINE__,   \
                    static_cast<int>(st_));                                        \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// ------------------------------------------------------------
// TODO 1: bring across your best matmul from Phase 3 exercise 04
// ------------------------------------------------------------
// The tiled + register version. The comparison is only meaningful
// against a real effort.
template <int TILE, int TM>
__global__ void matmulTiledRegister(const float* A, const float* B, float* C, int n) {
    (void)A;
    (void)B;
    (void)C;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 5 / 02 - cuBLAS GEMM (starter)");
    requireCudaDevice();

    const int n = kSize;
    printf("  C = A * B, n = %d (%.1f GFLOP per call)\n", n, gemmFlops(n) / 1e9);

    // --------------------------------------------------------
    // TODO 2: call cublasSgemm on ROW-major data
    // --------------------------------------------------------
    // cuBLAS is COLUMN-major: element (i, j) is at A[i + j*ld].
    // Hand it a row-major matrix and it computes the transpose of
    // what you wanted - silently.
    //
    // Do NOT transpose anything. Use the identity:
    //
    //     C   = A * B      (row-major)
    //   is the same bytes as
    //     C^T = B^T * A^T  (column-major)
    //
    // so pass B first and A second, with CUBLAS_OP_N for both.
    // Work out for yourself what each of m, n, k and the three ld
    // values must be. Getting this right the first time is unusual;
    // getting it right by reasoning rather than by permutation is
    // the exercise.
    //
    // Sanity check: if your result looks like the transpose of the
    // expected answer, you have swapped exactly one thing.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: compare against your own kernel
    // --------------------------------------------------------
    // Report both as GFLOP/s and as a percentage of the card's FP32
    // peak, then as a percentage of cuBLAS.
    //
    // Use cuBLAS as the reference for correctness - there is no CPU
    // golden at n = 2048. Both sum 2048 terms in different orders,
    // so compare with a loose tolerance.
    printSection("Performance");
    printf("  TODO\n");

    printTodoNotice("implement the kernel and the cuBLAS call");
    return verifySummary();
}
