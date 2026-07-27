// ============================================================
// 02 - cuBLAS GEMM, and the gap to it  [SOLUTION]
// ============================================================
// NOTE: not yet run on the reference RTX 3060. Builds and reviewed;
// the README carries no measured numbers.
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

// The best hand-written kernel from Phase 3 exercise 04: shared
// tiles plus a 4-deep register tile.
template <int TILE, int TM>
__global__ void matmulTiledRegister(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * TILE + tx;

    float acc[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) acc[i] = 0.0f;

    for (int t = 0; t < n / TILE; ++t) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int r = ty * TM + i;
            As[r][tx] = A[static_cast<size_t>(blockIdx.y * TILE + r) * n + t * TILE + tx];
            Bs[r][tx] = B[static_cast<size_t>(t * TILE + r) * n + col];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const float b = Bs[k][tx];
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[i] += As[ty * TM + i][k] * b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
        C[static_cast<size_t>(blockIdx.y * TILE + ty * TM + i) * n + col] = acc[i];
}

int main() {
    printBanner("Phase 5 / 02 - cuBLAS GEMM");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int n = kSize;
    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    printf("  C = A * B, n = %d (%.1f GFLOP per call)\n", n, gemmFlops(n) / 1e9);

    std::vector<float> hA(elems), hB(elems);
    fillRandom(hA.data(), elems, -1.0f, 1.0f, 161);
    fillRandom(hB.data(), elems, -1.0f, 1.0f, 162);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_Cref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_Cref, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // ========================================================
    // The column-major trick
    // ========================================================
    // Our A, B and C are ROW-major. cuBLAS reads COLUMN-major.
    //
    //   C = A * B          (row-major)
    // is the same bytes as
    //   C^T = B^T * A^T    (column-major)
    //
    // so we pass B first and A second, claim nothing is transposed,
    // and cuBLAS writes exactly the row-major C we wanted. No data
    // is moved - only the argument order changes.
    auto runCublas = [&](float* out) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n,         // rows of the column-major result = our cols
                                 n,         // cols of the column-major result = our rows
                                 n,         // shared dimension
                                 &alpha,
                                 d_B, n,    // "A" for cuBLAS is our B
                                 d_A, n,    // "B" for cuBLAS is our A
                                 &beta,
                                 out, n));
    };

    printSection("Correctness");
    runCublas(d_Cref);
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 blockReg(32, 8);
    const dim3 gridReg(n / 32, n / 32);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    matmulTiledRegister<32, 4><<<gridReg, blockReg>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_LAST();

    {
        std::vector<float> mine(elems), ref(elems);
        CUDA_CHECK(cudaMemcpy(mine.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ref.data(), d_Cref, bytes, cudaMemcpyDeviceToHost));
        // Both sum 2048 terms in different orders, so a loose
        // tolerance is correct. cuBLAS is the reference here -
        // there is no CPU golden at this size.
        checkArray("hand-written matches cuBLAS", mine, ref, 1e-2, 1e-2);
    }

    printSection("Performance");
    double msMine =
        timeGpuMs(10, [&] { matmulTiledRegister<32, 4><<<gridReg, blockReg>>>(d_A, d_B, d_C, n); });
    double msCublas = timeGpuMs(10, [&] { runCublas(d_Cref); });

    double peakGflops = 0.0;
#if CUDART_VERSION < 13000
    peakGflops = 2.0 * coresPerSM(prop.major, prop.minor) * prop.multiProcessorCount *
                 prop.clockRate * 1e3 / 1e9;
#endif

    ResultTable table;
    table.add("hand-written (Phase 3/04)", msMine, 0.0, gemmFlops(n));
    table.add("cublasSgemm", msCublas, 0.0, gemmFlops(n));
    table.print("SGEMM, 2048 x 2048");

    if (peakGflops > 0.0) {
        printf("\n  %-28s %8.0f GFLOP/s  %3.0f%% of FP32 peak\n", "hand-written",
               gflops(gemmFlops(n), msMine), 100.0 * gflops(gemmFlops(n), msMine) / peakGflops);
        printf("  %-28s %8.0f GFLOP/s  %3.0f%% of FP32 peak\n", "cuBLAS",
               gflops(gemmFlops(n), msCublas),
               100.0 * gflops(gemmFlops(n), msCublas) / peakGflops);
        printf("\n  Your kernel reaches %.0f%% of cuBLAS.\n", 100.0 * msCublas / msMine);
    }

    printSection("The column-major trap");
    printf("  cuBLAS inherits Fortran's convention: element (i, j) is at\n");
    printf("  A[i + j*ld], not A[i*ld + j]. Hand it a row-major matrix and it\n");
    printf("  computes the transpose of what you wanted - silently, with no error.\n");
    printf("\n  The trick used above moves no data at all:\n");
    printf("      C   = A * B      (row-major)\n");
    printf("    is the same bytes as\n");
    printf("      C^T = B^T * A^T  (column-major)\n");
    printf("  so pass B first, A second, claim no transposition, and cuBLAS writes\n");
    printf("  exactly the row-major C you wanted. Only the argument order changes.\n");

    printSection("What the remaining gap is made of");
    printf("  Roughly in order of what it would take to close it:\n");
    printf("    - wider register tiles (8x8 per thread, not 4x1)\n");
    printf("    - vectorised float4 loads into shared memory\n");
    printf("    - double buffering, so the next tile loads while this one computes\n");
    printf("    - a second level of shared-memory staging\n");
    printf("    - per-architecture tuning of every tile size\n");
    printf("    - Tensor Cores where the data type allows (exercise 03)\n");
    printf("\n  Reaching 80%% of cuBLAS by hand is an excellent result and a\n");
    printf("  multi-week project. Reaching 100%% is somebody's full-time job.\n");
    printf("  Knowing the number is what lets you decide whether to try.\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_Cref));

    reportCheck("cuBLAS is at least as fast as the hand-written kernel",
                msCublas <= msMine * 1.05);
    return verifySummary();
}
