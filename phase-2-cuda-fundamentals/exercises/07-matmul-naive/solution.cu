// ============================================================
// 07 - Naive matrix multiplication on the GPU  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// One thread per output element
// ------------------------------------------------------------
// Thread (row, col) walks the whole of row `row` of A and the whole
// of column `col` of B. That is 2N global loads per output element,
// and N^2 threads doing it - so A and B are each read N times over.
//
// Note the access patterns within a warp (threads differing in
// threadIdx.x, so in `col`):
//   A[row*n + k]  every thread reads the SAME address -> broadcast,
//                 which the hardware handles well
//   B[k*n + col]  consecutive threads read consecutive addresses
//                 -> perfectly coalesced
//
// So this kernel is already coalesced. It is slow anyway, because
// it moves N times more data than it needs to. Coalescing decides
// how efficiently you move bytes; it cannot reduce how many bytes
// you move. That is what shared memory is for (Phase 3/04).
__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += A[static_cast<size_t>(row) * n + k] * B[static_cast<size_t>(k) * n + col];
        }
        C[static_cast<size_t>(row) * n + col] = acc;
    }
}

// The same kernel with the axes swapped, for comparison: now the B
// access is strided and the A access is coalesced, which is the
// worse trade.
__global__ void matmulNaiveSwapped(const float* A, const float* B, float* C, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += A[static_cast<size_t>(row) * n + k] * B[static_cast<size_t>(k) * n + col];
        }
        C[static_cast<size_t>(row) * n + col] = acc;
    }
}

int main() {
    printBanner("Phase 2 / 07 - Naive matrix multiplication");
    requireCudaDevice();

    const int n = kSize;
    printf("  C = A * B with n = %d\n", n);
    printf("  Work: %.2f GFLOP, minimum traffic: %.1f MB\n", matmulFlops(n) / 1e9,
           minimumTrafficBytes(n) / (1024.0 * 1024.0));
    printf("  Building the CPU reference (a few seconds)...\n");

    MatmulProblem problem(n);
    std::vector<float> result(problem.elements());
    const size_t bytes = problem.bytes();

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, problem.A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, problem.B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(ceilDiv(n, static_cast<int>(block.x)),
                    ceilDiv(n, static_cast<int>(block.y)));

    printSection("Correctness");
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // The GPU sums the k loop in the same order as the CPU, but
    // uses FMA, so the two differ in the last bits. Over 1024 terms
    // the error accumulates - hence a looser tolerance than the
    // element-wise kernels needed.
    checkArray("C == A * B", result, problem.golden, 1e-4, 1e-4);

    printSection("Performance");

    double msCpu = timeCpuMs(2, [&] {
        matmulCPU(problem.A.data(), problem.B.data(), result.data(), n);
    }, /*warmup=*/1);
    double msGpu = timeGpuMs(10, [&] { matmulNaive<<<grid, block>>>(d_A, d_B, d_C, n); });
    double msSwapped =
        timeGpuMs(10, [&] { matmulNaiveSwapped<<<grid, block>>>(d_A, d_B, d_C, n); });

    ResultTable table;
    table.add("CPU ikj (single thread)", msCpu, 0.0, matmulFlops(n));
    table.add("GPU naive (coalesced B)", msGpu, 0.0, matmulFlops(n));
    table.add("GPU naive (strided B)", msSwapped, 0.0, matmulFlops(n));
    table.print("Matrix multiplication, 1024 x 1024");

    // ========================================================
    // Why this is nowhere near peak
    // ========================================================
    printSection("Where the performance went");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const double achievedGflops = gflops(matmulFlops(n), msGpu);
    const double requested = naiveTrafficBytes(n);
    const double requestRate = gbPerSec(requested, msGpu);
    const double peakBw = theoreticalBandwidthGBs();

    double peakGflops = 0.0;
#if CUDART_VERSION < 13000
    peakGflops = 2.0 * coresPerSM(prop.major, prop.minor) * prop.multiProcessorCount *
                 prop.clockRate * 1e3 / 1e9;
#endif

    printf("  Achieved              : %.1f GFLOP/s", achievedGflops);
    if (peakGflops > 0.0) printf("  (%.0f%% of this GPU's FP32 peak)", 100.0 * achievedGflops / peakGflops);
    printf("\n");
    printf("  Bytes REQUESTED       : %.1f GB per call\n", requested / 1e9);
    printf("  Bytes actually needed : %.3f GB  (read A and B once, write C once)\n",
           minimumTrafficBytes(n) / 1e9);
    printf("  Ratio                 : %.0fx more loads than the algorithm requires\n",
           requested / minimumTrafficBytes(n));

    printf("\n  Load rate             : %.0f GB/s", requestRate);
    if (peakBw > 0.0) printf("  = %.1fx the DRAM peak of %.0f GB/s", requestRate / peakBw, peakBw);
    printf("\n");
    printf("  That figure is impossible for DRAM, and that is the point: the L1 and\n");
    printf("  L2 caches served most of those loads. The kernel is NOT limited by\n");
    printf("  DRAM bandwidth - it is limited by how fast the memory pipeline can\n");
    printf("  issue and service %.0f GB of load REQUESTS.\n", requested / 1e9);

    printf("\n  Arithmetic intensity of THIS KERNEL\n");
    printf("    Each output element costs 2N FLOPs and issues 2N global loads:\n");
    printf("      intensity = 2N / (2N * 4 bytes) = 0.25 FLOP per byte loaded\n");
    printf("    and it stays 0.25 however large n gets.\n");
    printf("\n    The ALGORITHM is not memory bound - it does N^3 work on N^2 data,\n");
    printf("    so its intrinsic intensity grows with N. This KERNEL is, because\n");
    printf("    every thread re-reads its row of A and column of B instead of\n");
    printf("    sharing them with the 255 other threads in its block that need the\n");
    printf("    very same values.\n");

    printf("\n  Note the kernel is already COALESCED\n");
    printf("    Within a warp, threads differ in `col`:\n");
    printf("      A[row*n + k]  -> same address for all 32 -> broadcast, fine\n");
    printf("      B[k*n + col]  -> 32 consecutive addresses -> coalesced, fine\n");
    printf("    Swapping the axes breaks that and costs %.1fx (row above).\n",
           msSwapped / msGpu);
    printf("    But even the coalesced version reaches only %.0f%% of peak FLOP/s,\n",
           peakGflops > 0.0 ? 100.0 * achievedGflops / peakGflops : 0.0);
    printf("    because coalescing decides how EFFICIENTLY a request is served -\n");
    printf("    it cannot reduce HOW MANY requests you make.\n");

    printf("\n  What comes next\n");
    printf("    Phase 3 exercise 04 stages tiles in __shared__ memory, so each\n");
    printf("    value fetched from global memory is used by 16 threads instead of\n");
    printf("    1. That is the difference between 'move bytes faster' and 'move\n");
    printf("    fewer bytes', and it is worth several times this kernel's speed.\n");
    printf("    Phase 5 exercise 03 then compares both against cuBLAS.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    reportCheck("GPU beats the single-threaded CPU", msGpu < msCpu);
    reportCheck("coalesced B beats strided B", msGpu < msSwapped);
    return verifySummary();
}
