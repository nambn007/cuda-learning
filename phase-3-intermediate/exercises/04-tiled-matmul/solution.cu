// ============================================================
// 04 - Tiled matrix multiplication  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// Baseline: the naive kernel from Phase 2 exercise 07
// ------------------------------------------------------------
__global__ void matmulNaive(const float* A, const float* B, float* C, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += A[static_cast<size_t>(row) * n + k] * B[static_cast<size_t>(k) * n + col];
        C[static_cast<size_t>(row) * n + col] = acc;
    }
}

// ------------------------------------------------------------
// Tiled: one output per thread
// ------------------------------------------------------------
// The loop structure is Phase 1 exercise 02's cache blocking, with
// __shared__ playing the part of the cache - except that here you
// place the data yourself and the win is real.
//
// Each iteration:
//   1. every thread loads one element of the A tile and one of B
//   2. barrier, so the whole tile is visible
//   3. every thread does TILE multiply-adds out of shared memory
//   4. barrier, so nobody overwrites the tile while others read it
//
// Both barriers are necessary. Dropping the second one is a classic
// bug: fast threads start loading the next tile while slow threads
// are still reading the current one.
//
// Bank behaviour in the inner loop, for a warp (tx varies, ty fixed):
//   As[ty][k]  -> same address for all 32 threads -> broadcast, free
//   Bs[k][tx]  -> 32 consecutive words -> 32 distinct banks, free
// So no padding is needed here. It IS needed in exercise 05.
template <int TILE>
__global__ void matmulTiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < n / TILE; ++t) {
        // Both loads are coalesced: consecutive tx reads consecutive
        // addresses in both A and B.
        As[ty][tx] = A[static_cast<size_t>(row) * n + t * TILE + tx];
        Bs[ty][tx] = B[static_cast<size_t>(t * TILE + ty) * n + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[static_cast<size_t>(row) * n + col] = acc;
}

// ------------------------------------------------------------
// Tiled + register tiling: TM outputs per thread
// ------------------------------------------------------------
// Shared memory removed the global-memory bottleneck; now SHARED
// memory traffic is the limit. The fix is the same idea one level
// down: each value a thread reads from shared memory should serve
// several multiply-adds.
//
// A thread holds TM accumulators in registers and computes TM
// output rows. Bs[k][tx] is read once and used TM times, so shared
// reads per FLOP drop by roughly half.
//
// Block is TILE x (TILE/TM) threads; each loads TM elements of each
// tile so the tiles still get filled exactly once.
template <int TILE, int TM>
__global__ void matmulTiledRegister(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;  // 0 .. TILE-1
    const int ty = threadIdx.y;  // 0 .. TILE/TM-1
    const int col = blockIdx.x * TILE + tx;
    const int rowBase = blockIdx.y * TILE + ty * TM;

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
            // Read once from shared, reuse TM times from a register.
            const float b = Bs[k][tx];
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[i] += As[ty * TM + i][k] * b;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
        C[static_cast<size_t>(rowBase + i) * n + col] = acc[i];
}

int main() {
    printBanner("Phase 3 / 04 - Tiled matrix multiplication");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int n = kSize;
    printf("  C = A * B, n = %d (%.2f GFLOP)\n", n, matmulFlops(n) / 1e9);
    printf("  Building the CPU reference...\n");

    MatmulProblem problem(n);
    std::vector<float> result(problem.elements());
    const size_t bytes = problem.bytes();

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, problem.A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, problem.B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 blockNaive(16, 16);
    const dim3 gridNaive(ceilDiv(n, 16), ceilDiv(n, 16));

    const dim3 blockTiled(kTile, kTile);          // 1024 threads
    const dim3 gridTiled(n / kTile, n / kTile);

    const dim3 blockReg(kTile, kTile / kThreadTile);  // 32 x 8 = 256 threads
    const dim3 gridReg(n / kTile, n / kTile);

    auto run = [&](const char* label, auto launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(result.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        // Summing 1024 terms with FMA drifts from the host's separate
        // multiply and add, so a loose tolerance is correct here.
        checkArray(label, result, problem.golden, 1e-3, 1e-3);
    };

    printSection("Correctness");
    run("naive", [&] { matmulNaive<<<gridNaive, blockNaive>>>(d_A, d_B, d_C, n); });
    run("tiled (32x32, 1 output/thread)",
        [&] { matmulTiled<kTile><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, n); });
    run("tiled + register (4 outputs/thread)", [&] {
        matmulTiledRegister<kTile, kThreadTile><<<gridReg, blockReg>>>(d_A, d_B, d_C, n);
    });

    printSection("Performance");
    double msNaive =
        timeGpuMs(10, [&] { matmulNaive<<<gridNaive, blockNaive>>>(d_A, d_B, d_C, n); });
    double msTiled = timeGpuMs(
        10, [&] { matmulTiled<kTile><<<gridTiled, blockTiled>>>(d_A, d_B, d_C, n); });
    double msReg = timeGpuMs(10, [&] {
        matmulTiledRegister<kTile, kThreadTile><<<gridReg, blockReg>>>(d_A, d_B, d_C, n);
    });
    double msCpu = timeCpuMs(1, [&] {
        matmulCPU(problem.A.data(), problem.B.data(), result.data(), n);
    }, 0);

    double peakGflops = 0.0;
#if CUDART_VERSION < 13000
    peakGflops = 2.0 * coresPerSM(prop.major, prop.minor) * prop.multiProcessorCount *
                 prop.clockRate * 1e3 / 1e9;
#endif

    ResultTable table;
    table.add("CPU ikj (1 thread)", msCpu, 0.0, matmulFlops(n));
    table.add("GPU naive", msNaive, 0.0, matmulFlops(n));
    table.add("GPU tiled (shared)", msTiled, 0.0, matmulFlops(n));
    table.add("GPU tiled + register", msReg, 0.0, matmulFlops(n));
    table.print("Matrix multiplication, 1024 x 1024");

    printSection("Why it worked");
    printf("  %-30s %14s %16s %12s\n", "variant", "loads/output", "FLOP per byte",
           "%% of peak");
    printf("  --------------------------------------------------------------------------\n");
    printf("  %-30s %14.0f %16.2f %11.0f%%\n", "naive", naiveLoadsPerOutput(n),
           naiveIntensity(),
           peakGflops > 0 ? 100.0 * gflops(matmulFlops(n), msNaive) / peakGflops : 0.0);
    printf("  %-30s %14.0f %16.2f %11.0f%%\n", "tiled (TILE=32)",
           tiledLoadsPerOutput(n, kTile), tiledIntensity(kTile),
           peakGflops > 0 ? 100.0 * gflops(matmulFlops(n), msTiled) / peakGflops : 0.0);
    printf("  %-30s %14.0f %16.2f %11.0f%%\n", "tiled + register (TM=4)",
           tiledLoadsPerOutput(n, kTile), tiledIntensity(kTile),
           peakGflops > 0 ? 100.0 * gflops(matmulFlops(n), msReg) / peakGflops : 0.0);

    printf("\n  On paper, tiling raised the arithmetic intensity from %.2f to %.0f\n",
           naiveIntensity(), tiledIntensity(kTile));
    printf("  FLOP/byte - a %.0fx improvement - by making every value loaded from\n",
           tiledIntensity(kTile) / naiveIntensity());
    printf("  global memory serve %d threads instead of 1.\n", kTile);

    printf("\n  ---------------------------------------------------------------\n");
    printf("  AND YET plain tiling only bought %.2fx.\n", msNaive / msTiled);
    printf("  ---------------------------------------------------------------\n");
    printf("\n  This is the most important result in the exercise, and it is not the\n");
    printf("  one the textbook prepares you for. The reason is the same one you met\n");
    printf("  in Phase 2 exercise 07: the naive kernel's 2048 loads per output were\n");
    printf("  never really going to DRAM. L1 and L2 were already capturing almost\n");
    printf("  all of that reuse. Writing the tiling by hand mostly replaced an\n");
    printf("  automatic cache with a manual one.\n");

    printf("\n  Register tiling is where the real win is (%.2fx over naive), and it\n",
           msNaive / msReg);
    printf("  is a DIFFERENT kind of change. It does not move less data - the\n");
    printf("  loads/output column above is identical. What it does is give each\n");
    printf("  thread more independent work:\n");
    printf("    - each value read from SHARED memory now serves %d multiply-adds\n",
           kThreadTile);
    printf("    - %d accumulators live in registers, so the FMA pipeline has %d\n",
           kThreadTile, kThreadTile);
    printf("      independent chains and stops stalling on its own latency\n");
    printf("      (the same effect you measured on the CPU in Phase 1 exercise 07)\n");
    printf("    - the block shrinks from %d to %d threads, freeing registers and\n",
           kTile * kTile, kTile * (kTile / kThreadTile));
    printf("      scheduler slots\n");

    printf("\n  The lesson to carry forward: on a modern GPU, shared memory is rarely\n");
    printf("  valuable because it is 'faster than global' - the cache already gave\n");
    printf("  you that. It is valuable because it lets you RESTRUCTURE the\n");
    printf("  computation so each thread does more work per byte it touches. The\n");
    printf("  restructuring is the point; the shared memory is just what makes it\n");
    printf("  possible.\n");

    printSection("How far is this from the real thing?");
    printf("  Speedup over naive        : %.2fx\n", msNaive / msReg);
    printf("  Speedup over one CPU core : %.0fx\n", msCpu / msReg);
    if (peakGflops > 0.0) {
        const double achieved = gflops(matmulFlops(n), msReg);
        printf("  Fraction of FP32 peak     : %.0f%%\n", 100.0 * achieved / peakGflops);
        printf("\n  cuBLAS reaches 80-90%% on this problem. The remaining gap is more\n");
        printf("  levels of the same idea: a second shared-memory stage, wider\n");
        printf("  register tiles (8x8 rather than 4x1), vectorised float4 loads,\n");
        printf("  double buffering so loads overlap with maths, and hand-tuned\n");
        printf("  parameters per architecture. Phase 5 exercise 03 measures the gap\n");
        printf("  directly.\n");
    }

    printSection("Two barriers, not one");
    printf("  The loop needs __syncthreads() BOTH after loading the tile and after\n");
    printf("  consuming it. Dropping the second is a classic bug: fast threads begin\n");
    printf("  loading the next tile while slow threads are still reading the current\n");
    printf("  one. It usually still produces the right answer on small inputs, which\n");
    printf("  is what makes it dangerous. compute-sanitizer --tool racecheck finds it.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    reportCheck("tiling beats naive", msTiled < msNaive);
    reportCheck("register tiling beats plain tiling", msReg < msTiled);
    return verifySummary();
}
