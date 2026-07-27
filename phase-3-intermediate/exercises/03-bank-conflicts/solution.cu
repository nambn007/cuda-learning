// ============================================================
// 03 - Shared memory bank conflicts  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// Experiment 1: stride through shared memory
// ------------------------------------------------------------
// Thread `tid` starts at (tid * stride) and every iteration all
// threads advance by 1 together, so the OFFSETS BETWEEN threads -
// and therefore the bank pattern - stay the same throughout.
//
// The array is a power of two so the wrap-around is a mask rather
// than a modulo, keeping integer work out of the measurement.
__global__ void sharedStride(float* out, int stride, int iters) {
    __shared__ float s[kSharedWords];

    const int tid = threadIdx.x;
    for (int i = tid; i < kSharedWords; i += blockDim.x) s[i] = static_cast<float>(i) * 0.5f;
    __syncthreads();

    int idx = (tid * stride) & (kSharedWords - 1);
    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) {
        acc += s[idx];
        idx = (idx + 1) & (kSharedWords - 1);
    }
    out[blockIdx.x * blockDim.x + tid] = acc;
}

// ------------------------------------------------------------
// Experiment 2: the 2D tile, the classic trap and the classic fix
// ------------------------------------------------------------
// Row access: tile[i][tid] -> address i*32 + tid -> bank tid % 32.
// All 32 banks, no conflict.
__global__ void tileRowAccess(float* out, int iters) {
    __shared__ float tile[kTile][kTile];

    const int tid = threadIdx.x;
    for (int r = 0; r < kTile; ++r) tile[r][tid] = static_cast<float>(r * kTile + tid);
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) acc += tile[k & (kTile - 1)][tid];
    out[blockIdx.x * blockDim.x + tid] = acc;
}

// Column access: tile[tid][i] -> address tid*32 + i -> bank i % 32.
// Every thread in the warp hits the SAME bank. 32-way conflict.
__global__ void tileColumnAccess(float* out, int iters) {
    __shared__ float tile[kTile][kTile];

    const int tid = threadIdx.x;
    for (int r = 0; r < kTile; ++r) tile[r][tid] = static_cast<float>(r * kTile + tid);
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) acc += tile[tid][k & (kTile - 1)];
    out[blockIdx.x * blockDim.x + tid] = acc;
}

// The fix: one extra column. Now tile[tid][i] is at tid*33 + i, so
// the bank is (tid + i) % 32 - all 32 banks, no conflict. Costs
// 128 bytes per tile.
__global__ void tileColumnPadded(float* out, int iters) {
    __shared__ float tile[kTile][kTile + 1];

    const int tid = threadIdx.x;
    for (int r = 0; r < kTile; ++r) tile[r][tid] = static_cast<float>(r * kTile + tid);
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) acc += tile[tid][k & (kTile - 1)];
    out[blockIdx.x * blockDim.x + tid] = acc;
}

// ------------------------------------------------------------
// Experiment 3: broadcast is free
// ------------------------------------------------------------
// Every thread reads the SAME address. That looks like the worst
// possible 32-way conflict, and it is in fact the best case: the
// hardware detects it and broadcasts one value to the whole warp
// in a single cycle.
__global__ void sharedBroadcast(float* out, int iters) {
    __shared__ float s[kSharedWords];

    const int tid = threadIdx.x;
    for (int i = tid; i < kSharedWords; i += blockDim.x) s[i] = static_cast<float>(i) * 0.5f;
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) acc += s[k & (kSharedWords - 1)];
    out[blockIdx.x * blockDim.x + tid] = acc;
}

int main() {
    printBanner("Phase 3 / 03 - Shared memory bank conflicts");
    requireCudaDevice();

    const int block = kBanks;  // one warp per block, to isolate the effect
    const int grid = kBlocks;
    const size_t outCount = static_cast<size_t>(grid) * block;

    printf("  %d banks of 4 bytes; bank = (address / 4) %% %d\n", kBanks, kBanks);
    printf("  %d blocks of %d threads (one warp each), %d shared reads per thread\n", grid,
           block, kIterations);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, outCount * sizeof(float)));

    // ========================================================
    // Correctness: every variant must read real data
    // ========================================================
    printSection("Correctness");
    {
        std::vector<float> host(outCount);
        sharedStride<<<grid, block>>>(d_out, 1, 4);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(host.data(), d_out, outCount * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Thread 0 reads s[0..3] = 0, 0.5, 1.0, 1.5 -> 3.0
        reportCheck("strided kernel reads the expected values", host[0] == 3.0f);

        tileColumnAccess<<<grid, block>>>(d_out, 4);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(host.data(), d_out, outCount * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Thread 0 reads tile[0][0..3] = 0, 1, 2, 3 -> 6
        reportCheck("tile kernel reads the expected values", host[0] == 6.0f);
    }

    // ========================================================
    // Experiment 1: stride sweep
    // ========================================================
    printSection("Experiment 1 - stride through shared memory");
    printf("  With stride s, a warp touches 32/gcd(s,32) distinct banks, so the\n");
    printf("  access serialises into gcd(s,32) cycles.\n\n");
    printf("  %-8s %12s %14s %12s %10s\n", "stride", "time (ms)", "predicted ways",
           "vs stride 1", "banks hit");
    printf("  --------------------------------------------------------------------\n");

    double baseline = 0.0;
    for (int stride : kStrides) {
        double ms =
            timeGpuMs(20, [&] { sharedStride<<<grid, block>>>(d_out, stride, kIterations); });
        if (stride == 1) baseline = ms;
        const int ways = predictedWays(stride);
        printf("  %-8d %12.3f %14d %11.2fx %10d\n", stride, ms, ways,
               baseline > 0.0 ? ms / baseline : 0.0, kBanks / ways);
    }

    printf("\n  Note stride 33: it is LARGER than 32 and yet conflict-free, because\n");
    printf("  gcd(33, 32) = 1. What matters is the stride modulo the bank count,\n");
    printf("  not its size. That is exactly why padding a tile by one column works.\n");

    printf("\n  The measured slowdown is roughly HALF the predicted way count (32-way\n");
    printf("  costs about 15x, not 32x), and stride 2 shows no penalty at all. That\n");
    printf("  is not the model being wrong - it is the loop having other work to do.\n");
    printf("  Each iteration also does an add, a mask and a branch, so a couple of\n");
    printf("  extra shared-memory cycles hide behind them. Only once the conflict\n");
    printf("  degree is large does shared memory become the binding constraint.\n");
    printf("  Predict the RATIO from the model; measure the absolute cost.\n");

    // ========================================================
    // Experiment 2: the 2D tile
    // ========================================================
    printSection("Experiment 2 - a 2D tile, by row and by column");

    double msRow = timeGpuMs(20, [&] { tileRowAccess<<<grid, block>>>(d_out, kIterations); });
    double msCol =
        timeGpuMs(20, [&] { tileColumnAccess<<<grid, block>>>(d_out, kIterations); });
    double msPad =
        timeGpuMs(20, [&] { tileColumnPadded<<<grid, block>>>(d_out, kIterations); });

    ResultTable table;
    table.add("tile[32][32], by row", msRow);
    table.add("tile[32][32], by column", msCol);
    table.add("tile[32][33], by column (padded)", msPad);
    table.print("32x32 shared tile");

    printf("\n  by row     tile[i][tid]  -> address i*32 + tid -> bank tid %% 32\n");
    printf("                              all 32 banks, 1 cycle\n");
    printf("  by column  tile[tid][i]  -> address tid*32 + i -> bank i %% 32\n");
    printf("                              every thread the SAME bank, 32 cycles\n");
    printf("  padded     tile[tid][i]  -> address tid*33 + i -> bank (tid+i) %% 32\n");
    printf("                              all 32 banks again, 1 cycle\n");
    printf("\n  Column access costs %.2fx. One extra column - %d bytes per tile -\n",
           msCol / msRow, kTile * static_cast<int>(sizeof(float)));
    printf("  recovers %.0f%% of it.\n", 100.0 * (msCol - msPad) / (msCol - msRow));

    // ========================================================
    // Experiment 3: broadcast
    // ========================================================
    printSection("Experiment 3 - broadcast");
    double msBroadcast =
        timeGpuMs(20, [&] { sharedBroadcast<<<grid, block>>>(d_out, kIterations); });
    printf("  Every thread reads the SAME address: %.3f ms (%.2fx vs stride 1)\n",
           msBroadcast, msBroadcast / baseline);
    printf("\n  That looks like the worst possible 32-way conflict and is in fact the\n");
    printf("  BEST case. The hardware detects that all 32 addresses are identical\n");
    printf("  and broadcasts one value to the whole warp in a single cycle.\n");
    printf("  A conflict is 32 threads wanting 32 DIFFERENT addresses in one bank -\n");
    printf("  not 32 threads wanting one address.\n");

    printSection("The rule");
    printf("  Within a warp, shared-memory addresses must map to distinct banks -\n");
    printf("  OR be identical (broadcast). Anything in between serialises.\n");
    printf("\n  In practice this means one habit: when a shared tile is written by\n");
    printf("  row and read by column, PAD THE ROW by one element. You will do it\n");
    printf("  again in exercise 04 (tiled matmul) and exercise 05 (transpose).\n");
    printf("\n  Diagnose with:\n");
    printf("    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./prog\n");

    CUDA_CHECK(cudaFree(d_out));

    reportCheck("column access is slower than row access", msCol > msRow * 1.5);
    reportCheck("padding removes most of the penalty", msPad < msRow * 1.5);
    reportCheck("broadcast is not a conflict", msBroadcast < baseline * 1.5);
    return verifySummary();
}
