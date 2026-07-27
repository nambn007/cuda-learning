// ============================================================
// 02 - Shared memory and __syncthreads  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// Reversing a block, three ways
// ------------------------------------------------------------

// No shared memory: read directly from the mirrored global address.
// This is correct and, for this particular problem, perfectly fine -
// both the read and the write are contiguous across the warp, just
// in opposite directions. Shared memory is not always the answer.
__global__ void reverseGlobal(const float* in, float* out, size_t n, int blockSize) {
    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    if (base + tid < n) {
        const size_t count = min(static_cast<size_t>(blockSize), n - base);
        if (static_cast<size_t>(tid) < count) {
            out[base + tid] = in[base + count - 1 - tid];
        }
    }
}

// Staged through shared memory. `__shared__` inside a kernel
// declares one array PER BLOCK, with a compile-time size.
__global__ void reverseShared(const float* in, float* out, size_t n, int blockSize) {
    __shared__ float tile[kBlockSize];

    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    const size_t count = (base < n) ? min(static_cast<size_t>(blockSize), n - base) : 0;

    if (static_cast<size_t>(tid) < count) tile[tid] = in[base + tid];

    // The barrier. Every thread in the block must reach it, and
    // every shared-memory write issued before it becomes visible to
    // every thread after it. Note it is OUTSIDE the `if` above -
    // putting a __syncthreads() inside a conditional that not all
    // threads take is undefined behaviour.
    __syncthreads();

    if (static_cast<size_t>(tid) < count) {
        out[base + tid] = tile[count - 1 - tid];
    }
}

// The same kernel with the barrier removed. Thread `tid` reads
// tile[count-1-tid], which is written by a DIFFERENT thread - very
// possibly one in a different warp that has not run yet.
//
// This is here to be run, not to be copied.
__global__ void reverseSharedBroken(const float* in, float* out, size_t n, int blockSize) {
    __shared__ float tile[kBlockSize];

    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    const size_t count = (base < n) ? min(static_cast<size_t>(blockSize), n - base) : 0;

    if (static_cast<size_t>(tid) < count) tile[tid] = in[base + tid];
    // __syncthreads();   <- deliberately missing
    if (static_cast<size_t>(tid) < count) out[base + tid] = tile[count - 1 - tid];
}

// Dynamic shared memory: the size is a launch parameter rather than
// a compile-time constant. `extern __shared__` declares it; the
// third argument of <<<grid, block, bytes>>> supplies the size.
// Use it when the tile size is a runtime tuning knob.
extern __shared__ float dynamicTile[];

__global__ void reverseSharedDynamic(const float* in, float* out, size_t n, int blockSize) {
    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    const size_t count = (base < n) ? min(static_cast<size_t>(blockSize), n - base) : 0;

    if (static_cast<size_t>(tid) < count) dynamicTile[tid] = in[base + tid];
    __syncthreads();
    if (static_cast<size_t>(tid) < count) out[base + tid] = dynamicTile[count - 1 - tid];
}

// ------------------------------------------------------------
// Reuse: every thread needs every value in its block
// ------------------------------------------------------------

// Naive: each thread reads all blockSize values from global memory.
// That is blockSize^2 global reads per block. The L1 cache absorbs
// much of it - which is exactly why the speedup below is smaller
// than the read-count ratio suggests.
__global__ void broadcastGlobal(const float* in, float* out, size_t n, int blockSize) {
    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    const size_t count = (base < n) ? min(static_cast<size_t>(blockSize), n - base) : 0;
    if (static_cast<size_t>(tid) >= count) return;

    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) sum += in[base + i];
    out[base + tid] = in[base + tid] * sum;
}

// Shared: one global read per thread, then everything from on-chip
// memory. blockSize^2 global reads become blockSize.
__global__ void broadcastShared(const float* in, float* out, size_t n, int blockSize) {
    __shared__ float tile[kBlockSize];

    const size_t base = blockIdx.x * static_cast<size_t>(blockSize);
    const int tid = threadIdx.x;
    const size_t count = (base < n) ? min(static_cast<size_t>(blockSize), n - base) : 0;

    if (static_cast<size_t>(tid) < count) tile[tid] = in[base + tid];
    __syncthreads();

    if (static_cast<size_t>(tid) >= count) return;
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) sum += tile[i];
    out[base + tid] = tile[tid] * sum;
}

int main() {
    printBanner("Phase 3 / 02 - Shared memory and __syncthreads");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    const int blockSize = kBlockSize;
    const int grid = static_cast<int>(ceilDiv(n, static_cast<size_t>(blockSize)));

    printf("  n = %zu floats, block = %d, grid = %d\n", n, blockSize, grid);
    printf("  Shared memory: %zu KB per block, %zu KB per SM\n",
           prop.sharedMemPerBlock / 1024, prop.sharedMemPerMultiprocessor / 1024);

    std::vector<float> host(n), result(n), goldenReverse(n), goldenBroadcast(n);
    fillRandom(host.data(), n, -1.0f, 1.0f, 81);
    reverseBlocksCPU(host.data(), goldenReverse.data(), n, blockSize);
    blockBroadcastCPU(host.data(), goldenBroadcast.data(), n, blockSize);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));

    auto fetch = [&] {
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    };

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    reverseGlobal<<<grid, blockSize>>>(d_in, d_out, n, blockSize);
    CUDA_CHECK_LAST();
    fetch();
    checkArray("reverse via global memory", result, goldenReverse, 0.0, 0.0);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    reverseShared<<<grid, blockSize>>>(d_in, d_out, n, blockSize);
    CUDA_CHECK_LAST();
    fetch();
    checkArray("reverse via shared memory", result, goldenReverse, 0.0, 0.0);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    reverseSharedDynamic<<<grid, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, n,
                                                                        blockSize);
    CUDA_CHECK_LAST();
    fetch();
    checkArray("reverse via dynamic shared memory", result, goldenReverse, 0.0, 0.0);

    // ========================================================
    // The missing barrier
    // ========================================================
    printSection("What happens without __syncthreads()");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    reverseSharedBroken<<<grid, blockSize>>>(d_in, d_out, n, blockSize);
    CUDA_CHECK_LAST();
    fetch();

    size_t wrong = 0;
    for (size_t i = 0; i < n; ++i)
        if (result[i] != goldenReverse[i]) ++wrong;

    printf("  %zu of %zu elements wrong (%.1f%%)\n", wrong, n,
           100.0 * static_cast<double>(wrong) / static_cast<double>(n));

    if (wrong == 0) {
        printf("\n  It happened to produce the right answer THIS TIME. That is the\n");
        printf("  worst possible outcome: the code is still wrong, and the bug will\n");
        printf("  appear on a different GPU, a different block size, or under load.\n");
        printf("  Run `compute-sanitizer --tool racecheck` on this binary - it\n");
        printf("  reports the hazard whether or not it fired.\n");
    } else {
        printf("\n  Thread `tid` reads tile[count-1-tid], which a DIFFERENT thread\n");
        printf("  wrote - very possibly one in a warp that has not run yet. Without\n");
        printf("  the barrier there is no ordering between them at all.\n");
    }

    printf("\n  Two rules worth memorising\n");
    printf("    1. Every thread in the block must reach every __syncthreads().\n");
    printf("       One inside `if (tid < 100)` in a 256-thread block is undefined\n");
    printf("       behaviour - typically a hang.\n");
    printf("    2. Since Volta, threads in a warp are NOT implicitly synchronised.\n");
    printf("       Code that omitted the barrier and 'worked' on Kepler because a\n");
    printf("       warp moved in lockstep is broken on every modern GPU. If you\n");
    printf("       mean warp-level synchronisation, write __syncwarp().\n");

    // ========================================================
    // Reuse - what shared memory is actually for
    // ========================================================
    printSection("Reuse: every thread needs every value in its block");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    broadcastGlobal<<<grid, blockSize>>>(d_in, d_out, n, blockSize);
    CUDA_CHECK_LAST();
    fetch();
    checkArray("broadcast via global memory", result, goldenBroadcast, 1e-3, 1e-4);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    broadcastShared<<<grid, blockSize>>>(d_in, d_out, n, blockSize);
    CUDA_CHECK_LAST();
    fetch();
    checkArray("broadcast via shared memory", result, goldenBroadcast, 1e-3, 1e-4);

    double msGlobal =
        timeGpuMs(10, [&] { broadcastGlobal<<<grid, blockSize>>>(d_in, d_out, n, blockSize); });
    double msShared =
        timeGpuMs(10, [&] { broadcastShared<<<grid, blockSize>>>(d_in, d_out, n, blockSize); });

    ResultTable table;
    table.add("global memory", msGlobal);
    table.add("shared memory", msShared);
    table.print("Block-wide broadcast");

    printf("\n  Global reads per output element\n");
    printf("    naive  : %.0f   (every thread reads the whole block)\n",
           naiveReadsPerElement(blockSize));
    printf("    shared : %.0f   (each value read once, then served on-chip)\n",
           sharedReadsPerElement());
    printf("    ratio  : %.0fx fewer reads, %.2fx faster measured\n",
           naiveReadsPerElement(blockSize) / sharedReadsPerElement(), msGlobal / msShared);
    printf("\n  The measured speedup is smaller than the read ratio because L1\n");
    printf("  absorbs much of the naive version's repetition - the block fits in\n");
    printf("  cache. Shared memory wins because it is EXPLICIT: guaranteed to be\n");
    printf("  on-chip, guaranteed not to be evicted by another block's traffic.\n");

    // ========================================================
    // The cost: shared memory limits occupancy
    // ========================================================
    printSection("Shared memory is a budget");
    {
        const size_t perSM = prop.sharedMemPerMultiprocessor;
        printf("  This SM has %zu KB of shared memory, and it is divided among all\n",
               perSM / 1024);
        printf("  resident blocks. That makes it an occupancy limit:\n\n");
        printf("  %-22s %14s %16s\n", "shared per block", "blocks per SM", "threads per SM");
        printf("  ------------------------------------------------------\n");
        for (size_t kb : {4u, 8u, 16u, 32u, 48u}) {
            const size_t perBlock = kb * 1024;
            size_t blocks = perSM / perBlock;
            const size_t maxBlocks = static_cast<size_t>(prop.maxBlocksPerMultiProcessor);
            if (blocks > maxBlocks) blocks = maxBlocks;
            const size_t threads =
                blocks * blockSize > static_cast<size_t>(prop.maxThreadsPerMultiProcessor)
                    ? static_cast<size_t>(prop.maxThreadsPerMultiProcessor)
                    : blocks * blockSize;
            printf("  %18zu KB %14zu %16zu\n", kb, blocks, threads);
        }
        printf("\n  Ask for 48 KB per block and only %zu blocks fit per SM, so you\n",
               perSM / (48 * 1024));
        printf("  have far fewer warps to hide memory latency with. Shared memory\n");
        printf("  buys you reuse and costs you occupancy - Phase 3 exercise 15 is\n");
        printf("  about finding the balance.\n");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    reportCheck("shared memory beats global for the reuse pattern", msShared < msGlobal);
    return verifySummary();
}
