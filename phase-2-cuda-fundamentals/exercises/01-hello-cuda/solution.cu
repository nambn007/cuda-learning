// ============================================================
// 01 - Hello CUDA  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"

__global__ void helloCUDA() {
    // Every one of the 8 threads created by <<<2, 4>>> runs this.
    printf("  Hello from block %d, thread %d (global id %d)\n", blockIdx.x, threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

__global__ void writeGlobalId(int* out, int n) {
    // The most important line in CUDA.
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // The bounds check is not optional. With n = 1000 and 256
    // threads per block we launch 1024 threads, so 24 of them have
    // gid >= n. Without this guard they would write past the end of
    // the allocation - which on a GPU usually does not crash, it
    // just silently corrupts whatever is next in memory.
    if (gid < n) {
        out[gid] = gid;
    }
}

int main() {
    printBanner("Phase 2 / 01 - Hello CUDA");
    requireCudaDevice();
    printDeviceInfo();

    // ========================================================
    // Part 1: a kernel that talks
    // ========================================================
    printSection("Kernel printf");
    printf("  Launching <<<2, 4>>> = 2 blocks x 4 threads = 8 threads\n\n");

    helloCUDA<<<2, 4>>>();

    // Two different failures need two different checks:
    //
    // cudaGetLastError() catches problems with the LAUNCH itself -
    // too many threads per block, too much shared memory, no such
    // device function. It reports immediately.
    CUDA_CHECK_LAST();

    // cudaDeviceSynchronize() waits for the kernel to finish and
    // reports problems during EXECUTION - an illegal address, an
    // assertion. Kernel launches are asynchronous, so without this
    // the host would race ahead and the printf output would appear
    // at some arbitrary later point.
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n  Notice the lines are not in order. Blocks are scheduled onto SMs\n");
    printf("  in whatever order resources free up, and nothing in CUDA promises\n");
    printf("  an ordering between blocks. Code that depends on one is broken.\n");

    // ========================================================
    // Part 2: a kernel we can verify
    // ========================================================
    printSection("Verifiable kernel");

    const int n = 1000;
    const int blockSize = 256;
    const int gridSize = ceilDiv(n, blockSize);
    printf("  n = %d, block = %d, grid = %d (%d threads, %d idle)\n", n, blockSize,
           gridSize, gridSize * blockSize, gridSize * blockSize - n);

    // The four-step dance that every CUDA program performs:
    //   1. allocate on the device
    //   2. (copy input host -> device)
    //   3. launch
    //   4. copy result device -> host, then free
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    writeGlobalId<<<gridSize, blockSize>>>(d_out, n);
    CUDA_CHECK_LAST();

    std::vector<int> host(n, -1);
    // cudaMemcpy is synchronous with respect to the host: it waits
    // for all previously issued work in the stream, so it doubles
    // as the synchronisation point here.
    CUDA_CHECK(cudaMemcpy(host.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<int> expected(n);
    for (int i = 0; i < n; ++i) expected[i] = i;
    checkArrayExact("every thread wrote its global id", host.data(), expected.data(), n);

    printArray("first values", host.data(), n, 10);

    // ========================================================
    // Part 3: what a launch configuration costs
    // ========================================================
    printSection("Block size is a choice, not a formality");
    printf("  A block runs entirely on one SM and is scheduled in warps of 32.\n");
    printf("  A few consequences:\n\n");
    printf("    - Always make blockDim a multiple of 32. A block of 100 threads\n");
    printf("      occupies 4 warps (128 slots) and wastes 28 of them, forever.\n");
    printf("    - 256 is a good default: it divides into 8 warps, and several\n");
    printf("      blocks still fit per SM so the scheduler has work to switch to.\n");
    printf("    - The grid should cover the data: ceilDiv(n, blockSize).\n\n");

    printf("  %-14s %-10s %-10s %s\n", "n", "block", "grid", "threads launched (waste)");
    printf("  ----------------------------------------------------------------\n");
    for (int size : {1000, 1024, 100000, 1000000}) {
        for (int bs : {128, 256}) {
            int g = ceilDiv(size, bs);
            printf("  %-14d %-10d %-10d %d (%d idle)\n", size, bs, g, g * bs, g * bs - size);
        }
    }

    return verifySummary();
}
