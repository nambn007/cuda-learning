// ============================================================
// 02 - Shared memory and __syncthreads  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: reverse each block WITHOUT shared memory
// ------------------------------------------------------------
//     out[base + tid] = in[base + count - 1 - tid]
// Correct, and for this particular problem perfectly fast - both
// sides are contiguous across the warp, just in opposite
// directions. Shared memory is not always the answer, and this
// baseline is here to make that point.
__global__ void reverseGlobal(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: the same thing staged through shared memory
// ------------------------------------------------------------
//     __shared__ float tile[kBlockSize];   // one array PER BLOCK
//     tile[tid] = in[base + tid];
//     __syncthreads();
//     out[base + tid] = tile[count - 1 - tid];
//
// Put the __syncthreads() OUTSIDE any conditional. A barrier that
// only some threads reach is undefined behaviour.
__global__ void reverseShared(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: the same kernel with the barrier deliberately removed
// ------------------------------------------------------------
// Copy your working version and delete the __syncthreads(). Then
// count how many elements come out wrong.
//
// Predict first: what fraction do you expect to be wrong, and why?
// (Hint: which threads write the value that thread `tid` reads?)
__global__ void reverseSharedBroken(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

// ------------------------------------------------------------
// TODO 4: dynamic shared memory
// ------------------------------------------------------------
// When the tile size is a runtime tuning knob rather than a
// compile-time constant:
//
//     extern __shared__ float dynamicTile[];   // at file scope
//     kernel<<<grid, block, bytes>>>(...);     // size at launch
extern __shared__ float dynamicTile[];

__global__ void reverseSharedDynamic(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

// ------------------------------------------------------------
// TODO 5: the reuse pattern - what shared memory is FOR
// ------------------------------------------------------------
// Every thread needs the sum of every value in its block, then
// scales its own element by it:
//     out[i] = in[i] * (sum of the whole block)
//
// Naive: each thread loops over all blockSize values in GLOBAL
// memory - blockSize^2 global reads per block.
// Shared: load once into a tile, barrier, then loop over the tile -
// blockSize global reads per block.
//
// Predict the speedup from the read counts, then measure it. The
// gap between prediction and measurement is the lesson.
__global__ void broadcastGlobal(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

__global__ void broadcastShared(const float* in, float* out, size_t n, int blockSize) {
    (void)in;
    (void)out;
    (void)n;
    (void)blockSize;
    // TODO
}

int main() {
    printBanner("Phase 3 / 02 - Shared memory (starter)");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Shared memory: %zu KB per block, %zu KB per SM\n",
           prop.sharedMemPerBlock / 1024, prop.sharedMemPerMultiprocessor / 1024);

    const size_t n = kElements;
    std::vector<float> host(n), goldenReverse(n), goldenBroadcast(n);
    fillRandom(host.data(), n, -1.0f, 1.0f, 81);
    reverseBlocksCPU(host.data(), goldenReverse.data(), n, kBlockSize);
    blockBroadcastCPU(host.data(), goldenBroadcast.data(), n, kBlockSize);

    // --------------------------------------------------------
    // TODO 6: allocate, run each kernel, verify
    // --------------------------------------------------------
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 7: run the broken kernel and COUNT the wrong elements
    // --------------------------------------------------------
    // Do not just assert that it fails - report the percentage. If
    // it happens to come out right, that is the worst outcome: the
    // code is still wrong and will fail on a different GPU, block
    // size, or under load. Confirm either way with:
    //     compute-sanitizer --tool racecheck ./this_binary
    printSection("What happens without __syncthreads()");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 8: measure the reuse pattern, and work out the occupancy
    //         cost of shared memory
    // --------------------------------------------------------
    // prop.sharedMemPerMultiprocessor divided by the shared memory
    // your block asks for is an upper bound on blocks per SM.
    // Tabulate it for 4, 8, 16, 32 and 48 KB per block and see how
    // fast the resident-warp count falls.
    printSection("Reuse and its cost");
    printf("  TODO\n");

    printTodoNotice("implement the six kernels and the three experiments");
    return verifySummary();
}
