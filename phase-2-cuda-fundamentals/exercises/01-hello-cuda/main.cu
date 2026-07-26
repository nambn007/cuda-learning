// ============================================================
// 01 - Hello CUDA  [STARTER]
// ============================================================
// Your first kernel. Three things happen here that never happen in
// ordinary C++:
//   1. a function runs on a different processor
//   2. it runs thousands of times at once
//   3. the call returns before the work is done
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"

// ------------------------------------------------------------
// TODO 1: write a kernel that greets you
// ------------------------------------------------------------
// `__global__` means: compiled for the GPU, callable from the CPU,
// must return void. Every thread that the launch creates runs this
// same body - that is the whole SIMT model.
//
// Inside a kernel you get four built-in variables:
//   threadIdx.x  index of this thread inside its block   [0, blockDim.x)
//   blockIdx.x   index of this block inside the grid     [0, gridDim.x)
//   blockDim.x   number of threads per block
//   gridDim.x    number of blocks in the grid
//
// Print the block and thread index. printf() works inside a kernel;
// output is buffered on the device and flushed at the next
// synchronisation point, which is why the lines can appear out of
// order.
__global__ void helloCUDA() {
    // TODO: printf("Hello from block %d, thread %d\n", ...);
}

// ------------------------------------------------------------
// TODO 2: write a kernel we can actually verify
// ------------------------------------------------------------
// printf proves a kernel ran; it does not prove it computed the
// right thing. Have every thread store its GLOBAL index:
//
//     global id = blockIdx.x * blockDim.x + threadIdx.x
//
// Memorise that line. It is the single most used expression in all
// of CUDA, and every exercise from here on starts with it.
//
// Guard the write with `if (gid < n)`. The grid almost never
// divides the data exactly, so the last block has idle threads that
// must not touch memory they do not own.
__global__ void writeGlobalId(int* out, int n) {
    (void)out;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 2 / 01 - Hello CUDA (starter)");
    requireCudaDevice();
    printDeviceInfo();

    // --------------------------------------------------------
    // Launching a kernel
    // --------------------------------------------------------
    // The <<<grid, block>>> syntax is the only piece of non-C++
    // syntax in CUDA. It says: create `grid` blocks of `block`
    // threads each, and run the kernel in every one of them.
    printSection("Kernel printf");
    printf("  Launching <<<2, 4>>> = 2 blocks x 4 threads = 8 threads\n\n");

    helloCUDA<<<2, 4>>>();

    // A launch is ASYNCHRONOUS: it returns immediately, before the
    // GPU has done anything. Two separate checks are therefore
    // needed - one for the launch itself, one for execution.
    CUDA_CHECK_LAST();              // did the launch configuration fail?
    CUDA_CHECK(cudaDeviceSynchronize());  // wait, then check execution

    // --------------------------------------------------------
    // TODO 3: allocate, launch, copy back, verify
    // --------------------------------------------------------
    printSection("Verifiable kernel");
    const int n = 1000;
    const int blockSize = 256;
    const int gridSize = ceilDiv(n, blockSize);  // 4 blocks covers 1000
    printf("  n = %d, block = %d, grid = %d (%d threads, %d idle)\n", n, blockSize,
           gridSize, gridSize * blockSize, gridSize * blockSize - n);

    // TODO:
    //   int* d_out = nullptr;
    //   CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    //   writeGlobalId<<<gridSize, blockSize>>>(d_out, n);
    //   CUDA_CHECK_LAST();
    //   std::vector<int> host(n);
    //   CUDA_CHECK(cudaMemcpy(host.data(), d_out, n * sizeof(int),
    //                         cudaMemcpyDeviceToHost));
    //   CUDA_CHECK(cudaFree(d_out));
    // then check that host[i] == i for every i.

    std::vector<int> host(n, -1);
    std::vector<int> expected(n);
    for (int i = 0; i < n; ++i) expected[i] = i;

    if (host[0] != 0) {
        printTodoNotice("write the kernels and the memory plumbing in main.cu");
        return verifySummary();
    }

    checkArrayExact("every thread wrote its global id", host.data(), expected.data(), n);
    return verifySummary();
}
