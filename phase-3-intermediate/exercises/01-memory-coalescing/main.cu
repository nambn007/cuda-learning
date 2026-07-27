// ============================================================
// 01 - Memory coalescing, measured  [STARTER]
// ============================================================
// The GPU version of Phase 1 exercise 05. Same physical fact,
// different block size: a CPU moves 64-byte cache lines, a GPU
// moves 32-byte sectors.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: strided read
// ------------------------------------------------------------
//     out[i] = in[i * stride]
// The write is always contiguous, so the read is the only variable.
// Grid-stride loop over `touched` elements.
__global__ void stridedRead(const float* in, float* out, size_t touched, int stride) {
    (void)in;
    (void)out;
    (void)touched;
    (void)stride;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: offset (misaligned) read
// ------------------------------------------------------------
//     out[i] = in[i + offset]
// Perfectly contiguous - it just does not start on a sector
// boundary. Predict what that should cost before measuring it.
__global__ void offsetRead(const float* in, float* out, size_t n, int offset) {
    (void)in;
    (void)out;
    (void)n;
    (void)offset;
    // TODO
}

// ------------------------------------------------------------
// TODO 3: the mapping that feels right and is not
// ------------------------------------------------------------
// Give each thread a contiguous chunk of `chunk` elements:
//     start = tid * chunk
//     for k in 0..chunk:  out[start+k] = in[start+k]
//
// This is how you parallelise on a CPU, where each core wants its
// own cache-friendly region. Write it, then write the interleaved
// version below, and predict which wins and by how much before you
// run either.
__global__ void blockPerThread(const float* in, float* out, size_t n, int chunk) {
    (void)in;
    (void)out;
    (void)n;
    (void)chunk;
    // TODO
}

// TODO 4: the same work, mapped the GPU way - neighbouring threads
// take neighbouring elements, each striding by the whole grid.
__global__ void interleaved(const float* in, float* out, size_t n) {
    (void)in;
    (void)out;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 3 / 01 - Memory coalescing (starter)");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Array: %zu floats (%.0f MB), L2 is %d KB - this does not fit\n", kElements,
           kElements * sizeof(float) / (1024.0 * 1024.0), prop.l2CacheSize / 1024);

    // --------------------------------------------------------
    // TODO 5: allocate, fill, and verify each kernel gathers the
    //         right elements before timing anything
    // --------------------------------------------------------
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 6: sweep the stride
    // --------------------------------------------------------
    // For each stride in kStrides, touch n/stride elements and
    // print three columns:
    //   useful GB/s    = gbPerSec(usefulBytes(touched), ms)
    //   sectors/warp   = sectorsPerWarp(stride)     (the model)
    //   predicted GB/s = gbPerSec(predictedReadBytes(...) + writes, ms)
    //
    // The interesting part is that the LAST column should stay
    // roughly constant while the first collapses. Work out what
    // that means before reading the solution.
    printSection("Experiment 1 - stride");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 7: sweep the offset
    // --------------------------------------------------------
    // Predict first: a warp reads 128 consecutive bytes either way.
    // How much can a misaligned start possibly cost? And what do
    // you expect at offset 32 specifically?
    printSection("Experiment 2 - alignment");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 8: chunk-per-thread versus interleaved
    // --------------------------------------------------------
    // Write your prediction down first. This one is usually the
    // biggest surprise in Phase 3.
    printSection("Experiment 3 - chunk per thread versus interleaved");
    printf("  TODO\n");

    printTodoNotice("implement the four kernels and the three experiments");
    return verifySummary();
}
