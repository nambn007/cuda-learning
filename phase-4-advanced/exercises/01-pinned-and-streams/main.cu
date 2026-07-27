// ============================================================
// 01 - Pinned memory and stream overlap  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void workload(float a, const float* x, float* y, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) {
        float v = x[i];
#pragma unroll
        for (int k = 0; k < 8; ++k) v = v * a + 1.0f;
        y[i] = v;
    }
}

int main() {
    printBanner("Phase 4 / 01 - Pinned memory and streams (starter)");
    requireCudaDevice();

    const size_t n = kElements;
    printf("  n = %zu floats (%.0f MB per buffer)\n", n,
           n * sizeof(float) / (1024.0 * 1024.0));

    // --------------------------------------------------------
    // TODO 1: allocate pinned host memory
    // --------------------------------------------------------
    //     cudaMallocHost(&p, bytes)  /  cudaFreeHost(p)
    // Compare its host-to-device bandwidth with a std::vector.
    printSection("Transfer bandwidth");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 2: the sequential pipeline
    // --------------------------------------------------------
    //     H2D (whole buffer) -> kernel -> D2H (whole buffer)
    // Verify the result, then time it.
    printSection("Sequential pipeline");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: the overlapped pipeline
    // --------------------------------------------------------
    // Create kStreams streams. Split the data into kChunks pieces
    // and, for each, issue into a rotating stream:
    //     cudaMemcpyAsync(H2D, ..., stream)
    //     kernel<<<g, b, 0, stream>>>(...)
    //     cudaMemcpyAsync(D2H, ..., stream)
    // then one cudaDeviceSynchronize() at the end.
    //
    // EVERY operation must name its stream. Zero the output buffer
    // before verifying, so a pipeline that silently does nothing
    // cannot pass by reusing the previous result.
    //
    // Predict the speedup from:
    //     sequential  = H2D + kernel + D2H
    //     overlapped -> max(H2D, kernel, D2H)
    printSection("Overlapped pipeline");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: reproduce the default-stream trap
    // --------------------------------------------------------
    // Copy your working pipeline and delete the stream argument
    // from the KERNEL LAUNCH only. The default stream synchronises
    // with every other stream, so this should serialise everything.
    //
    // Measure it. Then ask why the penalty is the size it is -
    // compare the kernel time with the total, and work out which
    // operations are still able to overlap.
    printSection("The default-stream trap");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 5: confirm the overlap on a timeline, not from the answer
    // --------------------------------------------------------
    //     nsys profile --stats=true ./build/bin/p4/p4_01_pinned_and_streams
    // A correct result proves nothing about concurrency.

    printTodoNotice("implement the three pipelines and the comparison");
    return verifySummary();
}
