// ============================================================
// 01 - Pinned memory and stream overlap  [SOLUTION]
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
    printBanner("Phase 4 / 01 - Pinned memory and stream overlap");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;

    printf("  n = %zu floats (%.0f MB per buffer)\n", n, bytes / (1024.0 * 1024.0));
    printf("  Copy engines: %d, concurrent copy+kernel: %s\n", prop.asyncEngineCount,
           prop.deviceOverlap ? "yes" : "no");

    // ========================================================
    // Host buffers: pageable and pinned
    // ========================================================
    std::vector<float> pageableIn(n), pageableOut(n), golden(n);
    fillRandom(pageableIn.data(), n, 0.0f, 1.0f, 121);
    workloadCPU(kAlpha, pageableIn.data(), golden.data(), n);

    float *pinnedIn = nullptr, *pinnedOut = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinnedIn, bytes));
    CUDA_CHECK(cudaMallocHost(&pinnedOut, bytes));
    for (size_t i = 0; i < n; ++i) pinnedIn[i] = pageableIn[i];

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // ========================================================
    // 1. Raw transfer bandwidth
    // ========================================================
    printSection("Transfer bandwidth");
    {
        double msPageable = timeCpuMs(5, [&] {
            CUDA_CHECK(cudaMemcpy(d_x, pageableIn.data(), bytes, cudaMemcpyHostToDevice));
        });
        double msPinned = timeCpuMs(5, [&] {
            CUDA_CHECK(cudaMemcpy(d_x, pinnedIn, bytes, cudaMemcpyHostToDevice));
        });

        ResultTable t;
        t.add("pageable host -> device", msPageable, static_cast<double>(bytes));
        t.add("pinned host -> device", msPinned, static_cast<double>(bytes));
        t.print("Host to device, 256 MB");

        printf("\n  Pageable memory can be swapped out, so the driver cannot give its\n");
        printf("  address to the DMA engine - it stages the copy through an internal\n");
        printf("  pinned buffer first. cudaMallocHost skips that: %.2fx here.\n",
               msPageable / msPinned);
    }

    // ========================================================
    // 2. Sequential: transfer, compute, transfer
    // ========================================================
    printSection("Sequential pipeline");
    double msSequential = 0.0;
    {
        auto round = [&] {
            CUDA_CHECK(cudaMemcpy(d_x, pinnedIn, bytes, cudaMemcpyHostToDevice));
            workload<<<grid, block>>>(kAlpha, d_x, d_y, n);
            CUDA_CHECK(cudaMemcpy(pinnedOut, d_y, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
        };
        round();
        checkArray("sequential result", pinnedOut, golden.data(), n, 1e-4, 1e-5);
        msSequential = timeCpuMs(5, round, 1);
        printKV("H2D + kernel + D2H", msSequential, "ms");
    }

    // Kernel time on its own, for the model below.
    double msKernel = timeGpuMs(20, [&] { workload<<<grid, block>>>(kAlpha, d_x, d_y, n); });

    // ========================================================
    // 3. Chunked pipeline across several streams
    // ========================================================
    // Split the data into chunks and issue H2D / kernel / D2H for
    // each into a rotating set of streams. While chunk i computes,
    // chunk i+1 is being copied in and chunk i-1 copied out.
    printSection("Overlapped pipeline");
    double msOverlapped = 0.0;
    {
        cudaStream_t streams[kStreams];
        for (int s = 0; s < kStreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

        const size_t chunk = (n + kChunks - 1) / kChunks;

        auto round = [&] {
            for (int c = 0; c < kChunks; ++c) {
                const size_t offset = static_cast<size_t>(c) * chunk;
                if (offset >= n) break;
                const size_t count = (offset + chunk <= n) ? chunk : (n - offset);
                const size_t chunkBytes = count * sizeof(float);
                cudaStream_t s = streams[c % kStreams];

                // Every operation names its stream. A single call
                // without one would land in the default stream and
                // serialise everything.
                CUDA_CHECK(cudaMemcpyAsync(d_x + offset, pinnedIn + offset, chunkBytes,
                                           cudaMemcpyHostToDevice, s));
                workload<<<grid / kStreams + 1, block, 0, s>>>(kAlpha, d_x + offset,
                                                               d_y + offset, count);
                CUDA_CHECK(cudaMemcpyAsync(pinnedOut + offset, d_y + offset, chunkBytes,
                                           cudaMemcpyDeviceToHost, s));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        // Wipe the output first, so a pipeline that silently does
        // nothing cannot pass by reusing the previous result.
        for (size_t i = 0; i < n; ++i) pinnedOut[i] = 0.0f;
        round();
        checkArray("overlapped result", pinnedOut, golden.data(), n, 1e-4, 1e-5);

        msOverlapped = timeCpuMs(5, round, 1);
        printKV("chunked across streams", msOverlapped, "ms");

        for (int s = 0; s < kStreams; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    // ========================================================
    // 4. The trap: one call in the default stream
    // ========================================================
    // The default stream synchronises with every other stream. A
    // single launch or copy that forgets its stream argument turns
    // the pipeline back into a sequence - and the answer stays
    // correct, so nothing warns you.
    printSection("The default-stream trap");
    double msPoisoned = 0.0;
    {
        cudaStream_t streams[kStreams];
        for (int s = 0; s < kStreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
        const size_t chunk = (n + kChunks - 1) / kChunks;

        msPoisoned = timeCpuMs(3, [&] {
            for (int c = 0; c < kChunks; ++c) {
                const size_t offset = static_cast<size_t>(c) * chunk;
                if (offset >= n) break;
                const size_t count = (offset + chunk <= n) ? chunk : (n - offset);
                const size_t chunkBytes = count * sizeof(float);
                cudaStream_t s = streams[c % kStreams];

                CUDA_CHECK(cudaMemcpyAsync(d_x + offset, pinnedIn + offset, chunkBytes,
                                           cudaMemcpyHostToDevice, s));
                // <- the bug: no stream argument, so this goes to
                //    the default stream and acts as a barrier.
                workload<<<grid / kStreams + 1, block>>>(kAlpha, d_x + offset,
                                                         d_y + offset, count);
                CUDA_CHECK(cudaMemcpyAsync(pinnedOut + offset, d_y + offset, chunkBytes,
                                           cudaMemcpyDeviceToHost, s));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }, 1);

        printKV("same pipeline, kernel in the default stream", msPoisoned, "ms");
        for (int s = 0; s < kStreams; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    // ========================================================
    // Summary
    // ========================================================
    printSection("Summary");
    ResultTable table;
    table.add("sequential", msSequential, transferBytes(n));
    table.add("overlapped, 4 streams", msOverlapped, transferBytes(n));
    table.add("overlapped, but one default-stream call", msPoisoned, transferBytes(n));
    table.print("End-to-end, 256 MB in and out");

    printf("\n  The model\n");
    printf("    sequential  = H2D + kernel + D2H\n");
    printf("    overlapped -> max(H2D, kernel, D2H) + one chunk of latency\n");
    printf("\n    kernel alone       : %.2f ms\n", msKernel);
    printf("    sequential total   : %.2f ms\n", msSequential);
    printf("    overlapped total   : %.2f ms  (%.2fx)\n", msOverlapped,
           msSequential / msOverlapped);
    printf("    with the default-stream bug: %.2f ms  (%.2fx of the good pipeline)\n",
           msPoisoned, msPoisoned / msOverlapped);

    printf("\n  Note how MILD the default-stream bug looks here: only %.0f%% worse.\n",
           100.0 * (msPoisoned / msOverlapped - 1.0));
    printf("  That is because the kernel is %.2f ms out of %.2f - the transfers\n",
           msKernel, msSequential);
    printf("  dominate, and H2D still overlaps with D2H even when the kernel does\n");
    printf("  not. Make the kernel the expensive part and the same bug costs you\n");
    printf("  everything. A small measured penalty does not mean a small bug; it\n");
    printf("  means this particular workload was not sensitive to it.\n");

    printf("\n  Two things to take away\n");
    printf("    1. cudaMemcpyAsync on PAGEABLE memory is silently synchronous. A\n");
    printf("       pipeline built on it overlaps nothing while looking correct.\n");
    printf("       Pinned memory is a requirement, not an optimisation.\n");
    printf("    2. The default stream synchronises with every other stream. One\n");
    printf("       call that forgets its stream argument collapses the timeline,\n");
    printf("       and the result stays correct, so no test catches it.\n");
    printf("       Confirm overlap on a timeline, not from the answer:\n");
    printf("           nsys profile --stats=true ./p4_01_pinned_and_streams\n");

    CUDA_CHECK(cudaFreeHost(pinnedIn));
    CUDA_CHECK(cudaFreeHost(pinnedOut));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    reportCheck("overlapping beats the sequential pipeline", msOverlapped < msSequential);
    return verifySummary();
}
