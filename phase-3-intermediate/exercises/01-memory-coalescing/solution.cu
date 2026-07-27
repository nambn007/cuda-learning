// ============================================================
// 01 - Memory coalescing, measured  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// Strided read
// ------------------------------------------------------------
// Thread i reads in[i * stride]. With stride 1 a warp reads 128
// contiguous bytes (4 sectors). With stride 8 the same warp reads
// 32 addresses 32 bytes apart - 32 separate sectors, four useful
// bytes from each.
//
// The write is always contiguous, so the read is the only variable.
__global__ void stridedRead(const float* in, float* out, size_t touched, int stride) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < touched; i += gridStride) {
        out[i] = in[i * static_cast<size_t>(stride)];
    }
}

// ------------------------------------------------------------
// Offset (misaligned) read
// ------------------------------------------------------------
// Perfectly contiguous, but starting `offset` floats into the
// array. A warp still reads 128 consecutive bytes - they just do
// not begin on a sector boundary, so they span 5 sectors instead
// of 4. One extra sector per warp: 25% more traffic for a
// completely innocent-looking `+ offset`.
__global__ void offsetRead(const float* in, float* out, size_t n, int offset) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += gridStride) {
        out[i] = in[i + offset];
    }
}

// ------------------------------------------------------------
// The pattern that looks reasonable and is not
// ------------------------------------------------------------
// "Give each thread a contiguous chunk of 32 elements to work on."
// It is how you would parallelise on a CPU, where each core wants
// its own cache-friendly region. On a GPU it is the worst possible
// mapping: at any instant the 32 threads of a warp are each at the
// start of their own chunk, 128 bytes apart.
__global__ void blockPerThread(const float* in, float* out, size_t n, int chunk) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t start = tid * chunk;
    for (int k = 0; k < chunk; ++k) {
        const size_t i = start + k;
        if (i < n) out[i] = in[i];
    }
}

// The same work, mapped the GPU way: neighbouring threads take
// neighbouring elements, and each thread strides by the whole grid.
__global__ void interleaved(const float* in, float* out, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += gridStride) out[i] = in[i];
}

int main() {
    printBanner("Phase 3 / 01 - Memory coalescing");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;
    const double peak = theoreticalBandwidthGBs();

    printf("  Array: %zu floats (%.0f MB), L2 is %d KB - this does not fit\n", n,
           bytes / (1024.0 * 1024.0), prop.l2CacheSize / 1024);
    printf("  Sector size: %d bytes, warp size: %d threads\n", kSectorBytes, kWarpSize);

    std::vector<float> host(n);
    fillSequence(host.data(), n, 0.0f, 1.0f);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");
    {
        const size_t touched = n / 4;
        std::vector<float> result(touched), expected(touched);
        stridedRead<<<grid, block>>>(d_in, d_out, touched, 4);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, touched * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < touched; ++i) expected[i] = host[i * 4];
        checkArray("stride-4 read gathers the right elements", result, expected, 0.0, 0.0);

        offsetRead<<<grid, block>>>(d_in, d_out, touched, 3);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, touched * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < touched; ++i) expected[i] = host[i + 3];
        checkArray("offset-3 read gathers the right elements", result, expected, 0.0, 0.0);
    }

    // ========================================================
    // Experiment 1: stride
    // ========================================================
    printSection("Experiment 1 - stride");
    printf("  Each row touches the same NUMBER of elements, spread further apart.\n");
    printf("  'predicted' comes from the sector model in reference.h.\n\n");
    printf("  %-8s %11s %13s %13s %11s %9s\n", "stride", "time (ms)", "useful GB/s",
           "sectors/warp", "predicted", "of peak");
    printf("  ----------------------------------------------------------------------------\n");

    double bwStride1 = 0.0;
    for (int stride : kStrides) {
        const size_t touched = n / static_cast<size_t>(stride);
        double ms = timeGpuMs(
            20, [&] { stridedRead<<<grid, block>>>(d_in, d_out, touched, stride); });

        const double useful = gbPerSec(usefulBytes(touched), ms);
        const double predicted =
            gbPerSec(predictedReadBytes(touched, stride) +
                         static_cast<double>(touched) * sizeof(float),
                     ms);
        if (stride == 1) bwStride1 = useful;

        printf("  %-8d %11.3f %13.1f %13.1f %11.1f %8.0f%%\n", stride, ms, useful,
               sectorsPerWarp(stride), predicted,
               peak > 0.0 ? 100.0 * useful / peak : 0.0);
    }

    printf("\n  Reading the table\n");
    printf("    stride 1  : 32 threads read 128 contiguous bytes = 4 sectors. Every\n");
    printf("                byte fetched is used, and you get ~90%% of peak.\n");
    printf("    stride 8  : neighbouring threads are 32 bytes apart, so each one\n");
    printf("                needs its own sector: 32 sectors instead of 4. Useful\n");
    printf("                bandwidth has collapsed to about an eighth.\n");
    printf("    stride >8 : nothing more is lost. You cannot waste more than one\n");
    printf("                whole sector per access, so the curve flattens.\n");
    printf("\n    The 'predicted' column stays near peak throughout, which is the\n");
    printf("    point: the memory system is working just as hard in every row. It\n");
    printf("    is moving data you throw away.\n");

    // ========================================================
    // Experiment 2: alignment
    // ========================================================
    printSection("Experiment 2 - alignment");
    printf("  Perfectly contiguous reads, just not starting on a sector boundary.\n\n");
    printf("  %-8s %11s %13s %11s\n", "offset", "time (ms)", "useful GB/s", "vs offset 0");
    printf("  ---------------------------------------------------\n");

    double bwOffset0 = 0.0;
    for (int offset : kOffsets) {
        const size_t touched = n - 64;
        double ms =
            timeGpuMs(20, [&] { offsetRead<<<grid, block>>>(d_in, d_out, touched, offset); });
        const double useful = gbPerSec(usefulBytes(touched), ms);
        if (offset == 0) bwOffset0 = useful;
        printf("  %-8d %11.3f %13.1f %10.2fx\n", offset, ms, useful,
               bwOffset0 > 0.0 ? useful / bwOffset0 : 0.0);
    }

    printf("\n  A warp still reads 128 consecutive bytes. They just no longer begin\n");
    printf("  on a 32-byte boundary, so they span 5 sectors instead of 4 - one extra\n");
    printf("  sector per warp, for a `+ offset` that looks completely innocent.\n");

    printf("\n  Look at WHICH offsets are fast. 8, 16 and 32 are back at full speed;\n");
    printf("  1, 2, 4 and 31 are not. 8 floats = 32 bytes = exactly one sector, so\n");
    printf("  any offset that is a multiple of 8 is still sector-aligned.\n");
    printf("  ALIGNMENT IS PERIODIC, NOT MONOTONIC: what matters is offset %% 8, not\n");
    printf("  how large the offset is.\n");

    printf("\n  The penalty is around 9%% here rather than the 25%% the sector count\n");
    printf("  suggests, because the extra sector of one warp is usually the FIRST\n");
    printf("  sector of the next warp, so L2 serves it. The effect is real but\n");
    printf("  partly absorbed - the same pattern you saw in Phase 2 exercise 06.\n");

    printf("\n  cudaMalloc returns 256-byte-aligned memory, so this only bites when\n");
    printf("  you offset into an array - a halo region, a tile origin, a sub-matrix.\n");
    printf("  When it matters, pad the row pitch (or use cudaMallocPitch) so every\n");
    printf("  row starts aligned.\n");

    // ========================================================
    // Experiment 3: the mapping that feels right and is not
    // ========================================================
    printSection("Experiment 3 - chunk per thread versus interleaved");
    {
        const int chunk = 32;
        const size_t threadsNeeded = n / chunk;
        const int gridChunk = static_cast<int>(ceilDiv(threadsNeeded, size_t(block)));

        double msChunk =
            timeGpuMs(10, [&] { blockPerThread<<<gridChunk, block>>>(d_in, d_out, n, chunk); });
        double msInter = timeGpuMs(10, [&] { interleaved<<<grid, block>>>(d_in, d_out, n); });

        ResultTable t;
        t.add("32-element chunk per thread", msChunk, usefulBytes(n));
        t.add("interleaved (grid-stride)", msInter, usefulBytes(n));
        t.print("Same work, two mappings");

        printf("\n  'Give each thread a contiguous chunk' is how you parallelise on a\n");
        printf("  CPU, where each core wants its own cache-friendly region. On a GPU\n");
        printf("  it is the worst possible mapping: at any instant the 32 threads of\n");
        printf("  a warp are each at the start of their own chunk, %d bytes apart.\n",
               chunk * 4);
        printf("  Cost here: %.2fx.\n", msChunk / msInter);
        printf("\n  This is the mistake to watch for when porting CPU code. The loop\n");
        printf("  structure that was right there is exactly wrong here.\n");
    }

    printSection("The rule");
    printf("  Adjacent THREADS must touch adjacent ADDRESSES.\n");
    printf("\n  Everything else in Phase 3 follows from working around cases where\n");
    printf("  you cannot arrange that directly:\n");
    printf("    - transpose (exercise 05): stage a tile in shared memory so both\n");
    printf("      the read and the write can be contiguous\n");
    printf("    - tiled matmul (exercise 04): same trick, for reuse as well\n");
    printf("    - histogram (exercise 09): privatise so the scatter happens in\n");
    printf("      shared memory instead of global\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    reportCheck("contiguous access reaches at least 50% of peak",
                peak <= 0.0 || bwStride1 > 0.5 * peak);
    return verifySummary();
}
