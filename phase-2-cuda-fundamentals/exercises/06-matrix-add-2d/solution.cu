// ============================================================
// 06 - 2D matrix addition, and the axis order  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// The right way: threadIdx.x indexes the COLUMN
// ------------------------------------------------------------
// Threads in a warp differ in threadIdx.x first, so within a warp
// `col` varies and `row` does not. Consecutive threads therefore
// read consecutive addresses, and the hardware merges the whole
// warp's request into a handful of transactions.
__global__ void matrixAddRowMajor(const float* a, const float* b, float* c, int rows,
                                  int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        const size_t i = static_cast<size_t>(row) * cols + col;
        c[i] = a[i] + b[i];
    }
}

// ------------------------------------------------------------
// The wrong way: threadIdx.x indexes the ROW
// ------------------------------------------------------------
// Identical arithmetic, identical results. But now consecutive
// threads within a warp are `cols` floats apart in memory, so each
// one drags in its own 32-byte sector and 31/32 of every fetch is
// discarded.
__global__ void matrixAddColumnMajor(const float* a, const float* b, float* c, int rows,
                                     int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;  // <- swapped
    const int col = blockIdx.y * blockDim.y + threadIdx.y;  // <- swapped
    if (row < rows && col < cols) {
        const size_t i = static_cast<size_t>(row) * cols + col;
        c[i] = a[i] + b[i];
    }
}

// ------------------------------------------------------------
// The 2D shape was never necessary
// ------------------------------------------------------------
// Row-major storage is one contiguous array, so a flat grid-stride
// loop does the same work. Launch a 2D grid when the KERNEL needs
// (row, col) - a stencil, a transpose - not because the data is
// drawn as a rectangle.
__global__ void matrixAddFlat(const float* a, const float* b, float* c, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) c[i] = a[i] + b[i];
}

int main() {
    printBanner("Phase 2 / 06 - 2D matrix addition");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int rows = kRows, cols = kCols;
    MatrixProblem problem(rows, cols);
    std::vector<float> result(problem.elements());
    const size_t bytes = problem.bytes();

    printf("  Matrix: %d x %d (%.0f MB each, %.0f MB of traffic)\n", rows, cols,
           bytes / (1024.0 * 1024.0), matrixAddBytes(rows, cols) / (1024.0 * 1024.0));

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, problem.a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, problem.b.data(), bytes, cudaMemcpyHostToDevice));

    // 16x16 = 256 threads. Multiple of 32, and square so that 2D
    // neighbourhoods stay inside one block for later stencil work.
    const dim3 block(16, 16);
    const dim3 gridRowMajor(ceilDiv(cols, static_cast<int>(block.x)),
                            ceilDiv(rows, static_cast<int>(block.y)));
    // The swapped kernel needs the grid swapped too, or it would not
    // cover the matrix.
    const dim3 gridColumnMajor(ceilDiv(rows, static_cast<int>(block.x)),
                               ceilDiv(cols, static_cast<int>(block.y)));

    printf("  Launch : grid(%u, %u) x block(%u, %u)\n", gridRowMajor.x, gridRowMajor.y,
           block.x, block.y);

    // ========================================================
    // Correctness: all three must agree
    // ========================================================
    printSection("Correctness");

    auto runAndCheck = [&](const char* label, auto launch) {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        checkArray(label, result, problem.golden, 1e-6, 1e-6);
    };

    runAndCheck("row-major mapping (x = column)", [&] {
        matrixAddRowMajor<<<gridRowMajor, block>>>(d_a, d_b, d_c, rows, cols);
    });
    runAndCheck("column-major mapping (x = row)", [&] {
        matrixAddColumnMajor<<<gridColumnMajor, block>>>(d_a, d_b, d_c, rows, cols);
    });
    runAndCheck("flat 1D grid-stride", [&] {
        matrixAddFlat<<<prop.multiProcessorCount * 8, 256>>>(d_a, d_b, d_c,
                                                            problem.elements());
    });

    printf("\n  All three produce identical results. Now look at what they cost.\n");

    // ========================================================
    // Performance
    // ========================================================
    printSection("Performance");

    double msCpu = timeCpuMs(3, [&] {
        matrixAddCPU(problem.a.data(), problem.b.data(), result.data(), rows, cols);
    });
    double msRow = timeGpuMs(20, [&] {
        matrixAddRowMajor<<<gridRowMajor, block>>>(d_a, d_b, d_c, rows, cols);
    });
    double msCol = timeGpuMs(20, [&] {
        matrixAddColumnMajor<<<gridColumnMajor, block>>>(d_a, d_b, d_c, rows, cols);
    });
    double msFlat = timeGpuMs(20, [&] {
        matrixAddFlat<<<prop.multiProcessorCount * 8, 256>>>(d_a, d_b, d_c,
                                                            problem.elements());
    });

    const double traffic = matrixAddBytes(rows, cols);
    ResultTable table;
    table.add("CPU (single thread)", msCpu, traffic);
    table.add("GPU 2D, x = column", msRow, traffic);
    table.add("GPU 2D, x = row (swapped)", msCol, traffic);
    table.add("GPU 1D flat grid-stride", msFlat, traffic);
    table.print("Matrix addition, 4096 x 4096");

    // ========================================================
    // Interpretation
    // ========================================================
    printSection("What the axis order cost");

    const double peak = theoreticalBandwidthGBs();
    const double bwRow = gbPerSec(traffic, msRow);
    const double bwCol = gbPerSec(traffic, msCol);

    printf("  x = column : %7.1f GB/s", bwRow);
    if (peak > 0.0) printf("  (%.0f%% of peak)", 100.0 * bwRow / peak);
    printf("\n  x = row    : %7.1f GB/s", bwCol);
    if (peak > 0.0) printf("  (%.0f%% of peak)", 100.0 * bwCol / peak);
    printf("\n\n  Swapping two lines cost %.1fx.\n", msCol / msRow);

    printf("\n  Why\n");
    printf("    Threads are linearised with x fastest, so one warp of 32 threads\n");
    printf("    covers threadIdx.x = 0..15 for two values of threadIdx.y.\n");
    printf("\n    x = column : those 32 threads span 16 consecutive columns in two\n");
    printf("                 adjacent rows - two runs of 64 contiguous bytes. The\n");
    printf("                 memory system serves that in a handful of sectors.\n");
    printf("    x = row    : the 16 varying threads are now %d floats apart, so the\n",
           cols);
    printf("                 warp touches 16 SEPARATE 32-byte sectors and uses only\n");
    printf("                 8 bytes of each. It requests about 4x the traffic.\n");

    printf("\n    Measured penalty here: %.1fx, not 4x. The difference is L2. This\n",
           msCol / msRow);
    printf("    GPU has %d KB of it, and neighbouring blocks re-read the sectors a\n",
           prop.l2CacheSize / 1024);
    printf("    previous block already pulled in, so the cache absorbs much of the\n");
    printf("    waste. Change the block shape to 256x1 and the cache stops helping;\n");
    printf("    Phase 3 exercise 01 does exactly that and measures the full effect.\n");

    printf("\n    The point stands: both versions are CORRECT, nothing warns you,\n");
    printf("    and one is %.0f%% slower for the sake of two swapped lines. This is\n",
           100.0 * (msCol / msRow - 1.0));
    printf("    the most common reason a first CUDA kernel disappoints.\n");

    printf("\n  On 2D launches in general\n");
    printf("    The flat 1D kernel is just as fast as the good 2D one, because\n");
    printf("    row-major storage is one contiguous array either way. Use a 2D\n");
    printf("    grid when the KERNEL needs (row, col) - stencils, transposes,\n");
    printf("    convolutions - not because the data is drawn as a rectangle.\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    reportCheck("coalesced mapping beats the swapped one", msRow < msCol);
    reportCheck("flat 1D is competitive with 2D", msFlat < msRow * 1.25);
    return verifySummary();
}
