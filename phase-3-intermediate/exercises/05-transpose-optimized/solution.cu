// ============================================================
// 05 - Transpose with a shared-memory tile  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// Baselines
// ------------------------------------------------------------
// A plain copy: the speed of light for this problem, since it moves
// exactly the same bytes.
__global__ void copyKernel(const float* in, float* out, int width, int height) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < width && y + j < height) {
            const size_t i = static_cast<size_t>(y + j) * width + x;
            out[i] = in[i];
        }
    }
}

// Phase 2 exercise 08's version: coalesced read, scattered write.
__global__ void transposeNaive(const float* in, float* out, int width, int height) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < width && y + j < height) {
            out[static_cast<size_t>(x) * height + (y + j)] =
                in[static_cast<size_t>(y + j) * width + x];
        }
    }
}

// ------------------------------------------------------------
// Shared-memory staging, WITHOUT padding
// ------------------------------------------------------------
// Both global accesses are now coalesced - the transpose happens on
// chip. But step 4 walks a COLUMN of the shared tile:
//
//   tile[threadIdx.x][threadIdx.y + j]
//     address = tx * 32 + (ty + j)
//     bank    = (ty + j) % 32   -> identical for all 32 threads
//
// A 32-way bank conflict, exactly as in exercise 03. The shared
// memory penalty has replaced the global one.
__global__ void transposeSharedNoPad(const float* in, float* out, int width, int height) {
    __shared__ float tile[kTile][kTile];

    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;

    // Coalesced read: consecutive tx -> consecutive addresses.
    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < width && y + j < height)
            tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * width + x];
    }
    __syncthreads();

    // The output block is at the transposed position in the grid.
    x = blockIdx.y * kTile + threadIdx.x;
    y = blockIdx.x * kTile + threadIdx.y;

    // Coalesced write, transposed read from shared.
    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < height && y + j < width)
            out[static_cast<size_t>(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ------------------------------------------------------------
// Shared-memory staging, WITH one padding column
// ------------------------------------------------------------
//   tile[threadIdx.x][threadIdx.y + j]
//     address = tx * 33 + (ty + j)
//     bank    = (tx + ty + j) % 32   -> all 32 banks
//
// 128 bytes of padding per tile, and the conflict is gone.
__global__ void transposeSharedPadded(const float* in, float* out, int width, int height) {
    __shared__ float tile[kTile][kTile + 1];

    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;

    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < width && y + j < height)
            tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * kTile + threadIdx.x;
    y = blockIdx.x * kTile + threadIdx.y;

    for (int j = 0; j < kTile; j += kBlockRows) {
        if (x < height && y + j < width)
            out[static_cast<size_t>(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main() {
    printBanner("Phase 3 / 05 - Transpose with a shared-memory tile");
    requireCudaDevice();

    const int rows = kRows, cols = kCols;
    TransposeProblem problem(rows, cols);
    std::vector<float> result(problem.elements());
    const size_t bytes = problem.bytes();

    printf("  Matrix: %d x %d (%.0f MB), tile %dx%d, block %dx%d\n", rows, cols,
           bytes / (1024.0 * 1024.0), kTile, kTile, kTile, kBlockRows);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, problem.in.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTile, kBlockRows);
    const dim3 grid(ceilDiv(cols, kTile), ceilDiv(rows, kTile));

    auto run = [&](const char* label, auto launch) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        checkArray(label, result, problem.golden, 0.0, 0.0);
    };

    printSection("Correctness");
    run("naive transpose", [&] { transposeNaive<<<grid, block>>>(d_in, d_out, cols, rows); });
    run("shared, no padding",
        [&] { transposeSharedNoPad<<<grid, block>>>(d_in, d_out, cols, rows); });
    run("shared, padded",
        [&] { transposeSharedPadded<<<grid, block>>>(d_in, d_out, cols, rows); });

    printSection("Performance");
    double msCopy = timeGpuMs(20, [&] { copyKernel<<<grid, block>>>(d_in, d_out, cols, rows); });
    double msNaive =
        timeGpuMs(20, [&] { transposeNaive<<<grid, block>>>(d_in, d_out, cols, rows); });
    double msNoPad =
        timeGpuMs(20, [&] { transposeSharedNoPad<<<grid, block>>>(d_in, d_out, cols, rows); });
    double msPadded =
        timeGpuMs(20, [&] { transposeSharedPadded<<<grid, block>>>(d_in, d_out, cols, rows); });

    const double traffic = transposeBytes(rows, cols);
    ResultTable table;
    table.add("copy (speed of light)", msCopy, traffic);
    table.add("transpose naive", msNaive, traffic);
    table.add("transpose shared, no padding", msNoPad, traffic);
    table.add("transpose shared, padded", msPadded, traffic);
    table.print("Transpose, 4096 x 4096");

    printSection("As a fraction of copy speed");
    const double bwCopy = gbPerSec(traffic, msCopy);
    printf("  %-32s %7.1f GB/s  %5.0f%%\n", "copy", bwCopy, 100.0);
    printf("  %-32s %7.1f GB/s  %5.0f%%\n", "naive", gbPerSec(traffic, msNaive),
           100.0 * msCopy / msNaive);
    printf("  %-32s %7.1f GB/s  %5.0f%%\n", "shared, no padding",
           gbPerSec(traffic, msNoPad), 100.0 * msCopy / msNoPad);
    printf("  %-32s %7.1f GB/s  %5.0f%%\n", "shared, padded", gbPerSec(traffic, msPadded),
           100.0 * msCopy / msPadded);

    printf("\n  What each step bought\n");
    printf("    naive -> shared          %.2fx\n", msNaive / msNoPad);
    printf("    shared -> shared padded  %.2fx\n", msNoPad / msPadded);
    printf("    naive -> shared padded   %.2fx\n", msNaive / msPadded);

    printf("\n  Staging in shared memory is the big win (%.2fx). Both global\n",
           msNaive / msNoPad);
    printf("  accesses become coalesced, and the scatter that could not be removed\n");
    printf("  by any index rearrangement now happens on chip, where there are no\n");
    printf("  coalescing rules at all.\n");

    printf("\n  The unpadded version DOES have a 32-way bank conflict:\n");
    printf("      tile[threadIdx.x][threadIdx.y + j]\n");
    printf("        address = tx * 32 + (ty + j)\n");
    printf("        bank    = (ty + j) %% 32   -> the SAME for all 32 threads\n");
    printf("  and padding to a row pitch of 33 makes the bank (tx + ty + j) %% 32,\n");
    printf("  spreading across all 32 banks for 128 bytes per tile.\n");

    printf("\n  But look at what that is actually worth here: %.2fx, taking the\n",
           msNoPad / msPadded);
    printf("  kernel from %.0f%% to %.0f%% of copy speed.\n", 100.0 * msCopy / msNoPad,
           100.0 * msCopy / msPadded);

    printf("\n  Compare that with exercise 03, where the same 32-way conflict cost\n");
    printf("  15x. The difference is what the kernel is BOUND BY. Exercise 03 was\n");
    printf("  pure shared-memory traffic in a tight loop, so every serialised cycle\n");
    printf("  showed up in the total. This kernel moves 128 MB through DRAM, and\n");
    printf("  the extra shared-memory cycles hide behind that latency.\n");

    printf("\n  A BANK CONFLICT ONLY COSTS YOU WHEN SHARED MEMORY IS THE BOTTLENECK.\n");
    printf("  Pad anyway - 128 bytes for 6%% is an easy trade, and the moment you\n");
    printf("  optimise the global side further the conflict stops being hidden.\n");
    printf("  But measure before assuming it is your problem.\n");

    printSection("Why it still does not reach copy speed");
    printf("  The padded version reaches %.0f%% of copy. The rest is structural:\n",
           100.0 * msCopy / msPadded);
    printf("    - two __syncthreads() per tile that a copy does not need\n");
    printf("    - the write goes to a different 4096-float row for each tile column,\n");
    printf("      so the DRAM page locality of a straight copy is lost\n");
    printf("    - shared memory limits how many blocks fit per SM\n");
    printf("  Getting closer means diagonal block reordering, wider tiles, or\n");
    printf("  vectorised loads - and at some point, not transposing at all: many\n");
    printf("  algorithms can consume a matrix in either orientation if you let them.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    reportCheck("padding helps the shared version", msPadded < msNoPad);
    reportCheck("padded shared beats the naive transpose", msPadded < msNaive);
    return verifySummary();
}
