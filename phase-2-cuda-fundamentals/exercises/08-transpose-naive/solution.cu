// ============================================================
// 08 - Naive transpose: the kernel you cannot coalesce  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// The speed of light for this problem
// ------------------------------------------------------------
// A plain copy moves exactly the same number of bytes as a
// transpose: one read and one write per element. Everything below
// is measured as a fraction of this.
__global__ void copyKernel(const float* in, float* out, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        const size_t i = static_cast<size_t>(row) * cols + col;
        out[i] = in[i];
    }
}

// ------------------------------------------------------------
// Coalesced read, scattered write
// ------------------------------------------------------------
// Within a warp `col` varies, so:
//   in [row * cols + col]  -> 32 consecutive floats  -> coalesced
//   out[col * rows + row]  -> 32 addresses `rows` apart -> 32 separate
//                             sectors, 4 useful bytes from each
__global__ void transposeReadCoalesced(const float* in, float* out, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[static_cast<size_t>(col) * rows + row] = in[static_cast<size_t>(row) * cols + col];
    }
}

// ------------------------------------------------------------
// Scattered read, coalesced write
// ------------------------------------------------------------
// The mirror image. Which of the two is faster is a hardware
// question, not a logical one: writes can be buffered and
// coalesced in the write-back path, reads cannot be predicted, so
// on most architectures it is better to scatter the READS.
__global__ void transposeWriteCoalesced(const float* in, float* out, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[static_cast<size_t>(col) * rows + row] = in[static_cast<size_t>(row) * cols + col];
    }
}

int main() {
    printBanner("Phase 2 / 08 - Naive transpose");
    requireCudaDevice();

    const int rows = kRows, cols = kCols;
    TransposeProblem problem(rows, cols);
    std::vector<float> result(problem.elements());
    const size_t bytes = problem.bytes();

    printf("  Matrix: %d x %d (%.0f MB), traffic: %.0f MB\n", rows, cols,
           bytes / (1024.0 * 1024.0),
           transposeBytes(rows, cols) / (1024.0 * 1024.0));

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, problem.in.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(32, 8);  // 256 threads, 32 wide so a warp spans one row
    const dim3 gridColMajor(ceilDiv(cols, static_cast<int>(block.x)),
                            ceilDiv(rows, static_cast<int>(block.y)));
    const dim3 gridRowMajor(ceilDiv(rows, static_cast<int>(block.x)),
                            ceilDiv(cols, static_cast<int>(block.y)));

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transposeReadCoalesced<<<gridColMajor, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    checkArray("coalesced read / scattered write", result, problem.golden, 0.0, 0.0);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transposeWriteCoalesced<<<gridRowMajor, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    checkArray("scattered read / coalesced write", result, problem.golden, 0.0, 0.0);

    // ========================================================
    // Performance, measured against a plain copy
    // ========================================================
    printSection("Performance");

    double msCopy =
        timeGpuMs(20, [&] { copyKernel<<<gridColMajor, block>>>(d_in, d_out, rows, cols); });
    double msReadC = timeGpuMs(
        20, [&] { transposeReadCoalesced<<<gridColMajor, block>>>(d_in, d_out, rows, cols); });
    double msWriteC = timeGpuMs(
        20, [&] { transposeWriteCoalesced<<<gridRowMajor, block>>>(d_in, d_out, rows, cols); });
    double msCpu = timeCpuMs(3, [&] {
        transposeCPU(problem.in.data(), result.data(), rows, cols);
    });

    const double traffic = transposeBytes(rows, cols);
    ResultTable table;
    table.add("copy (speed of light)", msCopy, traffic);
    table.add("transpose, coalesced read", msReadC, traffic);
    table.add("transpose, coalesced write", msWriteC, traffic);
    table.add("CPU transpose", msCpu, traffic);
    table.print("Transpose versus copy, 4096 x 4096");

    // ========================================================
    // Interpretation
    // ========================================================
    printSection("Reading the result");

    const double bwCopy = gbPerSec(traffic, msCopy);
    const double bwReadC = gbPerSec(traffic, msReadC);
    const double bwWriteC = gbPerSec(traffic, msWriteC);
    const double peak = theoreticalBandwidthGBs();

    printf("  %-32s %7.1f GB/s", "copy", bwCopy);
    if (peak > 0.0) printf("  (%3.0f%% of DRAM peak)", 100.0 * bwCopy / peak);
    printf("\n");
    printf("  %-32s %7.1f GB/s   %3.0f%% of copy\n", "transpose, coalesced read", bwReadC,
           100.0 * bwReadC / bwCopy);
    printf("  %-32s %7.1f GB/s   %3.0f%% of copy\n", "transpose, coalesced write", bwWriteC,
           100.0 * bwWriteC / bwCopy);

    printf("\n  Why transpose cannot reach copy speed\n");
    printf("    A transpose moves exactly the same bytes as a copy - one read and\n");
    printf("    one write per element, no arithmetic. It should be identical.\n");
    printf("\n    But out[col][row] = in[row][col] means a warp varying one index\n");
    printf("    is necessarily contiguous on one side and scattered on the other:\n");
    printf("      vary col -> read 32 consecutive floats, write 32 addresses\n");
    printf("                  `rows` apart\n");
    printf("      vary row -> the mirror image\n");
    printf("    The scatter is not a mistake in the indexing. The scatter IS the\n");
    printf("    transpose. No rearrangement of these two lines removes it.\n");

    printf("\n  Given that one side must scatter, which side?\n");
    if (msWriteC < msReadC) {
        printf("    Scattering the READS and coalescing the WRITES is %.2fx faster\n",
               msReadC / msWriteC);
        printf("    here. The asymmetry is real and worth understanding:\n");
        printf("      - a scattered READ can hit in L1/L2. A neighbouring warp may\n");
        printf("        already have pulled that sector in, and the latency of the\n");
        printf("        ones that miss is hidden by other resident warps.\n");
        printf("      - a scattered WRITE has no such escape. It must eventually\n");
        printf("        reach DRAM, and a partial-sector write wastes the rest of\n");
        printf("        that sector's bandwidth with no reuse to recover it.\n");
        printf("    So when you cannot coalesce both sides, coalesce the writes.\n");
    } else {
        printf("    Scattering the WRITES came out %.2fx faster on this run, which\n",
               msWriteC / msReadC);
        printf("    is the opposite of the usual result - normally a scattered read\n");
        printf("    is cheaper because it can hit in L2, while a scattered write\n");
        printf("    must reach DRAM. Rerun it; if it holds, the cause is worth\n");
        printf("    investigating with ncu rather than assuming.\n");
    }

    printf("\n  The real fix\n");
    printf("    Give both sides something contiguous to talk to: stage a tile in\n");
    printf("    __shared__ memory, read it coalesced, transpose WITHIN shared\n");
    printf("    memory (which has no coalescing requirement), then write it out\n");
    printf("    coalesced. That is Phase 3 exercise 05, and it recovers most of\n");
    printf("    the gap to copy speed - once you also deal with the bank\n");
    printf("    conflicts it introduces (Phase 3 exercise 03).\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    reportCheck("copy is faster than either transpose",
                msCopy < msReadC && msCopy < msWriteC);
    reportCheck("both transposes beat the CPU", msReadC < msCpu && msWriteC < msCpu);
    return verifySummary();
}
