// ============================================================
// 03 - Vector addition, and the cost of getting data there [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void vectorAdd(const float* a, const float* b, float* c, size_t n) {
    // Grid-stride loop: works for any n with any launch config, and
    // consecutive threads still read consecutive addresses so the
    // access stays perfectly coalesced.
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printBanner("Phase 2 / 03 - Vector addition");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    printf("  n = %zu floats (%.0f MB per array, %.0f MB of traffic)\n", n,
           bytes / (1024.0 * 1024.0), vectorAddBytes(n) / (1024.0 * 1024.0));

    VectorProblem problem(n);
    std::vector<float> result(n, 0.0f);

    // ========================================================
    // Device memory
    // ========================================================
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, problem.a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, problem.b.data(), bytes, cudaMemcpyHostToDevice));

    // Size the grid to the GPU rather than to the data - the
    // grid-stride loop covers whatever is left over.
    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;

    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    printSection("Correctness");
    // Both sides do the same single addition in the same order, so
    // the results are bit-identical here. That is unusual; as soon
    // as a kernel reorders a sum (Phase 3/06) it stops being true.
    checkArray("c == a + b", result, problem.golden, 1e-6, 1e-6);

    // ========================================================
    // Three measurements, three different stories
    // ========================================================
    printSection("Performance");

    // 1. The CPU. Also memory bound - it moves the same 192 MB.
    double msCpu = timeCpuMs(5, [&] {
        vectorAddCPU(problem.a.data(), problem.b.data(), result.data(), n);
    });

    // 2. The kernel alone, data already on the device. This is the
    //    number people quote in blog posts.
    double msKernel = timeGpuMs(20, [&] { vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n); });

    // 3. The honest number: what it costs to actually use the GPU
    //    from data that lives in host memory.
    double msTotal = timeCpuMs(5, [&] {
        CUDA_CHECK(cudaMemcpy(d_a, problem.a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, problem.b.data(), bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    ResultTable table;
    table.add("CPU (single thread)", msCpu, vectorAddBytes(n), vectorAddFlops(n));
    table.add("GPU kernel only", msKernel, vectorAddBytes(n), vectorAddFlops(n));
    table.add("GPU + PCIe round trip", msTotal, vectorAddBytes(n), vectorAddFlops(n));
    table.print("Vector addition");

    // ========================================================
    // Reading the numbers
    // ========================================================
    printSection("What just happened");

    const double peakBw = theoreticalBandwidthGBs();
    const double kernelBw = gbPerSec(vectorAddBytes(n), msKernel);
    if (peakBw > 0.0) {
        printf("  Kernel efficiency : %.1f of %.1f GB/s = %.0f%% of theoretical peak\n",
               kernelBw, peakBw, 100.0 * kernelBw / peakBw);
        printf("  For a kernel this simple, 70-90%% of peak is the expected result.\n");
        printf("  There is nothing left to optimise: it is pure memory traffic.\n\n");
    }

    // Isolate the transfers so the cost is unambiguous.
    double msH2D = timeCpuMs(5, [&] {
        CUDA_CHECK(cudaMemcpy(d_a, problem.a.data(), bytes, cudaMemcpyHostToDevice));
    });
    double msD2H = timeCpuMs(5, [&] {
        CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    });

    printf("  %-32s %8.2f ms   %6.1f GB/s\n", "host -> device (64 MB)", msH2D,
           gbPerSec(bytes, msH2D));
    printf("  %-32s %8.2f ms   %6.1f GB/s\n", "device -> host (64 MB)", msD2H,
           gbPerSec(bytes, msD2H));
    printf("  %-32s %8.2f ms\n", "kernel", msKernel);
    printf("\n  PCIe is roughly %.0fx slower than the GPU's own memory. The transfers\n",
           kernelBw / gbPerSec(bytes, msH2D));
    printf("  cost about %.0fx what the computation does.\n",
           (msH2D * 2 + msD2H) / msKernel);

    printf("\n  Conclusion\n");
    if (msTotal > msCpu) {
        printf("    The GPU is %.2fx SLOWER than the CPU once transfers are counted,\n",
               msTotal / msCpu);
        printf("    even though the kernel itself is %.1fx faster. This is the correct\n",
               msCpu / msKernel);
        printf("    and expected result for vector addition.\n");
    } else {
        printf("    The GPU wins by %.2fx overall on this machine - you have a fast\n",
               msCpu / msTotal);
        printf("    PCIe link or a slow CPU. The ratio is what matters, not the sign.\n");
    }
    printf("\n    Arithmetic intensity here is %.3f FLOP/byte against a ridge point\n",
           vectorAddFlops(n) / vectorAddBytes(n));
    printf("    near 37. There is no version of this kernel that is compute bound.\n");
    printf("\n    So when IS a GPU worth it?\n");
    printf("      - when data stays resident and is reused across many kernels\n");
    printf("      - when arithmetic intensity is high (matmul: Phase 3/04)\n");
    printf("      - when transfers overlap with compute (streams: Phase 4/03)\n");
    printf("      - when several operations are fused into one pass (Phase 5/14)\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // reportCheck takes a plain string, not a printf format - a
    // literal percent sign needs no escaping here.
    reportCheck("kernel reached at least 30% of peak bandwidth",
                peakBw <= 0.0 || kernelBw > 0.3 * peakBw);
    return verifySummary();
}
