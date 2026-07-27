// ============================================================
// 02 - CUDA graphs  [SOLUTION]
// ============================================================
// NOTE: this exercise has not yet been run on the reference RTX
// 3060. The code builds and the logic is checked, but the README
// deliberately contains no measured numbers - run it and fill them
// in for your own machine.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void step(float* v, size_t n) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < n; i += stride) v[i] = v[i] * 1.0001f + 0.5f;
}

int main() {
    printBanner("Phase 4 / 02 - CUDA graphs");
    requireCudaDevice();

    const size_t n = kElements;
    const size_t bytes = n * sizeof(float);
    const int block = 256;
    const int grid = static_cast<int>(ceilDiv(n, static_cast<size_t>(block)));

    printf("  %d kernels per iteration, %d iterations, %zu elements each\n",
           kKernelsPerIteration, kIterations, n);
    printf("  The kernels are deliberately tiny, so per-launch overhead matters.\n");

    std::vector<float> host(n), golden(n);
    fillRandom(host.data(), n, 0.0f, 1.0f, 131);
    golden = host;
    chainCPU(golden.data(), n, kKernelsPerIteration);

    float* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto reset = [&] {
        CUDA_CHECK(cudaMemcpy(d_v, host.data(), bytes, cudaMemcpyHostToDevice));
    };
    auto fetch = [&](std::vector<float>& out) {
        CUDA_CHECK(cudaMemcpy(out.data(), d_v, bytes, cudaMemcpyDeviceToHost));
    };

    // ========================================================
    // 1. Plain launches
    // ========================================================
    printSection("Correctness");
    std::vector<float> result(n);

    reset();
    for (int k = 0; k < kKernelsPerIteration; ++k) step<<<grid, block, 0, stream>>>(d_v, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fetch(result);
    checkArray("plain launches", result, golden, 1e-4, 1e-5);

    // ========================================================
    // 2. Capture the same sequence into a graph
    // ========================================================
    // Put the stream into capture mode, issue exactly the work you
    // would normally issue, and end capture. Nothing executes
    // during capture - the calls are recorded, not run.
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < kKernelsPerIteration; ++k) step<<<grid, block, 0, stream>>>(d_v, n);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate once: the driver validates the whole DAG here, so
    // that replaying it later costs almost nothing.
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    size_t nodeCount = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodeCount));
    printf("  Captured graph has %zu nodes\n", nodeCount);

    reset();
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fetch(result);
    checkArray("graph replay", result, golden, 1e-4, 1e-5);

    // ========================================================
    // 3. Timing
    // ========================================================
    printSection("Performance");

    double msPlain = timeCpuMs(5, [&] {
        for (int it = 0; it < kIterations; ++it)
            for (int k = 0; k < kKernelsPerIteration; ++k)
                step<<<grid, block, 0, stream>>>(d_v, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }, 1);

    double msGraph = timeCpuMs(5, [&] {
        for (int it = 0; it < kIterations; ++it) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }, 1);

    const int totalLaunches = kIterations * kKernelsPerIteration;
    ResultTable table;
    table.add("individual launches", msPlain);
    table.add("graph replay", msGraph);
    table.print("20 kernels x 200 iterations");

    printf("\n  Per-kernel launch cost\n");
    printf("    individual : %.2f us\n", msPlain * 1000.0 / totalLaunches);
    printf("    via graph  : %.2f us\n", msGraph * 1000.0 / totalLaunches);
    printf("    saved      : %.2f us per launch\n",
           (msPlain - msGraph) * 1000.0 / totalLaunches);

    printf("\n  A graph replaces %d driver round trips with %d, because the whole\n",
           kKernelsPerIteration, 1);
    printf("  DAG was validated once at instantiation time. The GPU work is\n");
    printf("  identical - what disappears is the CPU sitting in the driver between\n");
    printf("  kernels while the GPU waits.\n");

    printf("\n  When graphs pay off\n");
    printf("    - many SMALL kernels (single-digit microseconds each)\n");
    printf("    - the same sequence repeated many times\n");
    printf("    - inference, iterative solvers, physics steps\n");
    printf("  When they do not\n");
    printf("    - a few large kernels: launch overhead is already invisible\n");
    printf("    - a sequence that changes every iteration: you pay to rebuild\n");

    printSection("The classic graph bug");
    printf("  A graph captures POINTERS AND VALUES, not intent. If the next\n");
    printf("  iteration should operate on a different buffer, replaying the same\n");
    printf("  graph will happily rerun on the OLD one - silently, with no error.\n");
    printf("\n  Two ways out:\n");
    printf("    - keep the pointers stable (double-buffer into fixed slots), or\n");
    printf("    - update the node parameters:\n");
    printf("        cudaGraphExecKernelNodeSetParams(graphExec, node, &params)\n");
    printf("      which is far cheaper than re-instantiating the graph.\n");

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_v));

    reportCheck("graph replay is not slower than individual launches",
                msGraph <= msPlain * 1.05);
    return verifySummary();
}
