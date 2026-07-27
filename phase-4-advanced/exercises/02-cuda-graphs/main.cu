// ============================================================
// 02 - CUDA graphs  [STARTER]
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
    printBanner("Phase 4 / 02 - CUDA graphs (starter)");
    requireCudaDevice();

    printf("  %d kernels per iteration, %d iterations, %zu elements each\n",
           kKernelsPerIteration, kIterations, kElements);
    printf("  The kernels are deliberately tiny, so launch overhead matters.\n");

    // --------------------------------------------------------
    // TODO 1: the baseline - kKernelsPerIteration launches in a
    //         stream, repeated kIterations times
    // --------------------------------------------------------
    // Verify against chainCPU() before timing anything.
    printSection("Correctness");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 2: capture the same sequence into a graph
    // --------------------------------------------------------
    //   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    //   ... issue exactly the work you would normally issue ...
    //   cudaStreamEndCapture(stream, &graph);
    //   cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    //
    // Nothing executes during capture - the calls are recorded, not
    // run. Instantiation is where the driver validates the DAG, and
    // it is why replay is cheap.
    //
    // cudaGraphGetNodes() tells you how many nodes were captured;
    // print it and check it matches what you expect.
    printSection("Capture and replay");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: time both, and report the PER-LAUNCH cost
    // --------------------------------------------------------
    // Total milliseconds are not the interesting number here.
    // Divide by the number of kernel launches to get microseconds
    // per launch, and compare.
    //
    // Predict first: how many microseconds do you think a kernel
    // launch costs on the CPU side?
    printSection("Performance");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: reproduce the classic graph bug
    // --------------------------------------------------------
    // Allocate a SECOND buffer. Capture a graph that operates on
    // the first one, then replay it expecting the second to be
    // updated. Observe that nothing happens and no error is raised:
    // a graph captures pointers and values, not intent.
    //
    // Then fix it with cudaGraphExecKernelNodeSetParams(), which is
    // far cheaper than re-instantiating.

    printTodoNotice("implement the baseline, the capture, and the comparison");
    return verifySummary();
}
