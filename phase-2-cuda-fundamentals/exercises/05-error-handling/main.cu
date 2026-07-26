// ============================================================
// 05 - Error handling: recoverable, sticky, and asynchronous [STARTER]
// ============================================================
// Unusually for this repository, the goal here is to MAKE CUDA
// fail, on purpose, and observe how each kind of failure behaves.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void addOne(int* v, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) v[gid] += 1;
}

// Deliberately unguarded - `n` will be a lie.
__global__ void unguardedWrite(int* v, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) v[gid] = gid;
}

int main() {
    printBanner("Phase 2 / 05 - Error handling (starter)");
    requireCudaDevice();

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, kSafeElements * sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, kSafeElements * sizeof(int)));

    // --------------------------------------------------------
    // TODO 1: trigger a RECOVERABLE error and prove it recovers
    // --------------------------------------------------------
    // Launch addOne with more than 1024 threads per block. Then:
    //   - read the error with cudaGetLastError()
    //   - check it is cudaErrorInvalidConfiguration
    //   - launch a VALID kernel afterwards and confirm it works
    //
    // Do NOT use CUDA_CHECK_LAST() here - it calls exit(). Read the
    // error into a variable yourself.
    printSection("Recoverable errors");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 2: show the difference between peek and get
    // --------------------------------------------------------
    // Trigger a failure, then call cudaPeekAtLastError() twice and
    // cudaGetLastError() twice. Which one clears the flag?
    printSection("cudaGetLastError versus cudaPeekAtLastError");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 3: cause a STICKY error and observe the aftermath
    // --------------------------------------------------------
    // Call unguardedWrite with an `n` far larger than the
    // allocation, so threads write past the end. Then:
    //   - check cudaGetLastError() right after the launch. Is it
    //     an error yet? Why not?
    //   - call cudaDeviceSynchronize() and check again
    //   - try any unrelated call, such as cudaMalloc(&p, 16), and
    //     look at what it returns
    //
    // Put this LAST. Once the context is poisoned, nothing else in
    // the program will work.
    printSection("Sticky errors");
    printf("  TODO\n");

    // --------------------------------------------------------
    // TODO 4: run this binary under compute-sanitizer
    // --------------------------------------------------------
    //     compute-sanitizer ./build/bin/p2/p2_05_error_handling
    // and read what it says about the out-of-bounds write.

    printTodoNotice("implement the three experiments in main.cu");
    return verifySummary();
}
