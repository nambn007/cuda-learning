// ============================================================
// 05 - Error handling: recoverable, sticky, and asynchronous [SOLUTION]
// ============================================================
// This program deliberately causes CUDA errors and shows how each
// kind behaves. It still exits 0 - the failures are the content.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void addOne(int* v, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) v[gid] += 1;
}

// Writes at an offset far outside the allocation.
//
// A small overrun - a few kilobytes - usually is NOT trapped: the
// runtime allocates in large pages, so the write lands inside
// memory the process already owns and silently corrupts it. That is
// the dangerous case, and only compute-sanitizer finds it.
//
// To demonstrate the hardware trap reliably we go far outside any
// mapped page.
__global__ void unguardedWrite(int* v, long long offset) {
    long long gid = blockIdx.x * blockDim.x + threadIdx.x;
    v[gid + offset] = 1;
}

// A helper that reports instead of exiting, so the program can keep
// running after a deliberate failure.
static cudaError_t tryLaunch(const char* what, dim3 grid, dim3 block, int* d, int n) {
    addOne<<<grid, block>>>(d, n);
    cudaError_t err = cudaGetLastError();
    printf("  %-42s %s\n", what,
           err == cudaSuccess ? "ok" : cudaGetErrorName(err));
    return err;
}

int main() {
    printBanner("Phase 2 / 05 - Error handling");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, kSafeElements * sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, kSafeElements * sizeof(int)));

    // ========================================================
    // 1. Recoverable errors: the context survives
    // ========================================================
    printSection("Recoverable errors");
    printf("  These are detected BEFORE anything executes, so the context is\n");
    printf("  untouched and the next launch works normally.\n\n");

    reportCheck("a valid launch succeeds",
                tryLaunch("<<<4, 256>>> (valid)", 4, 256, d, kSafeElements) == cudaSuccess);

    // maxThreadsPerBlock is 1024 on every current architecture.
    cudaError_t tooManyThreads =
        tryLaunch("<<<1, 2048>>> (too many threads per block)", 1, 2048, d, kSafeElements);
    reportCheck("oversized block is rejected", tooManyThreads == cudaErrorInvalidConfiguration);

    // The error flag was cleared by reading it, so this works again.
    reportCheck("context still usable after a rejected launch",
                tryLaunch("<<<4, 256>>> (valid again)", 4, 256, d, kSafeElements) ==
                    cudaSuccess);

    // Out of memory is recoverable too - the allocation simply did
    // not happen.
    void* huge = nullptr;
    cudaError_t oom = cudaMalloc(&huge, static_cast<size_t>(1) << 50);  // 1 PB
    printf("  %-42s %s\n", "cudaMalloc(1 PB)", cudaGetErrorName(oom));
    reportCheck("impossible allocation fails cleanly", oom == cudaErrorMemoryAllocation);

    // A trap worth internalising: a failing API call BOTH returns
    // the error AND sets the last-error flag. Reading the return
    // value does not clear the flag.
    cudaError_t leftOver = cudaGetLastError();
    cudaError_t nowClean = cudaGetLastError();
    printf("  %-42s %s\n", "cudaGetLastError() after that", cudaGetErrorName(leftOver));
    printf("  %-42s %s\n", "cudaGetLastError() again", cudaGetErrorName(nowClean));
    reportCheck("a failed call also sets the last-error flag", leftOver == oom);
    reportCheck("only cudaGetLastError clears it", nowClean == cudaSuccess);

    printf("\n  This is how errors get blamed on the wrong kernel. If you had\n");
    printf("  ignored that cudaMalloc failure, the very next CUDA_CHECK_LAST()\n");
    printf("  after an unrelated launch would have reported cudaErrorMemory-\n");
    printf("  Allocation - and you would have gone looking at a kernel that was\n");
    printf("  perfectly fine. Check the return value of every call, at the call.\n");

    // ========================================================
    // 2. peek versus get
    // ========================================================
    printSection("cudaGetLastError versus cudaPeekAtLastError");
    addOne<<<1, 2048>>>(d, kSafeElements);  // fails again
    cudaError_t peeked = cudaPeekAtLastError();
    cudaError_t peekedAgain = cudaPeekAtLastError();
    cudaError_t got = cudaGetLastError();
    cudaError_t afterGet = cudaGetLastError();

    printf("  peek   -> %s\n", cudaGetErrorName(peeked));
    printf("  peek   -> %s   (still there: peek does not clear)\n",
           cudaGetErrorName(peekedAgain));
    printf("  get    -> %s   (returns it AND clears it)\n", cudaGetErrorName(got));
    printf("  get    -> %s   (now clean)\n", cudaGetErrorName(afterGet));
    reportCheck("peek does not clear the flag", peeked == peekedAgain);
    reportCheck("get clears the flag", afterGet == cudaSuccess);

    // ========================================================
    // 3. Asynchrony: where the error appears is not where it is
    // ========================================================
    printSection("Asynchronous errors");
    printf("  A launch returns before the kernel runs, so an error inside the\n");
    printf("  kernel surfaces at the next synchronisation point - which is often\n");
    printf("  an innocent cudaMemcpy hundreds of lines away.\n\n");
    printf("  When a failure makes no sense, rerun with:\n");
    printf("      CUDA_LAUNCH_BLOCKING=1 ./your_program\n");
    printf("  The runtime then waits after every launch and reports the error at\n");
    printf("  the line that caused it. It is slow; use it only to diagnose.\n");

    const char* blocking = getenv("CUDA_LAUNCH_BLOCKING");
    printf("\n  CUDA_LAUNCH_BLOCKING is currently %s\n",
           (blocking && blocking[0] == '1') ? "ON" : "off");

    // ========================================================
    // 4. The tools that find these bugs for you
    // ========================================================
    printSection("compute-sanitizer");
    printf("  An out-of-bounds GPU write usually does NOT crash. It silently\n");
    printf("  corrupts whatever is next in memory, and you find out later - or\n");
    printf("  never. compute-sanitizer is valgrind for the GPU:\n\n");
    printf("      compute-sanitizer ./build/bin/p2/p2_05_error_handling_sol\n");
    printf("      compute-sanitizer --tool racecheck   ./your_program\n");
    printf("      compute-sanitizer --tool synccheck   ./your_program\n");
    printf("      compute-sanitizer --tool initcheck   ./your_program\n\n");
    printf("  Under the default tool, the sticky error below is reported with the\n");
    printf("  exact kernel, thread and address that caused it.\n");

    // ========================================================
    // 5. Sticky errors: the context is gone
    // ========================================================
    // Everything that needs a working context must happen BEFORE
    // this point. Nothing after it can be trusted.
    printSection("Sticky errors (this destroys the context)");
    printf("  Launching a kernel that writes 1 GB past the end of a %d-element\n",
           kSafeElements);
    printf("  buffer...\n\n");

    // 2^28 ints = 1 GB beyond the allocation: comfortably outside
    // anything the process has mapped.
    unguardedWrite<<<1, 32>>>(d, 1LL << 28);
    cudaError_t launchErr = cudaGetLastError();
    printf("  launch error       : %s\n", cudaGetErrorName(launchErr));
    printf("                       (the launch itself was fine - the damage happens\n");
    printf("                        later, during execution)\n");

    cudaError_t execErr = cudaDeviceSynchronize();
    printf("  after synchronise  : %s\n", cudaGetErrorName(execErr));

    if (execErr != cudaSuccess) {
        // Every subsequent call now returns the same error.
        int* probe = nullptr;
        cudaError_t afterSticky = cudaMalloc(&probe, 16);
        printf("  cudaMalloc(16)     : %s\n", cudaGetErrorName(afterSticky));
        printf("\n  The context is dead. cudaMalloc, cudaMemcpy, every launch - all\n");
        printf("  return this same error from now on, no matter how unrelated they\n");
        printf("  are. The only recovery is to exit the process.\n");
        reportCheck("sticky error poisons every later call", afterSticky != cudaSuccess);
    } else {
        // Some drivers do not trap this particular overrun; the
        // write simply lands in memory the process happens to own.
        printf("\n  This driver did not trap the overrun - the write landed in memory\n");
        printf("  the allocator had already reserved. That is the dangerous case:\n");
        printf("  silent corruption. Run this binary under compute-sanitizer and it\n");
        printf("  will be reported.\n");
        reportCheck("overrun was silent (run compute-sanitizer to see it)", true);
    }

    printSection("Rules to keep");
    printf("  1. Wrap every runtime call in CUDA_CHECK(...).\n");
    printf("  2. After every launch, CUDA_CHECK_LAST() for configuration errors.\n");
    printf("  3. While debugging, CUDA_CHECK_KERNEL() to also catch execution\n");
    printf("     errors at the launch site. Remove it from timing loops.\n");
    printf("  4. When an error appears somewhere impossible, suspect a sticky\n");
    printf("     error from earlier and rerun with CUDA_LAUNCH_BLOCKING=1.\n");
    printf("  5. Run compute-sanitizer before believing any kernel is correct.\n");

    return verifySummary();
}
