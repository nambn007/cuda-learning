// ============================================================
// 07 - Warp-level primitives  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

static constexpr unsigned kFullMask = 0xffffffffu;

// ------------------------------------------------------------
// Butterfly reduction: every lane ends with the total
// ------------------------------------------------------------
// __shfl_down_sync (exercise 06) leaves the answer only in lane 0.
// __shfl_xor_sync exchanges symmetrically, so after 5 steps EVERY
// lane holds the sum. Same instruction count, more useful result,
// and no `if (lane == 0)` afterwards.
__device__ __forceinline__ float warpAllReduceSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(kFullMask, v, offset);
    }
    return v;
}

// ------------------------------------------------------------
// Warp-wide inclusive scan in 5 shuffles
// ------------------------------------------------------------
// Lane i adds the value from lane (i - offset), if that lane
// exists. This is the Hillis-Steele scan (exercise 10) done
// entirely in registers.
__device__ __forceinline__ float warpInclusiveScan(float v) {
    const unsigned lane = threadIdx.x % warpSize;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float n = __shfl_up_sync(kFullMask, v, offset);
        if (lane >= static_cast<unsigned>(offset)) v += n;
    }
    return v;
}

__global__ void scanKernel(const float* in, float* out, size_t n) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = warpInclusiveScan(in[i]);
}

// ------------------------------------------------------------
// Block reduction built from warp reductions
// ------------------------------------------------------------
__global__ void blockReduceKernel(const float* in, float* out, size_t n) {
    __shared__ float warpSums[32];
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned warp = threadIdx.x / warpSize;

    float v = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride)
        v += in[i];

    v = warpAllReduceSum(v);
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        const unsigned warpCount = (blockDim.x + warpSize - 1) / warpSize;
        v = (lane < warpCount) ? warpSums[lane] : 0.0f;
        v = warpAllReduceSum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ------------------------------------------------------------
// Filtering: the naive way
// ------------------------------------------------------------
// Every passing thread performs its own atomicAdd on the same
// counter. Within a warp that is up to 32 atomics to one address,
// and the hardware serialises them.
__global__ void filterNaive(const float* in, int* counter, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        if (in[i] > 0.75f) atomicAdd(counter, 1);
    }
}

// ------------------------------------------------------------
// Filtering: warp-aggregated atomics
// ------------------------------------------------------------
// __ballot_sync collects the predicate from all 32 lanes into one
// 32-bit word. __popc counts the set bits, so ONE lane can add the
// whole warp's contribution with a single atomic.
//
// 32 contended atomics become 1. This is the standard pattern for
// any kernel that appends to a shared output buffer, and it is what
// makes stream compaction (exercise 12) affordable.
__global__ void filterWarpAggregated(const float* in, int* counter, size_t n) {
    const unsigned lane = threadIdx.x % warpSize;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        const bool keep = (i < n) && (in[i] > 0.75f);

        // One bit per lane. __activemask() rather than a hard-coded
        // full mask, because the grid-stride loop can leave lanes
        // behind on the final iteration.
        const unsigned active = __activemask();
        const unsigned mask = __ballot_sync(active, keep);
        const int count = __popc(mask);

        // The lowest active lane does the atomic for everyone.
        const unsigned leader = __ffs(active) - 1;
        if (lane == leader && count > 0) atomicAdd(counter, count);
    }
}

// ------------------------------------------------------------
// Vote functions
// ------------------------------------------------------------
// __all_sync / __any_sync answer a question about the whole warp in
// one instruction. Useful for early exit: if no lane in the warp
// has work, skip the expensive path entirely.
__global__ void voteDemo(const float* in, int* flags, size_t n) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;

    const bool p = in[i] > 0.75f;
    const unsigned active = __activemask();
    const int allTrue = __all_sync(active, p) ? 1 : 0;
    const int anyTrue = __any_sync(active, p) ? 1 : 0;
    const unsigned ballot = __ballot_sync(active, p);

    // Only lane 0 of each warp records the warp-wide answers.
    if ((threadIdx.x % warpSize) == 0) {
        const size_t warpId = i / warpSize;
        flags[warpId * 3 + 0] = allTrue;
        flags[warpId * 3 + 1] = anyTrue;
        flags[warpId * 3 + 2] = __popc(ballot);
    }
}

int main() {
    printBanner("Phase 3 / 07 - Warp-level primitives");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const int block = kBlockSize;
    const int grid = prop.multiProcessorCount * 8;

    printf("  n = %zu floats, warp size %d\n", n, prop.warpSize);

    std::vector<float> host(n);
    fillRandom(host.data(), n, 0.0f, 1.0f, 111);
    const size_t keptGolden = countKeptCPU(host.data(), n);
    printf("  %zu of %zu elements pass the filter (%.1f%%)\n", keptGolden, n,
           100.0 * static_cast<double>(keptGolden) / static_cast<double>(n));

    float *d_in = nullptr, *d_out = nullptr;
    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================
    // Warp scan
    // ========================================================
    printSection("Warp-wide inclusive scan (__shfl_up_sync)");
    {
        const int scanGrid = static_cast<int>(ceilDiv(n, static_cast<size_t>(block)));
        scanKernel<<<scanGrid, block>>>(d_in, d_out, n);
        CUDA_CHECK_LAST();

        std::vector<float> result(n), golden(n);
        CUDA_CHECK(cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        warpScanCPU(host.data(), golden.data(), n);
        checkArray("inclusive scan within each warp", result, golden, 1e-5, 1e-6);

        printf("  Five __shfl_up_sync steps, no shared memory, no barriers.\n");
        printf("  This is the Hillis-Steele scan of exercise 10, done in registers.\n");
    }

    // ========================================================
    // Reduction
    // ========================================================
    printSection("Reduction via __shfl_xor_sync");
    {
        std::vector<float> partial(grid);
        blockReduceKernel<<<grid, block>>>(d_in, d_out, n);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(partial.data(), d_out, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double got = 0.0;
        for (int i = 0; i < grid; ++i) got += partial[i];
        double golden = 0.0;
        for (size_t i = 0; i < n; ++i) golden += host[i];
        reportCheck("block reduction is correct",
                    std::fabs(got - golden) / golden < 1e-4);

        printf("  __shfl_xor_sync exchanges symmetrically, so after 5 steps EVERY\n");
        printf("  lane holds the sum - unlike __shfl_down_sync, which leaves it only\n");
        printf("  in lane 0. Same instruction count, and no `if (lane == 0)` needed\n");
        printf("  when every thread wants the result.\n");
    }

    // ========================================================
    // Vote functions
    // ========================================================
    printSection("Vote functions");
    {
        const size_t warps = n / 32;
        int* d_flags = nullptr;
        CUDA_CHECK(cudaMalloc(&d_flags, warps * 3 * sizeof(int)));
        const int voteGrid = static_cast<int>(ceilDiv(n, static_cast<size_t>(block)));
        voteDemo<<<voteGrid, block>>>(d_in, d_flags, n);
        CUDA_CHECK_LAST();

        std::vector<int> flags(warps * 3);
        CUDA_CHECK(cudaMemcpy(flags.data(), d_flags, warps * 3 * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Check the first warp by hand.
        int expectAll = 1, expectAny = 0, expectCount = 0;
        for (int i = 0; i < 32; ++i) {
            const bool p = keepPredicate(host[i]);
            if (!p) expectAll = 0;
            if (p) expectAny = 1;
            if (p) ++expectCount;
        }
        reportCheck("__all_sync", flags[0] == expectAll);
        reportCheck("__any_sync", flags[1] == expectAny);
        reportCheck("__ballot_sync + __popc", flags[2] == expectCount);

        printf("  __all_sync / __any_sync answer a question about the whole warp in\n");
        printf("  one instruction - useful for skipping an expensive path entirely\n");
        printf("  when no lane needs it.\n");
        CUDA_CHECK(cudaFree(d_flags));
    }

    // ========================================================
    // The practical win: warp-aggregated atomics
    // ========================================================
    printSection("Warp-aggregated atomics");
    {
        auto runFilter = [&](auto kernel) {
            CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
            kernel<<<grid, block>>>(d_in, d_counter, n);
            CUDA_CHECK_LAST();
            int count = 0;
            CUDA_CHECK(cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
            return static_cast<size_t>(count);
        };

        reportCheck("naive filter counts correctly", runFilter(filterNaive) == keptGolden);
        reportCheck("aggregated filter counts correctly",
                    runFilter(filterWarpAggregated) == keptGolden);

        double msNaive = timeGpuMs(10, [&] {
            CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));
            filterNaive<<<grid, block>>>(d_in, d_counter, n);
        });
        double msAgg = timeGpuMs(10, [&] {
            CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));
            filterWarpAggregated<<<grid, block>>>(d_in, d_counter, n);
        });

        ResultTable t;
        t.add("one atomic per passing thread", msNaive,
              static_cast<double>(n) * sizeof(float));
        t.add("one atomic per warp (ballot)", msAgg,
              static_cast<double>(n) * sizeof(float));
        t.print("Counting elements that pass a filter");

        printf("\n  __ballot_sync collects the predicate from all 32 lanes into one\n");
        printf("  32-bit word; __popc counts the set bits; the lowest active lane\n");
        printf("  performs a SINGLE atomicAdd for the whole warp. Up to 32 contended\n");
        printf("  atomics on one address become 1.\n");

        printf("\n  Measured: %.2fx. In other words, NO GAIN.\n", msNaive / msAgg);

        printf("\n  The reason is worth more than the technique. Disassemble the\n");
        printf("  NAIVE kernel:\n");
        printf("      cuobjdump -sass build/bin/p3/p3_07_warp_shuffle_sol\n");
        printf("  and you will find, inside filterNaive:\n");
        printf("      REDUX.SUM UR10, R4              <- warp-wide sum, one instruction\n");
        printf("      ISETP.EQ.U32.AND P1, PT, ...    <- pick a single lane\n");
        printf("      @P1 RED.E.ADD.STRONG.GPU [...]  <- ONE atomic for the warp\n");
        printf("\n  nvcc already performed the aggregation. It has done so since CUDA\n");
        printf("  9, and on Ampere it uses the hardware REDUX.SUM instruction, which\n");
        printf("  is faster than the __ballot_sync + __popc sequence written by hand.\n");
        printf("  The hand-written version is not wrong - it is redundant.\n");

        printf("\n  So the lesson is not 'aggregate your atomics'. It is:\n");
        printf("    READ THE SASS BEFORE HAND-OPTIMISING.\n");
        printf("  A well-known technique from a 2017 blog post can be dead code by\n");
        printf("  2020 because the compiler absorbed it. Checking costs one command.\n");

        printf("\n  When manual aggregation still earns its keep:\n");
        printf("    - the compiler cannot see the pattern (the increment is\n");
        printf("      data-dependent, or the atomic hides behind a function call)\n");
        printf("    - you need each lane's OFFSET, not just the total - a warp scan\n");
        printf("      over the ballot gives every passing thread its own output slot\n");
        printf("      in one atomic, which is exactly how stream compaction works\n");
        printf("      (exercise 12)\n");
        printf("    - you are aggregating something other than a counter\n");
    }

    printSection("On the _sync suffix");
    printf("  The mask argument is not decoration. Before Volta a warp always moved\n");
    printf("  in lockstep and the old intrinsics (__shfl, __ballot) needed no mask.\n");
    printf("  Since Volta, threads in a warp diverge independently, so you must\n");
    printf("  state which lanes are expected to participate.\n");
    printf("\n  Use 0xffffffff when the whole warp is provably active, and\n");
    printf("  __activemask() when it may not be - inside a grid-stride loop whose\n");
    printf("  last iteration leaves some lanes behind, for example.\n");
    printf("  A mask that does not match reality is undefined behaviour, and the\n");
    printf("  symptom is a wrong answer that appears only under load.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_counter));
    return verifySummary();
}
