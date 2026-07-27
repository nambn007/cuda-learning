// ============================================================
// 06 - Parallel reduction, seven ways  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// v1: interleaved addressing, divergent branch
// ------------------------------------------------------------
// `tid % (2 * stride) == 0` is true for a scattered subset of the
// warp, so the warp DIVERGES: the hardware executes the taken path
// with the other threads masked off, then continues. Half the lanes
// are idle in the first step, three quarters in the second, and so
// on - while the warp still occupies a full issue slot each time.
__global__ void reduce1(const float* in, float* out, size_t n) {
    extern __shared__ float s[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v2: interleaved addressing, no divergence
// ------------------------------------------------------------
// The active threads are now contiguous - threads 0..k work and the
// rest do not, so whole warps are either fully active or fully idle.
// But the index 2*stride*tid means the shared-memory stride is a
// power of two, which is precisely the bank-conflict pattern from
// exercise 03: gcd(2*stride, 32) grows with the stride.
__global__ void reduce2(const float* in, float* out, size_t n) {
    extern __shared__ float s[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        const unsigned index = 2 * stride * tid;
        if (index < blockDim.x) s[index] += s[index + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v3: sequential addressing
// ------------------------------------------------------------
// Halve the stride each step instead of doubling it. Active threads
// stay contiguous AND the shared-memory access is stride 1, so
// there are no bank conflicts at any step.
//
// What remains: on the very first iteration only half the threads
// do anything, and it gets worse from there.
__global__ void reduce3(const float* in, float* out, size_t n) {
    extern __shared__ float s[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v4: first add during load
// ------------------------------------------------------------
// Instead of loading one element and immediately idling half the
// threads, load TWO and add them. The grid halves, every thread
// does useful work on the first step, and the launch overhead
// halves too.
__global__ void reduce4(const float* in, float* out, size_t n) {
    extern __shared__ float s[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) * 2 + threadIdx.x;

    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v5: unroll the last warp with warp shuffles
// ------------------------------------------------------------
// Once only 32 threads remain, they are a single warp. The classic
// trick was to drop __syncthreads() and mark the array `volatile`,
// relying on implicit warp-lockstep - which has been UNSAFE since
// Volta, where threads in a warp can diverge independently.
//
// The modern equivalent is __shfl_down_sync: lane `l` reads the
// value held by lane `l + delta` directly from its register. No
// shared memory, no barrier, five instructions.
__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void reduce5(const float* in, float* out, size_t n) {
    extern __shared__ float s[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) * 2 + threadIdx.x;

    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    // Tree in shared memory down to one warp.
    for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // Last 32: registers only.
    if (tid < 32) {
        v = s[tid] + s[tid + 32];
        v = warpReduceSum(v);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

// ------------------------------------------------------------
// v6: grid-stride loop + full warp-shuffle reduction
// ------------------------------------------------------------
// The grid is sized to the GPU rather than to the data, so each
// thread sums many elements in a coalesced sweep before any
// reduction happens. Shared memory is then touched exactly once,
// to combine the per-warp partial sums.
//
// This is the shape a production reduction actually has, and it is
// what CUB's BlockReduce generates.
__global__ void reduce6(const float* in, float* out, size_t n) {
    __shared__ float warpSums[32];  // at most 1024/32 warps per block

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid % warpSize;
    const unsigned warp = tid / warpSize;

    // Coalesced grid-stride sweep.
    float v = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + tid; i < n; i += stride) {
        v += in[i];
    }

    // Reduce within each warp - no shared memory, no barrier.
    v = warpReduceSum(v);
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();

    // One warp combines the per-warp sums.
    if (warp == 0) {
        const unsigned warpCount = (blockDim.x + warpSize - 1) / warpSize;
        v = (lane < warpCount) ? warpSums[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ------------------------------------------------------------
// v7: grid-stride with four independent accumulators
// ------------------------------------------------------------
// v6 turns out to be SLOWER than v5, and the reason is the one you
// measured on the CPU in Phase 1 exercise 07: its sweep accumulates
// into a single variable, so hundreds of adds form one serial
// dependency chain. An FMA issues every cycle but takes several to
// complete, so one chain runs at a fraction of peak.
//
// The fix is identical to the CPU fix: several independent chains.
__global__ void reduce7(const float* in, float* out, size_t n) {
    __shared__ float warpSums[32];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid % warpSize;
    const unsigned warp = tid / warpSize;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + tid;

    // Four chains, each still perfectly coalesced across the warp.
    for (; i + 3 * stride < n; i += 4 * stride) {
        a0 += in[i];
        a1 += in[i + stride];
        a2 += in[i + 2 * stride];
        a3 += in[i + 3 * stride];
    }
    for (; i < n; i += stride) a0 += in[i];

    float v = (a0 + a1) + (a2 + a3);
    v = warpReduceSum(v);
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        const unsigned warpCount = (blockDim.x + warpSize - 1) / warpSize;
        v = (lane < warpCount) ? warpSums[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

int main() {
    printBanner("Phase 3 / 06 - Parallel reduction, seven ways");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const int block = kBlockSize;
    printf("  n = %zu floats (%.0f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

    std::vector<float> host(n);
    // Values around 1.0 so that a float accumulator does not lose
    // everything to rounding across 32M terms.
    fillRandom(host.data(), n, 0.5f, 1.5f, 101);
    const double golden = sumCPU(host.data(), n);
    printf("  Reference sum: %.4f\n", golden);

    const double peak = theoreticalBandwidthGBs();
    const double floorMs = peak > 0.0 ? reductionBytes(n) / (peak * 1e9) * 1e3 : 0.0;
    if (peak > 0.0)
        printf("  Bandwidth floor: %.3f ms at %.0f GB/s (this kernel is memory bound)\n",
               floorMs, peak);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int gridFull = static_cast<int>(ceilDiv(n, static_cast<size_t>(block)));
    const int gridHalf = static_cast<int>(ceilDiv(n, static_cast<size_t>(block) * 2));
    const int gridStride = prop.multiProcessorCount * 8;

    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(gridFull) * sizeof(float)));

    // Sums the per-block partials on the host, in double. All
    // variants pay the same cost, so the comparison stays fair.
    auto finish = [&](int gridSize) {
        std::vector<float> partial(gridSize);
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, gridSize * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return sumCPU(partial.data(), gridSize);
    };

    // Float addition is not associative, so every variant sums in a
    // different order and lands on a slightly different answer.
    // 1e-4 relative is generous for 32M terms.
    auto check = [&](const char* label, double got) {
        const double rel = std::fabs(got - golden) / std::fabs(golden);
        reportCheck(label, rel < 1e-4);
        return rel;
    };

    printSection("Correctness");
    const size_t sharedBytes = block * sizeof(float);

    reduce1<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v1 interleaved, divergent", finish(gridFull));

    reduce2<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v2 interleaved, no divergence", finish(gridFull));

    reduce3<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v3 sequential addressing", finish(gridFull));

    reduce4<<<gridHalf, block, sharedBytes>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v4 first add during load", finish(gridHalf));

    reduce5<<<gridHalf, block, sharedBytes>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v5 unrolled last warp (shuffle)", finish(gridHalf));

    reduce6<<<gridStride, block>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v6 grid-stride + warp shuffle", finish(gridStride));

    reduce7<<<gridStride, block>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    check("v7 grid-stride, 4 accumulators", finish(gridStride));

    printSection("Performance");
    const double bytes = reductionBytes(n);
    ResultTable table;
    table.add("v1 interleaved, divergent",
              timeGpuMs(10, [&] { reduce1<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n); }),
              bytes);
    table.add("v2 interleaved, no divergence",
              timeGpuMs(10, [&] { reduce2<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n); }),
              bytes);
    table.add("v3 sequential addressing",
              timeGpuMs(10, [&] { reduce3<<<gridFull, block, sharedBytes>>>(d_in, d_partial, n); }),
              bytes);
    table.add("v4 first add during load",
              timeGpuMs(10, [&] { reduce4<<<gridHalf, block, sharedBytes>>>(d_in, d_partial, n); }),
              bytes);
    table.add("v5 unrolled last warp",
              timeGpuMs(10, [&] { reduce5<<<gridHalf, block, sharedBytes>>>(d_in, d_partial, n); }),
              bytes);
    const double ms5 =
        timeGpuMs(10, [&] { reduce5<<<gridHalf, block, sharedBytes>>>(d_in, d_partial, n); });
    const double ms6 = timeGpuMs(10, [&] { reduce6<<<gridStride, block>>>(d_in, d_partial, n); });
    const double ms7 = timeGpuMs(10, [&] { reduce7<<<gridStride, block>>>(d_in, d_partial, n); });
    table.add("v6 grid-stride + shuffle", ms6, bytes);
    table.add("v7 grid-stride, 4 accumulators", ms7, bytes);
    table.print("Reduction over 32M floats");

    printSection("What each step fixed");
    printf("  v1 -> v2  removed WARP DIVERGENCE. `tid %% (2*s) == 0` is true for a\n");
    printf("            scattered subset of the warp, so both paths execute with\n");
    printf("            lanes masked off. Making the active threads contiguous means\n");
    printf("            whole warps are either fully busy or fully retired.\n");
    printf("\n  v2 -> v3  removed BANK CONFLICTS. Index 2*s*tid is a power-of-two\n");
    printf("            stride through shared memory - exactly the pattern from\n");
    printf("            exercise 03. Halving the stride instead makes it stride 1.\n");
    printf("\n  v3 -> v4  removed IDLE THREADS. In v3 half the block does nothing on\n");
    printf("            the very first step. Loading two elements and adding them\n");
    printf("            before the tree starts halves the grid and wastes nobody.\n");
    printf("\n  v4 -> v5  removed BARRIERS. The last 32 threads are one warp, so\n");
    printf("            __shfl_down_sync moves values between lanes' REGISTERS - no\n");
    printf("            shared memory and no __syncthreads() for the final 5 steps.\n");
    printf("\n  v5 -> v6  changed the SHAPE. A grid-stride sweep does most of the\n");
    printf("            summing in one long coalesced pass, so the tree runs once\n");
    printf("            per block instead of log2(n/blockSize) times overall.\n");

    if (ms6 > ms5) {
        printf("\n  ...and it came out %.2fx SLOWER than v5. That is the most useful\n",
               ms6 / ms5);
        printf("     result in this exercise, because the cause is one you have\n");
        printf("     already measured - on a CPU, in Phase 1 exercise 07.\n");
        printf("\n     v6's sweep accumulates into ONE variable. Each thread performs\n");
        printf("     roughly %zu adds, and every one waits for the previous result.\n",
               n / (static_cast<size_t>(gridStride) * block));
        printf("     An FMA issues every cycle but takes several to complete, so a\n");
        printf("     single dependency chain runs at a fraction of peak. The kernel\n");
        printf("     is latency bound, not bandwidth bound, and no amount of memory\n");
        printf("     tuning fixes that.\n");
        printf("\n  v6 -> v7  four INDEPENDENT accumulators, exactly the CPU fix:\n");
        printf("            %.2fx over v6, %.2fx over v5.\n", ms6 / ms7, ms5 / ms7);
    } else {
        printf("\n  v6 -> v7  four independent accumulators instead of one, to break\n");
        printf("            the dependency chain in the sweep: %.2fx.\n", ms6 / ms7);
    }

    printSection("How close to the floor?");
    if (peak > 0.0) {
        struct Entry {
            const char* name;
            double ms;
        };
        const Entry entries[] = {{"v5 unrolled last warp", ms5},
                                 {"v6 grid-stride", ms6},
                                 {"v7 grid-stride, 4 accumulators", ms7}};
        const Entry* best = &entries[0];
        for (const Entry& e : entries)
            if (e.ms < best->ms) best = &e;

        printf("  Best variant : %-32s %.3f ms  (%.1f GB/s, %.0f%% of peak)\n", best->name,
               best->ms, gbPerSec(bytes, best->ms),
               100.0 * gbPerSec(bytes, best->ms) / peak);
        printf("  Floor        : %-32s %.3f ms  (reading %.0f MB once at peak)\n", "",
               floorMs, bytes / (1024.0 * 1024.0));
        printf("\n  A reduction reads every input exactly once and writes almost\n");
        printf("  nothing, so its ceiling is a memory copy. Anything above ~85%% of\n");
        printf("  peak means the kernel is finished and further tuning is wasted.\n");
    }

    printf("\n  A note on the answers\n");
    printf("    Every variant sums in a different order, and float addition is not\n");
    printf("    associative, so they land on slightly different values - all equally\n");
    printf("    'correct'. If you need reproducibility across variants or machines,\n");
    printf("    you need a fixed summation order or compensated (Kahan) summation,\n");
    printf("    and both cost performance.\n");

    printf("\n  In production, use CUB (Phase 5 exercise 02). cub::BlockReduce emits\n");
    printf("  essentially v6, tuned per architecture. Write it once by hand to\n");
    printf("  understand it, then never again.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return verifySummary();
}
