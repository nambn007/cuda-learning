// ============================================================
// 03 - Atomics, contention and memory ordering  [SOLUTION]
// ============================================================
// NOTE: not yet run on the reference RTX 3060. The code builds and
// the logic is reviewed, but the README carries no measured
// numbers - run it and fill in your own.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ============================================================
// 1. Contention and privatisation
// ============================================================

// Every thread hits one of 256 global counters. With 16M threads
// and a skewed distribution, thousands contend for the same address.
__global__ void histogramGlobal(const unsigned char* v, size_t n, unsigned int* bins) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        atomicAdd(&bins[v[i]], 1u);
    }
}

// Privatised: each block keeps its own copy of the histogram in
// shared memory. Contention is now within a block (fast, on chip)
// and the global atomics drop from one per element to kBins per
// block.
__global__ void histogramPrivatised(const unsigned char* v, size_t n, unsigned int* bins) {
    __shared__ unsigned int local[kBins];

    for (int i = threadIdx.x; i < kBins; i += blockDim.x) local[i] = 0;
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        atomicAdd(&local[v[i]], 1u);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
        if (local[i] != 0) atomicAdd(&bins[i], local[i]);
    }
}

// ============================================================
// 2. Building an atomic that does not exist: float max
// ============================================================
// There is no atomicMax for float. Build it from compare-and-swap.
//
// The loop retries until nobody else changed the value between the
// read and the write. `__float_as_uint` / `__uint_as_float` let us
// use the 32-bit integer CAS on float bits without converting.
__device__ float atomicMaxFloat(float* addr, float value) {
    unsigned int* asUint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *asUint, assumed;
    do {
        assumed = old;
        const float current = __uint_as_float(assumed);
        if (current >= value) break;  // somebody already wrote something larger
        old = atomicCAS(asUint, assumed, __float_as_uint(value));
    } while (assumed != old);
    return __uint_as_float(old);
}

__global__ void maxKernel(const float* v, size_t n, float* result) {
    __shared__ float blockMax;
    if (threadIdx.x == 0) blockMax = -INFINITY;
    __syncthreads();

    float local = -INFINITY;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        if (v[i] > local) local = v[i];
    }

    // Reduce within the block first, so only one atomic per block
    // reaches global memory - the same privatisation idea.
    atomicMaxFloat(&blockMax, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicMaxFloat(result, blockMax);
}

// ============================================================
// 3. Memory ordering: a single-pass reduction
// ============================================================
// Each block writes its partial sum, then atomically increments a
// counter. The LAST block to finish (the one that sees the counter
// reach gridDim.x - 1) reduces all the partials itself, so the
// whole reduction happens in one kernel launch.
//
// The subtle part: writing partial[blockIdx.x] and then
// atomicAdd(counter, 1) does NOT guarantee that the partial is
// visible to the last block. Atomicity is not visibility.
// __threadfence() is what orders the two - it makes every prior
// write visible device-wide before anything after it.
//
// Remove the fence and this kernel produces the right answer almost
// always, which is exactly what makes the bug expensive.
__device__ unsigned int retirementCount = 0;

__global__ void singlePassReduce(const float* v, size_t n, float* partial, float* result) {
    __shared__ float shared[kBlockSize];
    __shared__ bool isLastBlock;

    const unsigned tid = threadIdx.x;

    float sum = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + tid; i < n; i += stride)
        sum += v[i];

    shared[tid] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];

        // Make that write visible to every other block BEFORE the
        // counter increment that tells them it is ready.
        __threadfence();

        const unsigned int ticket = atomicAdd(&retirementCount, 1u);
        isLastBlock = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (isLastBlock) {
        float total = 0.0f;
        for (unsigned i = tid; i < gridDim.x; i += blockDim.x) total += partial[i];
        shared[tid] = total;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            *result = shared[0];
            retirementCount = 0;  // reset for the next launch
        }
    }
}

int main() {
    printBanner("Phase 4 / 03 - Atomics, contention and memory ordering");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t n = kElements;
    const int block = kBlockSize;
    const int grid = prop.multiProcessorCount * 4;

    // ========================================================
    // 1. Histogram: contention and privatisation
    // ========================================================
    printSection("Contention: global versus privatised atomics");
    {
        std::vector<unsigned char> host(n);
        // Skewed towards a few bins, so the contention is real.
        std::mt19937 gen(141);
        std::normal_distribution<float> dist(128.0f, 20.0f);
        for (size_t i = 0; i < n; ++i) {
            int v = static_cast<int>(dist(gen));
            host[i] = static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }

        std::vector<unsigned int> golden(kBins);
        histogramCPU(host.data(), n, golden.data());

        unsigned char* d_v = nullptr;
        unsigned int* d_bins = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, n));
        CUDA_CHECK(cudaMalloc(&d_bins, kBins * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_v, host.data(), n, cudaMemcpyHostToDevice));

        std::vector<unsigned int> result(kBins);
        auto run = [&](const char* label, auto kernel) {
            CUDA_CHECK(cudaMemset(d_bins, 0, kBins * sizeof(unsigned int)));
            kernel<<<grid, block>>>(d_v, n, d_bins);
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaMemcpy(result.data(), d_bins, kBins * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
            checkArrayExact(label, result.data(), golden.data(), kBins);
        };

        run("global atomics", histogramGlobal);
        run("privatised atomics", histogramPrivatised);

        double msGlobal = timeGpuMs(10, [&] {
            CUDA_CHECK(cudaMemsetAsync(d_bins, 0, kBins * sizeof(unsigned int)));
            histogramGlobal<<<grid, block>>>(d_v, n, d_bins);
        });
        double msPriv = timeGpuMs(10, [&] {
            CUDA_CHECK(cudaMemsetAsync(d_bins, 0, kBins * sizeof(unsigned int)));
            histogramPrivatised<<<grid, block>>>(d_v, n, d_bins);
        });

        ResultTable t;
        t.add("one global atomic per element", msGlobal, static_cast<double>(n));
        t.add("shared-memory privatisation", msPriv, static_cast<double>(n));
        t.print("256-bin histogram over 16M bytes");

        printf("\n  Global atomics per launch\n");
        printf("    naive      : %zu   (one per element)\n", n);
        printf("    privatised : %d   (at most kBins per block)\n", grid * kBins);
        printf("\n  Privatisation is the single most useful atomic technique. It\n");
        printf("  turns global contention into shared-memory contention, which is\n");
        printf("  both faster per operation and spread over far fewer operations.\n");

        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_bins));
    }

    // ========================================================
    // 2. atomicCAS: building an atomic that does not exist
    // ========================================================
    printSection("atomicCAS: a float maximum");
    {
        std::vector<float> host(n);
        fillRandom(host.data(), n, -100.0f, 100.0f, 142);
        const float golden = maxCPU(host.data(), n);

        float *d_v = nullptr, *d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_v, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        const float negInf = -INFINITY;
        CUDA_CHECK(cudaMemcpy(d_result, &negInf, sizeof(float), cudaMemcpyHostToDevice));
        maxKernel<<<grid, block>>>(d_v, n, d_result);
        CUDA_CHECK_LAST();

        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        reportCheck("atomicCAS-based float max", got == golden);
        printf("  max = %.6f (CPU: %.6f)\n", got, golden);

        printf("\n  Only a handful of atomics exist in hardware. Everything else is\n");
        printf("  a compare-and-swap retry loop:\n");
        printf("      do { assumed = old;\n");
        printf("           old = atomicCAS(addr, assumed, f(assumed)); }\n");
        printf("      while (assumed != old);\n");
        printf("  Correct, but it retries under contention - so reduce within the\n");
        printf("  block first and let one thread do the global CAS.\n");

        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_result));
    }

    // ========================================================
    // 3. Memory ordering
    // ========================================================
    printSection("Memory ordering: __threadfence and a single-pass reduction");
    {
        std::vector<float> host(n);
        fillRandom(host.data(), n, 0.5f, 1.5f, 143);
        const double golden = sumCPU(host.data(), n);

        float *d_v = nullptr, *d_partial = nullptr, *d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_v, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        singlePassReduce<<<grid, block>>>(d_v, n, d_partial, d_result);
        CUDA_CHECK_LAST();

        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        reportCheck("single-pass reduction",
                    std::fabs(got - golden) / golden < 1e-3);
        printf("  sum = %.4f (CPU: %.4f)\n", got, golden);

        printf("\n  How it works: every block writes its partial sum, then bumps a\n");
        printf("  counter. The block that sees the counter reach gridDim.x - 1 knows\n");
        printf("  it is last, and reduces all the partials itself - so the whole\n");
        printf("  reduction takes ONE launch instead of two.\n");

        printf("\n  Why __threadfence() is there\n");
        printf("    ATOMICITY IS NOT VISIBILITY. atomicAdd guarantees the counter is\n");
        printf("    correct. It says nothing about whether partial[blockIdx.x],\n");
        printf("    written on the line before, is visible to the block that reads\n");
        printf("    it afterwards. __threadfence() makes every prior write visible\n");
        printf("    device-wide before anything after it.\n");
        printf("\n    Delete the fence and this kernel will still produce the right\n");
        printf("    answer almost every time, on your GPU, at this block count.\n");
        printf("    That is exactly what makes the bug expensive: it surfaces on\n");
        printf("    somebody else's hardware, months later, under load.\n");
        printf("\n    __threadfence_block()  visible within the block\n");
        printf("    __threadfence()        visible device-wide\n");
        printf("    __threadfence_system() visible to the host and peer GPUs too\n");

        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_result));
    }

    return verifySummary();
}
