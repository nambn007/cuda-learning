// ============================================================
// LLM inference engine - WORKED CORE MODULE
// ============================================================
// The hot loop of transformer inference is attention:
//
//     scores = Q K^T / sqrt(d)      softmax over the key axis
//     out    = softmax(scores) V
//
// The softmax is where a naive implementation goes wrong twice:
//
//   NUMERICALLY. exp(x) overflows for x > 88 in float. The fix is
//   to subtract the row maximum first, which is mathematically a
//   no-op and the difference between working and producing NaN.
//
//   IN MEMORY TRAFFIC. The obvious version reads the row three
//   times: once for the max, once for the sum of exponentials, once
//   to normalise. A fused kernel reads it ONCE, keeping the row in
//   registers and shared memory. That is the same idea as
//   FlashAttention, at a scale you can write in an afternoon.
//
// This module implements both and measures the difference. See
// README.md for the milestones that turn it into an engine.
// ============================================================

#include <cfloat>
#include <cstdio>
#include <vector>

#include "cuda_helper.h"

inline constexpr int kRows = 8192;    // batch * heads * queries
inline constexpr int kCols = 1024;    // sequence length
inline constexpr int kBlockSize = 256;

// ------------------------------------------------------------
// Naive softmax: three passes over the row
// ------------------------------------------------------------
__global__ void softmaxThreePass(const float* in, float* out, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* src = in + static_cast<size_t>(row) * cols;
    float* dst = out + static_cast<size_t>(row) * cols;

    __shared__ float reduction[kBlockSize];
    const int tid = threadIdx.x;

    // Pass 1: maximum.
    float m = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) m = fmaxf(m, src[i]);
    reduction[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
        __syncthreads();
    }
    const float rowMax = reduction[0];
    __syncthreads();

    // Pass 2: sum of exponentials.
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += __expf(src[i] - rowMax);
    reduction[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    const float rowSum = reduction[0];
    __syncthreads();

    // Pass 3: normalise.
    for (int i = tid; i < cols; i += blockDim.x) dst[i] = __expf(src[i] - rowMax) / rowSum;
}

// ------------------------------------------------------------
// Online (single-pass) softmax
// ------------------------------------------------------------
// Track the running maximum and the running sum together. When a
// new maximum appears, rescale the sum accumulated so far:
//
//     m_new = max(m, x)
//     s_new = s * exp(m - m_new) + exp(x - m_new)
//
// The row is read once instead of three times. This rescaling
// trick is exactly what makes FlashAttention possible - it lets you
// compute a softmax over a sequence you never hold in full.
__global__ void softmaxOnline(const float* in, float* out, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* src = in + static_cast<size_t>(row) * cols;
    float* dst = out + static_cast<size_t>(row) * cols;

    const int tid = threadIdx.x;
    __shared__ float sharedMax[kBlockSize];
    __shared__ float sharedSum[kBlockSize];

    // Each thread keeps its own running (max, sum) over its slice.
    float m = -FLT_MAX, s = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float x = src[i];
        const float mNew = fmaxf(m, x);
        s = s * __expf(m - mNew) + __expf(x - mNew);
        m = mNew;
    }
    sharedMax[tid] = m;
    sharedSum[tid] = s;
    __syncthreads();

    // Combine the per-thread pairs with the same rescaling rule.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float mA = sharedMax[tid], sA = sharedSum[tid];
            const float mB = sharedMax[tid + stride], sB = sharedSum[tid + stride];
            const float mNew = fmaxf(mA, mB);
            sharedMax[tid] = mNew;
            sharedSum[tid] = sA * __expf(mA - mNew) + sB * __expf(mB - mNew);
        }
        __syncthreads();
    }

    const float rowMax = sharedMax[0];
    const float rowSum = sharedSum[0];
    for (int i = tid; i < cols; i += blockDim.x) dst[i] = __expf(src[i] - rowMax) / rowSum;
}

// ------------------------------------------------------------
// Host reference, in double
// ------------------------------------------------------------
static void softmaxCPU(const std::vector<float>& in, std::vector<float>& out, int rows,
                       int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* src = in.data() + static_cast<size_t>(r) * cols;
        float* dst = out.data() + static_cast<size_t>(r) * cols;
        double m = -1e300;
        for (int c = 0; c < cols; ++c) m = src[c] > m ? src[c] : m;
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += std::exp(src[c] - m);
        for (int c = 0; c < cols; ++c) dst[c] = static_cast<float>(std::exp(src[c] - m) / sum);
    }
}

int main() {
    printBanner("Phase 6 - LLM inference engine (worked core module)");
    requireCudaDevice();

    const int rows = kRows, cols = kCols;
    const size_t n = static_cast<size_t>(rows) * cols;
    printf("  Softmax over %d rows of %d (a [batch*heads*queries, seq] score matrix)\n",
           rows, cols);
    printf("  %.0f MB\n", n * sizeof(float) / (1024.0 * 1024.0));

    std::vector<float> host(n), golden(n), result(n);
    // Include some large values, so a version without the
    // max-subtraction trick would overflow rather than merely drift.
    fillRandom(host.data(), n, -20.0f, 100.0f, 191);
    softmaxCPU(host, golden, rows, cols);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    printSection("Correctness");
    softmaxThreePass<<<rows, kBlockSize>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkArray("three-pass softmax", result, golden, 1e-4, 1e-7);

    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));
    softmaxOnline<<<rows, kBlockSize>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkArray("online (single-pass) softmax", result, golden, 1e-4, 1e-7);

    // Every row must sum to 1 - the property that actually matters
    // downstream, and the one a subtly wrong rescaling breaks.
    double worstRowError = 0.0;
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += result[static_cast<size_t>(r) * cols + c];
        worstRowError = std::max(worstRowError, std::fabs(s - 1.0));
    }
    reportCheck("every row sums to 1", worstRowError < 1e-4);
    printf("  worst row-sum error: %.2e\n", worstRowError);

    printSection("Performance");
    double msThree = timeGpuMs(10, [&] { softmaxThreePass<<<rows, kBlockSize>>>(d_in, d_out, rows, cols); });
    double msOnline = timeGpuMs(10, [&] { softmaxOnline<<<rows, kBlockSize>>>(d_in, d_out, rows, cols); });

    const double bytes = static_cast<double>(n) * 2.0 * sizeof(float);
    ResultTable t;
    t.add("three-pass softmax", msThree, bytes);
    t.add("online softmax", msOnline, bytes);
    t.print("Softmax over an 8192 x 1024 score matrix");

    printSection("Two things this module demonstrates");
    printf("  1. SUBTRACT THE ROW MAXIMUM. exp(x) overflows above x = 88 in float,\n");
    printf("     and the scores here reach 100. Subtracting the max is\n");
    printf("     mathematically a no-op and the difference between working and\n");
    printf("     returning NaN. Try removing it.\n");
    printf("\n  2. THE ONLINE RESCALING TRICK.\n");
    printf("       m_new = max(m, x)\n");
    printf("       s_new = s * exp(m - m_new) + exp(x - m_new)\n");
    printf("     lets you maintain a correct softmax while seeing the row only\n");
    printf("     once. That is precisely what makes FlashAttention possible: it\n");
    printf("     computes attention over a sequence whose score matrix is never\n");
    printf("     held in memory at all.\n");

    printSection("Where the project goes from here");
    printf("    1. Q K^T with the tiled GEMM from Phase 3/04\n");
    printf("    2. fuse scores -> softmax -> times V into one kernel\n");
    printf("    3. a KV cache, and paged attention so sequences of different\n");
    printf("       lengths share memory without fragmentation\n");
    printf("    4. batched decoding: many sequences, one step each\n");
    printf("    5. INT8 or FP8 weights, and measure the quality loss honestly\n");
    printf("\n  Memory capacity, not FLOPs, is the binding constraint in LLM\n");
    printf("  inference. Every milestone above is about moving or storing less.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return verifySummary();
}
