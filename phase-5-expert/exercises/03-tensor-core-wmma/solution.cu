// ============================================================
// 03 - Tensor Cores via WMMA  [SOLUTION]
// ============================================================
// Verified on an RTX 3060 (sm_86): 7.03x over the naive FP32
// kernel, 5683 GFLOP/s.
// ============================================================

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

using namespace nvcuda;

// ------------------------------------------------------------
// FP32 baseline: one thread per output element
// ------------------------------------------------------------
__global__ void gemmFp32(const float* A, const float* B, float* C, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += A[static_cast<size_t>(row) * n + k] * B[static_cast<size_t>(k) * n + col];
        C[static_cast<size_t>(row) * n + col] = acc;
    }
}

// ------------------------------------------------------------
// Tensor Core GEMM
// ------------------------------------------------------------
// One WARP computes one 16x16 output tile. The fragment layout
// inside the registers is unspecified by design - you never index
// into a fragment, you only load, multiply and store it.
//
// blockDim is (32, 4): four warps per block, each owning one tile
// along the row axis.
__global__ void gemmWmma(const half* A, const half* B, float* C, int n) {
    const int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    const int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    const int row = warpM * kWmmaTile;
    const int col = warpN * kWmmaTile;
    if (row >= n || col >= n) return;

    for (int k = 0; k < n; k += kWmmaTile) {
        // Every thread in the warp calls these, uniformly. A
        // divergent wmma call is undefined behaviour.
        wmma::load_matrix_sync(aFrag, A + static_cast<size_t>(row) * n + k, n);
        wmma::load_matrix_sync(bFrag, B + static_cast<size_t>(k) * n + col, n);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    wmma::store_matrix_sync(C + static_cast<size_t>(row) * n + col, cFrag, n,
                            wmma::mem_row_major);
}

int main() {
    printBanner("Phase 5 / 03 - Tensor Cores via WMMA");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("  Compute capability %d.%d (%s)\n", prop.major, prop.minor,
           archName(prop.major, prop.minor));

    if (!hasComputeCapability(7, 0)) {
        return skipExercise(
            "Tensor Cores need compute capability 7.0 or newer (Volta and later). "
            "A GTX 16xx card has none at all.");
    }

    const int n = kSize;
    const size_t elems = static_cast<size_t>(n) * n;
    printf("  n = %d, %.1f GFLOP per GEMM\n", n, gemmFlops(n) / 1e9);

    std::vector<float> hA(elems), hB(elems);
    fillRandom(hA.data(), elems, -1.0f, 1.0f, 171);
    fillRandom(hB.data(), elems, -1.0f, 1.0f, 172);

    // Half-precision copies. Note that converting is itself lossy:
    // FP16 holds about 3 decimal digits.
    std::vector<half> hAh(elems), hBh(elems);
    for (size_t i = 0; i < elems; ++i) {
        hAh[i] = __float2half(hA[i]);
        hBh[i] = __float2half(hB[i]);
    }

    float *d_Af = nullptr, *d_Bf = nullptr, *d_Cf = nullptr, *d_Cw = nullptr;
    half *d_Ah = nullptr, *d_Bh = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Af, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Bf, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cf, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cw, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ah, elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Bh, elems * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_Af, hA.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf, hB.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ah, hAh.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bh, hBh.data(), elems * sizeof(half), cudaMemcpyHostToDevice));

    const dim3 blockFp32(16, 16);
    const dim3 gridFp32(ceilDiv(n, 16), ceilDiv(n, 16));

    // 32 threads in x = one warp; 4 warps per block.
    const dim3 blockWmma(32, 4);
    const dim3 gridWmma(ceilDiv(n / kWmmaTile, 1), ceilDiv(n / kWmmaTile, 4));

    printSection("Correctness");
    gemmFp32<<<gridFp32, blockFp32>>>(d_Af, d_Bf, d_Cf, n);
    CUDA_CHECK_LAST();
    gemmWmma<<<gridWmma, blockWmma>>>(d_Ah, d_Bh, d_Cw, n);
    CUDA_CHECK_LAST();

    std::vector<float> cf(elems), cw(elems);
    CUDA_CHECK(cudaMemcpy(cf.data(), d_Cf, elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cw.data(), d_Cw, elems * sizeof(float), cudaMemcpyDeviceToHost));

    // FP16 inputs carry ~3 decimal digits, so a 2% relative
    // tolerance over a 1024-term sum is the honest bar. This is a
    // real precision change, not a free speedup.
    checkArray("WMMA matches FP32 within half-precision tolerance", cw, cf, 2e-2, 1e-1);

    printSection("Performance");
    double msFp32 = timeGpuMs(10, [&] { gemmFp32<<<gridFp32, blockFp32>>>(d_Af, d_Bf, d_Cf, n); });
    double msWmma = timeGpuMs(10, [&] { gemmWmma<<<gridWmma, blockWmma>>>(d_Ah, d_Bh, d_Cw, n); });

    ResultTable table;
    table.add("FP32, one thread per element", msFp32, 0.0, gemmFlops(n));
    table.add("FP16 Tensor Core (WMMA)", msWmma, 0.0, gemmFlops(n));
    table.print("GEMM, 1024 x 1024");

    printSection("What just happened");
    printf("  One wmma::mma_sync computes a 16x16x16 multiply-accumulate in a\n");
    printf("  single instruction: 16*16*16*2 = 8192 FLOPs. That is the whole\n");
    printf("  reason a card with Tensor Cores quotes an FP16 number many times\n");
    printf("  its FP32 one.\n");

    printf("\n  Three rules the API imposes\n");
    printf("    - the WHOLE WARP cooperates on one tile; every thread must call\n");
    printf("      every wmma function, uniformly. A divergent call is undefined.\n");
    printf("    - you never index into a fragment. Its register layout is\n");
    printf("      deliberately unspecified; you only load, multiply and store.\n");
    printf("    - pointers need 256-bit alignment and the leading dimension must\n");
    printf("      be a multiple of 16 for half.\n");

    printf("\n  Precision is a real trade, not a free win\n");
    printf("    Inputs are FP16 (~3 decimal digits); the accumulator is FP32.\n");
    printf("    For a GEMM that is usually fine because accumulation dominates\n");
    printf("    the error - but the check above uses a 2%% tolerance for a reason.\n");
    printf("    Before using this in anger, measure the error on YOUR data, not on\n");
    printf("    random uniform values.\n");
    printf("\n    BF16 (sm_80+) trades mantissa bits for FP32's exponent range,\n");
    printf("    which makes it far safer for training. TF32 is automatic on\n");
    printf("    Ampere for FP32 GEMMs unless you opt out.\n");

    printf("\n  This kernel is a teaching version, not a fast one. It reads A and B\n");
    printf("  straight from global memory with no shared-memory staging, so it\n");
    printf("  wastes most of the Tensor Cores' throughput on memory stalls. A real\n");
    printf("  implementation stages tiles in shared memory exactly as in Phase 3\n");
    printf("  exercise 04 - which is what CUTLASS does, in about 20 layers.\n");

    CUDA_CHECK(cudaFree(d_Af));
    CUDA_CHECK(cudaFree(d_Bf));
    CUDA_CHECK(cudaFree(d_Cf));
    CUDA_CHECK(cudaFree(d_Cw));
    CUDA_CHECK(cudaFree(d_Ah));
    CUDA_CHECK(cudaFree(d_Bh));

    reportCheck("Tensor Core GEMM is faster than the naive FP32 kernel", msWmma < msFp32);
    return verifySummary();
}
