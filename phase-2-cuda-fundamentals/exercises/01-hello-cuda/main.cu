// ============================================================
// Hello CUDA – Bài tập đầu tiên
// ============================================================
// Mục tiêu: Verify CUDA environment hoạt động
// Build:    nvcc -arch=sm_75 -o hello_cuda main.cu
// ============================================================

#include <cstdio>
#include "../../../common/cuda_helper.h"

__global__ void helloCUDA() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU! Thread %d (Block %d, Thread %d)\n",
           tid, blockIdx.x, threadIdx.x);
}

int main() {
    // Print device info
    printDeviceInfo();

    // Launch kernel: 2 blocks × 4 threads = 8 threads total
    printf("\nLaunching kernel: 2 blocks × 4 threads\n\n");
    helloCUDA<<<2, 4>>>();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n✅ Done! CUDA is working correctly.\n");
    return 0;
}
