// ============================================================
// 02 - Thread indexing in 1D, 2D and 3D  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__global__ void index1D(int* out, int n) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = gid;
}

__global__ void index2D(int* out, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // BOTH bounds have to be checked. A 1920x1080 image with 16x16
    // blocks needs a 120x68 grid, which covers 1920x1088 - the last
    // row of blocks has 8 rows of threads with nothing to do.
    if (x < width && y < height) {
        // x is the fastest-varying index, so threads within a warp
        // (which differ in threadIdx.x) write to consecutive
        // addresses. Swap x and y here and the kernel still gives
        // the right answer, roughly 10x slower.
        out[y * width + x] = x + y;
    }
}

__global__ void index3D(int* out, int dx, int dy, int dz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dx && y < dy && z < dz) {
        out[(z * dy + y) * dx + x] = x + y + z;
    }
}

__global__ void gridStride(int* out, int n) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;  // total threads in the grid
    for (int i = gid; i < n; i += stride) {
        out[i] = i;
    }
}

int main() {
    printBanner("Phase 2 / 02 - Thread indexing");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ========================================================
    // 1D
    // ========================================================
    printSection("1D - one thread per element");
    {
        const int n = kLength;
        const int block = 256;
        const int grid = ceilDiv(n, block);
        printf("  n = %d  ->  <<<%d, %d>>> = %d threads\n", n, grid, block, grid * block);

        int* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
        index1D<<<grid, block>>>(d, n);
        CUDA_CHECK_LAST();

        std::vector<int> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        auto expected = golden1D(n);
        checkArrayExact("1D indexing", host.data(), expected.data(), n);
    }

    // ========================================================
    // 2D
    // ========================================================
    printSection("2D - one thread per pixel");
    {
        const int w = kWidth, h = kHeight;
        // 16x16 = 256 threads: a multiple of 32, and square enough
        // that 2D neighbourhoods (used by every stencil kernel) stay
        // inside one block.
        const dim3 block(16, 16);
        const dim3 grid(ceilDiv(w, static_cast<int>(block.x)),
                        ceilDiv(h, static_cast<int>(block.y)));
        printf("  %dx%d image -> grid(%u, %u) x block(%u, %u) = %u threads\n", w, h, grid.x,
               grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y);
        printf("  covers %ux%u pixels, %u of them idle\n", grid.x * block.x, grid.y * block.y,
               grid.x * block.x * grid.y * block.y - w * h);

        int* d = nullptr;
        const size_t bytes = static_cast<size_t>(w) * h * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d, bytes));
        index2D<<<grid, block>>>(d, w, h);
        CUDA_CHECK_LAST();

        std::vector<int> host(static_cast<size_t>(w) * h);
        CUDA_CHECK(cudaMemcpy(host.data(), d, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        auto expected = golden2D(w, h);
        checkArrayExact("2D indexing", host.data(), expected.data(), host.size());
    }

    // ========================================================
    // 3D
    // ========================================================
    printSection("3D - one thread per voxel");
    {
        const int dx = kDimX, dy = kDimY, dz = kDimZ;
        // blockDim.z is capped at 64, and the product of all three
        // must stay <= maxThreadsPerBlock (1024).
        const dim3 block(8, 8, 4);  // 256 threads
        const dim3 grid(ceilDiv(dx, static_cast<int>(block.x)),
                        ceilDiv(dy, static_cast<int>(block.y)),
                        ceilDiv(dz, static_cast<int>(block.z)));
        printf("  %dx%dx%d volume -> grid(%u, %u, %u) x block(%u, %u, %u)\n", dx, dy, dz,
               grid.x, grid.y, grid.z, block.x, block.y, block.z);

        int* d = nullptr;
        const size_t bytes = static_cast<size_t>(dx) * dy * dz * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d, bytes));
        index3D<<<grid, block>>>(d, dx, dy, dz);
        CUDA_CHECK_LAST();

        std::vector<int> host(static_cast<size_t>(dx) * dy * dz);
        CUDA_CHECK(cudaMemcpy(host.data(), d, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        auto expected = golden3D(dx, dy, dz);
        checkArrayExact("3D indexing", host.data(), expected.data(), host.size());
    }

    // ========================================================
    // Grid-stride loop
    // ========================================================
    printSection("Grid-stride loop");
    {
        const int n = kLength;
        const int block = 256;
        // Size the grid to the GPU, not to the data. A few blocks
        // per SM is enough to keep every SM busy; more just adds
        // scheduling overhead.
        const int grid = prop.multiProcessorCount * 4;
        printf("  n = %d, but launching only <<<%d, %d>>> = %d threads\n", n, grid, block,
               grid * block);
        printf("  each thread handles about %.1f elements\n",
               static_cast<double>(n) / (grid * block));

        int* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
        gridStride<<<grid, block>>>(d, n);
        CUDA_CHECK_LAST();

        std::vector<int> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        auto expected = golden1D(n);
        checkArrayExact("grid-stride loop", host.data(), expected.data(), n);

        // The same kernel, unchanged, handles a completely different
        // size. A one-thread-per-element kernel would need a new
        // grid; this one does not.
        const int n2 = 7;
        int* d2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d2, n2 * sizeof(int)));
        gridStride<<<grid, block>>>(d2, n2);
        CUDA_CHECK_LAST();
        std::vector<int> host2(n2);
        CUDA_CHECK(cudaMemcpy(host2.data(), d2, n2 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d2));
        auto expected2 = golden1D(n2);
        checkArrayExact("same launch config, n = 7", host2.data(), expected2.data(), n2);
    }

    printSection("Launch limits on this device");
    printf("  %-34s %d\n", "max threads per block", prop.maxThreadsPerBlock);
    printf("  %-34s (%d, %d, %d)\n", "max block dimensions", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  %-34s (%d, %d, %d)\n", "max grid dimensions", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n  Note blockDim.z is limited to %d and gridDim.y/z to %d - if your\n",
           prop.maxThreadsDim[2], prop.maxGridSize[1]);
    printf("  problem is bigger than that in one axis, a grid-stride loop is not\n");
    printf("  an optimisation, it is the only option.\n");

    return verifySummary();
}
