// ============================================================
// 02 - Thread indexing in 1D, 2D and 3D  [STARTER]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

// ------------------------------------------------------------
// TODO 1: 1D - one thread per element
// ------------------------------------------------------------
//     gid = blockIdx.x * blockDim.x + threadIdx.x
// Write out[gid] = gid, bounds-checked.
__global__ void index1D(int* out, int n) {
    (void)out;
    (void)n;
    // TODO
}

// ------------------------------------------------------------
// TODO 2: 2D - one thread per pixel
// ------------------------------------------------------------
// dim3 has .x, .y and .z, so a 2D launch gives you two independent
// index pairs:
//     x = blockIdx.x * blockDim.x + threadIdx.x
//     y = blockIdx.y * blockDim.y + threadIdx.y
// Flatten with y * width + x, and check BOTH bounds.
//
// Which dimension should be x? The one that is contiguous in
// memory. Threads in a warp differ in threadIdx.x first, so making
// x the column index means a warp reads 32 adjacent addresses -
// one memory transaction instead of 32.
__global__ void index2D(int* out, int width, int height) {
    (void)out;
    (void)width;
    (void)height;
    // TODO: out[y * width + x] = x + y;
}

// ------------------------------------------------------------
// TODO 3: 3D - one thread per voxel
// ------------------------------------------------------------
// Same idea with a third axis. Note gridDim.z is limited to 65535
// on every architecture, and blockDim.z to 64.
__global__ void index3D(int* out, int dx, int dy, int dz) {
    (void)out;
    (void)dx;
    (void)dy;
    (void)dz;
    // TODO: out[(z * dy + y) * dx + x] = x + y + z;
}

// ------------------------------------------------------------
// TODO 4: the grid-stride loop
// ------------------------------------------------------------
// So far the grid has had to be at least as large as the data. A
// grid-stride loop decouples the two: launch a grid sized for the
// GPU, and let each thread walk the array in steps of the total
// thread count.
//
//     int gid    = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = gridDim.x * blockDim.x;
//     for (int i = gid; i < n; i += stride) out[i] = i;
//
// Why it is worth the extra line:
//   - one launch configuration works for any n
//   - the grid can be tuned to the GPU (a few blocks per SM) rather
//     than to the data
//   - consecutive threads still touch consecutive addresses in
//     every iteration, so it stays coalesced
//   - the bounds check comes free from the loop condition
__global__ void gridStride(int* out, int n) {
    (void)out;
    (void)n;
    // TODO
}

int main() {
    printBanner("Phase 2 / 02 - Thread indexing (starter)");
    requireCudaDevice();

    printSection("1D");
    printf("  TODO: allocate, launch index1D<<<ceilDiv(n,256), 256>>>, verify\n");

    printSection("2D");
    printf("  TODO: launch with dim3 block(16,16) and dim3 grid(...)\n");

    printSection("3D");
    printf("  TODO: launch with dim3 block(8,8,4)\n");

    printSection("Grid-stride loop");
    printf("  TODO: launch a FIXED grid and let each thread handle many elements\n");

    printTodoNotice("implement the four kernels and their launches in main.cu");
    return verifySummary();
}
