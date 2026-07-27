// ============================================================
// 09 - RGB to grayscale: AoS versus SoA  [SOLUTION]
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "reference.h"

__device__ __forceinline__ unsigned char luma(unsigned char r, unsigned char g,
                                              unsigned char b) {
    float y = kWeightR * r + kWeightG * g + kWeightB * b;
    int v = static_cast<int>(y + 0.5f);
    return static_cast<unsigned char>(v > 255 ? 255 : (v < 0 ? 0 : v));
}

// ------------------------------------------------------------
// Array of Structures: RGBRGBRGB...
// ------------------------------------------------------------
// Thread i reads bytes 3i, 3i+1, 3i+2. Across a warp that is 96
// consecutive bytes - already contiguous, already coalesced. The
// only cost is that it takes three separate byte-load instructions.
__global__ void grayInterleaved(const unsigned char* rgb, unsigned char* gray,
                                size_t pixels) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < pixels; i += stride) {
        gray[i] = luma(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }
}

// ------------------------------------------------------------
// Structure of Arrays: RRRR... GGGG... BBBB...
// ------------------------------------------------------------
// Three separate planes, each read contiguously: three runs of 32
// bytes per warp. Compare that with AoS above - 96 consecutive
// bytes per warp. The same sectors, the same traffic, the same
// three load instructions. Expect no meaningful difference, and
// see the redChannel kernels below for the case where the layout
// really does decide the answer.
__global__ void grayPlanar(const unsigned char* r, const unsigned char* g,
                           const unsigned char* b, unsigned char* gray, size_t pixels) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < pixels; i += stride) {
        gray[i] = luma(r[i], g[i], b[i]);
    }
}

// ------------------------------------------------------------
// Where AoS actually hurts: reading ONE field of each structure
// ------------------------------------------------------------
// Extract just the red channel. From the interleaved buffer that
// is a stride-3 read: a warp touches 96 bytes and uses 32 of them,
// so two thirds of every sector fetched is discarded.
__global__ void redFromInterleaved(const unsigned char* rgb, unsigned char* out,
                                   size_t pixels) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < pixels; i += stride) out[i] = rgb[i * 3];
}

// From the planar buffer the same extraction is a plain contiguous
// copy: 32 bytes per warp, all of them used.
__global__ void redFromPlanar(const unsigned char* r, unsigned char* out, size_t pixels) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < pixels; i += stride) out[i] = r[i];
}

// ------------------------------------------------------------
// AoS with a vectorised load
// ------------------------------------------------------------
// If the layout is fixed - and with an image file it usually is -
// the trick is to have each thread handle FOUR pixels, so it can
// read 12 bytes as three uchar4 loads instead of twelve byte loads.
__global__ void grayInterleavedVec4(const uchar4* rgb12, uchar4* gray4, size_t quads) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = gid; i < quads; i += stride) {
        // 4 pixels = 12 bytes = exactly three uchar4.
        uchar4 a = rgb12[i * 3 + 0];  // R0 G0 B0 R1
        uchar4 b = rgb12[i * 3 + 1];  // G1 B1 R2 G2
        uchar4 c = rgb12[i * 3 + 2];  // B2 R3 G3 B3

        uchar4 out;
        out.x = luma(a.x, a.y, a.z);
        out.y = luma(a.w, b.x, b.y);
        out.z = luma(b.z, b.w, c.x);
        out.w = luma(c.y, c.z, c.w);
        gray4[i] = out;
    }
}

int main() {
    printBanner("Phase 2 / 09 - RGB to grayscale");
    requireCudaDevice();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Uses the file if it exists, otherwise generates one and saves
    // it so you can look at the input too.
    Image img = loadOrCreateTestImage("input.ppm", kImageWidth, kImageHeight);
    const size_t pixels = img.pixelCount();
    printf("  %d x %d = %zu pixels (%.1f MB RGB)\n", img.width, img.height, pixels,
           img.byteCount() / (1024.0 * 1024.0));

    // Golden result on the host.
    std::vector<unsigned char> golden(pixels);
    rgbToGrayCPU(img.data.data(), golden.data(), pixels);

    // Planar copies of the same image.
    std::vector<unsigned char> hr(pixels), hg(pixels), hb(pixels);
    deinterleave(img.data.data(), hr.data(), hg.data(), hb.data(), pixels);

    unsigned char *d_rgb = nullptr, *d_gray = nullptr;
    unsigned char *d_r = nullptr, *d_g = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, img.byteCount()));
    CUDA_CHECK(cudaMalloc(&d_gray, pixels));
    CUDA_CHECK(cudaMalloc(&d_r, pixels));
    CUDA_CHECK(cudaMalloc(&d_g, pixels));
    CUDA_CHECK(cudaMalloc(&d_b, pixels));

    CUDA_CHECK(cudaMemcpy(d_rgb, img.data.data(), img.byteCount(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, hr.data(), pixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, hg.data(), pixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, hb.data(), pixels, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = prop.multiProcessorCount * 8;
    std::vector<unsigned char> result(pixels);

    auto check = [&](const char* label) {
        CUDA_CHECK(cudaMemcpy(result.data(), d_gray, pixels, cudaMemcpyDeviceToHost));
        // Not an exact comparison - see the note in reference.h on
        // FMA contraction changing the rounding by one grey level.
        checkImage(label, result.data(), golden.data(), pixels);
    };

    // ========================================================
    // Correctness
    // ========================================================
    printSection("Correctness");

    CUDA_CHECK(cudaMemset(d_gray, 0, pixels));
    grayInterleaved<<<grid, block>>>(d_rgb, d_gray, pixels);
    CUDA_CHECK_LAST();
    check("interleaved (AoS)");

    CUDA_CHECK(cudaMemset(d_gray, 0, pixels));
    grayPlanar<<<grid, block>>>(d_r, d_g, d_b, d_gray, pixels);
    CUDA_CHECK_LAST();
    check("planar (SoA)");

    const size_t quads = pixels / 4;
    CUDA_CHECK(cudaMemset(d_gray, 0, pixels));
    grayInterleavedVec4<<<grid, block>>>(reinterpret_cast<const uchar4*>(d_rgb),
                                         reinterpret_cast<uchar4*>(d_gray), quads);
    CUDA_CHECK_LAST();
    if (quads * 4 < pixels) {
        // Tail: the last few pixels that do not fill a group of 4.
        grayInterleaved<<<1, 256>>>(d_rgb + quads * 4 * 3, d_gray + quads * 4,
                                    pixels - quads * 4);
        CUDA_CHECK_LAST();
    }
    check("interleaved with uchar4 loads");

    // ========================================================
    // Performance
    // ========================================================
    printSection("Performance");

    double msCpu =
        timeCpuMs(5, [&] { rgbToGrayCPU(img.data.data(), result.data(), pixels); });
    double msAoS =
        timeGpuMs(20, [&] { grayInterleaved<<<grid, block>>>(d_rgb, d_gray, pixels); });
    double msSoA = timeGpuMs(
        20, [&] { grayPlanar<<<grid, block>>>(d_r, d_g, d_b, d_gray, pixels); });
    double msVec = timeGpuMs(20, [&] {
        grayInterleavedVec4<<<grid, block>>>(reinterpret_cast<const uchar4*>(d_rgb),
                                             reinterpret_cast<uchar4*>(d_gray), quads);
    });

    const double traffic = grayscaleBytes(pixels);
    ResultTable table;
    table.add("CPU (single thread)", msCpu, traffic);
    table.add("GPU interleaved (AoS)", msAoS, traffic);
    table.add("GPU interleaved, uchar4", msVec, traffic);
    table.add("GPU planar (SoA)", msSoA, traffic);
    table.print("RGB to grayscale");

    printSection("AoS versus SoA");
    const double peak = theoreticalBandwidthGBs();
    printf("  %-30s %7.1f GB/s", "interleaved (AoS)", gbPerSec(traffic, msAoS));
    if (peak > 0.0) printf("  (%2.0f%% of peak)", 100.0 * gbPerSec(traffic, msAoS) / peak);
    printf("\n  %-30s %7.1f GB/s", "interleaved, uchar4", gbPerSec(traffic, msVec));
    if (peak > 0.0) printf("  (%2.0f%% of peak)", 100.0 * gbPerSec(traffic, msVec) / peak);
    printf("\n  %-30s %7.1f GB/s", "planar (SoA)", gbPerSec(traffic, msSoA));
    if (peak > 0.0) printf("  (%2.0f%% of peak)", 100.0 * gbPerSec(traffic, msSoA) / peak);
    printf("\n");

    printf("\n  The result you probably did not expect\n");
    printf("    AoS and SoA come out within noise of each other. The usual advice\n");
    printf("    'prefer SoA on a GPU' does not apply to THIS kernel, because it\n");
    printf("    needs all three channels of every pixel:\n");
    printf("      AoS: a warp reads 96 consecutive bytes  -> 3 sectors\n");
    printf("      SoA: a warp reads three runs of 32 bytes -> 3 sectors\n");
    printf("    Same traffic, same instruction count, same speed.\n");
    printf("\n    The real win here is uchar4: four pixels per thread turns twelve\n");
    printf("    byte-loads into three 4-byte loads. That is an INSTRUCTION-count\n");
    printf("    win, not a layout win, and it is worth %.2fx.\n", msAoS / msVec);

    // ========================================================
    // Now the case where the layout does decide the answer
    // ========================================================
    printSection("When AoS really does hurt: reading one channel");

    unsigned char* d_red = nullptr;
    CUDA_CHECK(cudaMalloc(&d_red, pixels));

    // Correctness: both must extract the same red channel.
    std::vector<unsigned char> redGolden(pixels);
    for (size_t i = 0; i < pixels; ++i) redGolden[i] = img.data[i * 3];

    redFromInterleaved<<<grid, block>>>(d_rgb, d_red, pixels);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_red, pixels, cudaMemcpyDeviceToHost));
    checkArrayExact("red channel from AoS", result.data(), redGolden.data(), pixels);

    CUDA_CHECK(cudaMemset(d_red, 0, pixels));
    redFromPlanar<<<grid, block>>>(d_r, d_red, pixels);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(result.data(), d_red, pixels, cudaMemcpyDeviceToHost));
    checkArrayExact("red channel from SoA", result.data(), redGolden.data(), pixels);

    double msRedAoS =
        timeGpuMs(20, [&] { redFromInterleaved<<<grid, block>>>(d_rgb, d_red, pixels); });
    double msRedSoA =
        timeGpuMs(20, [&] { redFromPlanar<<<grid, block>>>(d_r, d_red, pixels); });

    const double redTraffic = singleChannelBytes(pixels);
    ResultTable redTable;
    redTable.add("red channel from AoS (stride 3)", msRedAoS, redTraffic);
    redTable.add("red channel from SoA (planar)", msRedSoA, redTraffic);
    redTable.print("Extracting one channel");

    printf("\n  Here the layout is decisive: %.2fx.\n", msRedAoS / msRedSoA);
    printf("    AoS: a warp touches 96 bytes and uses 32. Two thirds of every\n");
    printf("         sector fetched is thrown away.\n");
    printf("    SoA: 32 bytes fetched, 32 bytes used.\n");

    printf("\n  So the rule is not 'always use SoA'. It is:\n");
    printf("    AoS versus SoA matters in proportion to how much of each structure\n");
    printf("    you actually read. Need every field? The layouts tie. Need one\n");
    printf("    field out of many? SoA wins by roughly the ratio between them.\n");
    printf("\n    That is why a particle system with a 32-byte struct sees a huge\n");
    printf("    win from SoA when a kernel only touches `position`, and why this\n");
    printf("    grayscale kernel sees none.\n");

    CUDA_CHECK(cudaFree(d_red));

    // Save the result so it can actually be looked at.
    Image out;
    out.resize(img.width, img.height, 1);
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_gray, pixels, cudaMemcpyDeviceToHost));
    if (savePPM("output_gray.pgm", out)) {
        printf("\n  Wrote output_gray.pgm - open it, or convert with:\n");
        printf("      convert output_gray.pgm output_gray.png\n");
    }

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));

    reportCheck("vectorised loads beat scalar loads", msVec < msAoS);
    reportCheck("SoA wins clearly when only one channel is read", msRedSoA < msRedAoS);
    return verifySummary();
}
