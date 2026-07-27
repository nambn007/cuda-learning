// ============================================================
// GPU ray tracer - WORKED CORE MODULE
// ============================================================
// This is not the project. It is the foundation the project is
// built on: a working path tracer over a handful of spheres, with
// per-thread RNG, diffuse bounces and a PPM writer.
//
// It renders a real image in a few milliseconds and verifies that
// the result is deterministic and non-degenerate. Everything from
// here - BVH traversal, triangle meshes, materials, denoising - is
// yours to build. See README.md for the milestones.
//
// The two GPU-specific lessons already visible in this file:
//
//   RNG PER THREAD. Every thread needs its own independent random
//   sequence, seeded from its pixel index. A shared generator would
//   be a contention disaster; a shared seed would make every pixel
//   sample the same directions.
//
//   DIVERGENCE. Neighbouring pixels bounce in different directions
//   and terminate after different numbers of bounces, so threads in
//   a warp are doing genuinely different work. That is why ray
//   tracing is the hardest divergence problem in this curriculum,
//   and why the milestones put BVH traversal last.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"
#include "ppm.h"

// ------------------------------------------------------------
// Minimal vector maths
// ------------------------------------------------------------
// The default constructor must stay TRIVIAL: a __constant__ array
// of Spheres cannot be dynamically initialised, and Sphere is only
// trivially constructible if Vec3 is.
struct Vec3 {
    float x, y, z;
    Vec3() = default;
    __host__ __device__ constexpr Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline Vec3 operator*(Vec3 a, float s) {
    return Vec3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline Vec3 operator*(Vec3 a, Vec3 b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3 normalise(Vec3 v) {
    const float inv = rsqrtf(dot(v, v));
    return v * inv;
}

struct Sphere {
    Vec3 centre;
    float radius;
    Vec3 albedo;
    Vec3 emission;
};

// ------------------------------------------------------------
// Per-thread RNG
// ------------------------------------------------------------
// A tiny xorshift. Each thread seeds from its own pixel index, so
// every pixel gets an independent sequence and the whole image is
// reproducible. cuRAND is the production answer; this keeps the
// dependency list at zero.
__device__ inline unsigned int randNext(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline float randFloat(unsigned int& state) {
    return (randNext(state) & 0x00ffffffu) / 16777216.0f;
}

// Cosine-weighted hemisphere sample around a normal.
__device__ inline Vec3 sampleHemisphere(Vec3 n, unsigned int& state) {
    const float u1 = randFloat(state);
    const float u2 = randFloat(state);
    const float r = sqrtf(u1);
    const float theta = 6.2831853f * u2;

    // Build a basis around n.
    Vec3 up = fabsf(n.z) < 0.999f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 tangent = normalise(Vec3(up.y * n.z - up.z * n.y, up.z * n.x - up.x * n.z,
                                  up.x * n.y - up.y * n.x));
    Vec3 bitangent = Vec3(n.y * tangent.z - n.z * tangent.y,
                          n.z * tangent.x - n.x * tangent.z,
                          n.x * tangent.y - n.y * tangent.x);

    return normalise(tangent * (r * cosf(theta)) + bitangent * (r * sinf(theta)) +
                     n * sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

// ------------------------------------------------------------
// Ray-sphere intersection
// ------------------------------------------------------------
// Solve |o + t*d - c|^2 = r^2 for the smaller positive t.
__device__ inline float hitSphere(const Sphere& s, Vec3 origin, Vec3 dir, float tMin,
                                  float tMax) {
    const Vec3 oc = origin - s.centre;
    const float b = dot(oc, dir);
    const float c = dot(oc, oc) - s.radius * s.radius;
    const float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    const float sq = sqrtf(disc);
    float t = -b - sq;
    if (t < tMin || t > tMax) {
        t = -b + sq;
        if (t < tMin || t > tMax) return -1.0f;
    }
    return t;
}

__constant__ Sphere d_scene[8];
__constant__ int d_sphereCount;

__device__ Vec3 trace(Vec3 origin, Vec3 dir, unsigned int& state, int maxBounces) {
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    Vec3 radiance(0.0f, 0.0f, 0.0f);

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        float closest = 1e30f;
        int hit = -1;
        for (int i = 0; i < d_sphereCount; ++i) {
            const float t = hitSphere(d_scene[i], origin, dir, 1e-3f, closest);
            if (t > 0.0f) {
                closest = t;
                hit = i;
            }
        }

        if (hit < 0) {
            // Sky gradient.
            const float a = 0.5f * (dir.y + 1.0f);
            const Vec3 sky = Vec3(1.0f, 1.0f, 1.0f) * (1.0f - a) + Vec3(0.5f, 0.7f, 1.0f) * a;
            radiance = radiance + throughput * sky;
            break;
        }

        const Sphere& s = d_scene[hit];
        const Vec3 point = origin + dir * closest;
        const Vec3 normal = normalise(point - s.centre);

        radiance = radiance + throughput * s.emission;
        throughput = throughput * s.albedo;

        origin = point;
        dir = sampleHemisphere(normal, state);
    }
    return radiance;
}

__global__ void render(unsigned char* out, int width, int height, int samples,
                       int maxBounces) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Seed from the pixel index: independent per thread, and the
    // whole image is reproducible run to run.
    unsigned int state = static_cast<unsigned int>(y * width + x) * 9781u + 1u;
    randNext(state);

    const Vec3 camera(0.0f, 0.5f, 3.0f);
    const float aspect = static_cast<float>(width) / height;

    Vec3 colour(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; ++s) {
        const float u = (x + randFloat(state)) / width;
        const float v = (y + randFloat(state)) / height;
        const Vec3 dir = normalise(Vec3((2.0f * u - 1.0f) * aspect, 1.0f - 2.0f * v, -1.5f));
        colour = colour + trace(camera, dir, state, maxBounces);
    }
    colour = colour * (1.0f / samples);

    // Gamma 2.0, then clamp.
    auto encode = [](float c) {
        c = sqrtf(fmaxf(0.0f, fminf(1.0f, c)));
        return static_cast<unsigned char>(c * 255.0f + 0.5f);
    };

    const size_t i = (static_cast<size_t>(y) * width + x) * 3;
    out[i + 0] = encode(colour.x);
    out[i + 1] = encode(colour.y);
    out[i + 2] = encode(colour.z);
}

int main() {
    printBanner("Phase 6 - GPU ray tracer (worked core module)");
    requireCudaDevice();

    const int width = 800, height = 450;
    const int samples = 64;
    const int maxBounces = 6;

    // A Cornell-ish box made of spheres: two walls, a floor, a
    // light and two objects.
    Sphere scene[] = {
        {Vec3(0.0f, -100.5f, -1.0f), 100.0f, Vec3(0.8f, 0.8f, 0.8f), Vec3(0, 0, 0)},
        {Vec3(-1.0f, 0.0f, -1.2f), 0.5f, Vec3(0.9f, 0.3f, 0.3f), Vec3(0, 0, 0)},
        {Vec3(0.2f, 0.0f, -1.0f), 0.5f, Vec3(0.3f, 0.5f, 0.9f), Vec3(0, 0, 0)},
        {Vec3(1.2f, 0.2f, -1.4f), 0.4f, Vec3(0.9f, 0.9f, 0.4f), Vec3(0, 0, 0)},
        {Vec3(0.0f, 2.2f, -1.0f), 1.0f, Vec3(0, 0, 0), Vec3(4.0f, 3.8f, 3.4f)},
    };
    const int sphereCount = sizeof(scene) / sizeof(scene[0]);

    CUDA_CHECK(cudaMemcpyToSymbol(d_scene, scene, sizeof(scene)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sphereCount, &sphereCount, sizeof(int)));

    printf("  %dx%d, %d samples per pixel, up to %d bounces, %d spheres\n", width, height,
           samples, maxBounces, sphereCount);

    const size_t pixels = static_cast<size_t>(width) * height;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, pixels * 3));

    const dim3 block(16, 16);
    const dim3 grid(ceilDiv(width, 16), ceilDiv(height, 16));

    render<<<grid, block>>>(d_out, width, height, samples, maxBounces);
    CUDA_CHECK_KERNEL();

    Image img;
    img.resize(width, height, 3);
    CUDA_CHECK(cudaMemcpy(img.data.data(), d_out, pixels * 3, cudaMemcpyDeviceToHost));

    printSection("Correctness");
    // A renderer has no golden image, so check the properties that
    // a broken one would violate: the image must not be uniform,
    // must not be black, and must be reproducible.
    long long sum = 0;
    unsigned char minV = 255, maxV = 0;
    for (size_t i = 0; i < img.data.size(); ++i) {
        sum += img.data[i];
        if (img.data[i] < minV) minV = img.data[i];
        if (img.data[i] > maxV) maxV = img.data[i];
    }
    const double mean = static_cast<double>(sum) / img.data.size();

    reportCheck("image is not uniform", maxV > minV + 40);
    reportCheck("image is not black", mean > 20.0);
    reportCheck("image is not saturated", mean < 235.0);
    printf("  mean %.1f, range %d..%d\n", mean, minV, maxV);

    // Determinism: the same seed must give the same image.
    std::vector<unsigned char> again(pixels * 3);
    render<<<grid, block>>>(d_out, width, height, samples, maxBounces);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(again.data(), d_out, pixels * 3, cudaMemcpyDeviceToHost));
    reportCheck("render is deterministic", again == img.data);

    printSection("Performance");
    double ms = timeGpuMs(3, [&] { render<<<grid, block>>>(d_out, width, height, samples, maxBounces); });
    const double rays = static_cast<double>(pixels) * samples;
    printKV("render time", ms, "ms");
    printf("  %.1f million primary rays/second (up to %dx that including bounces)\n",
           rays / (ms * 1e-3) / 1e6, maxBounces);

    savePPM("render.ppm", img);
    printf("\n  Wrote render.ppm - open it. A ray tracer's best test is your eyes.\n");
    printf("  convert render.ppm render.png\n");

    printSection("Where the GPU difficulty lies");
    printf("  Two things in this file are already the hard part of the project:\n");
    printf("\n  RNG PER THREAD. Each thread seeds from its pixel index, so every\n");
    printf("  pixel has an independent sequence and the image is reproducible. A\n");
    printf("  shared generator would be a contention disaster; a shared seed would\n");
    printf("  make every pixel sample the same directions.\n");
    printf("\n  DIVERGENCE. Neighbouring pixels bounce in different directions and\n");
    printf("  terminate after different numbers of bounces, so threads in a warp do\n");
    printf("  genuinely different work. Ray tracing is the hardest divergence\n");
    printf("  problem in this curriculum - which is why the milestones in README.md\n");
    printf("  put BVH traversal last, after you have measured what divergence costs\n");
    printf("  you here.\n");

    CUDA_CHECK(cudaFree(d_out));
    return verifySummary();
}
