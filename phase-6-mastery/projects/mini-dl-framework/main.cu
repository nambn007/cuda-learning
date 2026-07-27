// ============================================================
// Mini deep-learning framework - WORKED CORE MODULE
// ============================================================
// The core of any training framework is one thing done correctly:
// a layer's BACKWARD pass must be the exact derivative of its
// FORWARD pass. Everything else - autograd graphs, optimisers,
// fusion - is bookkeeping on top of that.
//
// This module implements a linear layer,
//
//     Y = X W + b        X: [B, I]   W: [I, O]   b: [O]
//
// with its three gradients,
//
//     dX = dY W^T        dW = X^T dY        db = sum_rows(dY)
//
// and checks them against NUMERICAL differentiation. That check is
// the single most valuable tool in this project: it catches a
// transposed index or a missing sum in seconds, where a training
// run would just converge slightly worse and never tell you why.
//
// See README.md for the milestones that build a framework on this.
// ============================================================

#include <cstdio>
#include <vector>

#include "cuda_helper.h"

// ------------------------------------------------------------
// Forward: Y = X W + b
// ------------------------------------------------------------
// One thread per output element. Naive on purpose - the point here
// is correctness, and Phase 3/04 already showed how to make a GEMM
// fast. Swapping this for the tiled version is milestone 2.
__global__ void linearForward(const float* X, const float* W, const float* b, float* Y,
                              int B, int I, int O) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= B || o >= O) return;

    float acc = b[o];
    for (int i = 0; i < I; ++i) acc += X[n * I + i] * W[i * O + o];
    Y[n * O + o] = acc;
}

// dX = dY W^T
__global__ void linearBackwardInput(const float* dY, const float* W, float* dX, int B,
                                    int I, int O) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= B || i >= I) return;

    float acc = 0.0f;
    for (int o = 0; o < O; ++o) acc += dY[n * O + o] * W[i * O + o];
    dX[n * I + i] = acc;
}

// dW = X^T dY
__global__ void linearBackwardWeight(const float* X, const float* dY, float* dW, int B,
                                     int I, int O) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= I || o >= O) return;

    float acc = 0.0f;
    for (int n = 0; n < B; ++n) acc += X[n * I + i] * dY[n * O + o];
    dW[i * O + o] = acc;
}

// db = sum over the batch dimension
__global__ void linearBackwardBias(const float* dY, float* db, int B, int O) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    float acc = 0.0f;
    for (int n = 0; n < B; ++n) acc += dY[n * O + o];
    db[o] = acc;
}

// ------------------------------------------------------------
// Host reference, used for the numerical gradient check
// ------------------------------------------------------------
static void forwardCPU(const std::vector<float>& X, const std::vector<float>& W,
                       const std::vector<float>& b, std::vector<float>& Y, int B, int I,
                       int O) {
    for (int n = 0; n < B; ++n)
        for (int o = 0; o < O; ++o) {
            float acc = b[o];
            for (int i = 0; i < I; ++i) acc += X[n * I + i] * W[i * O + o];
            Y[n * O + o] = acc;
        }
}

// A scalar loss, so that a numerical gradient is well defined:
// L = sum(Y * Y) / 2, hence dL/dY = Y.
static double lossCPU(const std::vector<float>& Y) {
    double s = 0.0;
    for (float v : Y) s += 0.5 * static_cast<double>(v) * v;
    return s;
}

int main() {
    printBanner("Phase 6 - Mini DL framework (worked core module)");
    requireCudaDevice();

    const int B = 64, I = 128, O = 96;
    printf("  Linear layer: X[%d,%d] @ W[%d,%d] + b[%d]\n", B, I, I, O, O);

    std::vector<float> X(B * I), W(I * O), b(O), Y(B * O);
    fillRandom(X.data(), X.size(), -1.0f, 1.0f, 181);
    fillRandom(W.data(), W.size(), -0.5f, 0.5f, 182);
    fillRandom(b.data(), b.size(), -0.1f, 0.1f, 183);

    float *d_X, *d_W, *d_b, *d_Y, *d_dY, *d_dX, *d_dW, *d_db;
    CUDA_CHECK(cudaMalloc(&d_X, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, W.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, Y.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dY, Y.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW, W.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db, b.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 blk(16, 16);

    // ========================================================
    // Forward
    // ========================================================
    printSection("Forward");
    linearForward<<<dim3(ceilDiv(O, 16), ceilDiv(B, 16)), blk>>>(d_X, d_W, d_b, d_Y, B, I, O);
    CUDA_CHECK_KERNEL();

    std::vector<float> gpuY(Y.size()), cpuY(Y.size());
    CUDA_CHECK(cudaMemcpy(gpuY.data(), d_Y, Y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    forwardCPU(X, W, b, cpuY, B, I, O);
    checkArray("forward matches the CPU reference", gpuY, cpuY, 1e-4, 1e-5);

    // ========================================================
    // Backward
    // ========================================================
    // With L = sum(Y*Y)/2 the upstream gradient is simply dY = Y.
    printSection("Backward");
    CUDA_CHECK(cudaMemcpy(d_dY, gpuY.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice));

    linearBackwardInput<<<dim3(ceilDiv(I, 16), ceilDiv(B, 16)), blk>>>(d_dY, d_W, d_dX, B, I, O);
    linearBackwardWeight<<<dim3(ceilDiv(O, 16), ceilDiv(I, 16)), blk>>>(d_X, d_dY, d_dW, B, I, O);
    linearBackwardBias<<<ceilDiv(O, 256), 256>>>(d_dY, d_db, B, O);
    CUDA_CHECK_KERNEL();

    std::vector<float> dW(W.size()), db(b.size()), dX(X.size());
    CUDA_CHECK(cudaMemcpy(dW.data(), d_dW, dW.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(db.data(), d_db, db.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dX.data(), d_dX, dX.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ========================================================
    // The gradient check
    // ========================================================
    // Central differences: dL/dp ~= (L(p + h) - L(p - h)) / 2h.
    // This is the single most valuable tool in the whole project.
    // A transposed index or a missing sum shows up here in seconds;
    // without it, training just converges slightly worse and never
    // tells you why.
    printSection("Numerical gradient check");
    const float h = 1e-2f;
    auto numericalGrad = [&](std::vector<float>& param, size_t idx) {
        const float original = param[idx];
        std::vector<float> tmp(Y.size());

        param[idx] = original + h;
        forwardCPU(X, W, b, tmp, B, I, O);
        const double lossPlus = lossCPU(tmp);

        param[idx] = original - h;
        forwardCPU(X, W, b, tmp, B, I, O);
        const double lossMinus = lossCPU(tmp);

        param[idx] = original;
        return static_cast<float>((lossPlus - lossMinus) / (2.0 * h));
    };

    auto compare = [&](const char* name, std::vector<float>& param,
                       const std::vector<float>& analytic, int samples) {
        double worst = 0.0;
        for (int s = 0; s < samples; ++s) {
            const size_t idx = (s * 7919) % param.size();
            const float num = numericalGrad(param, idx);
            const float ana = analytic[idx];
            const double rel = std::fabs(num - ana) / (std::fabs(num) + std::fabs(ana) + 1e-6);
            if (rel > worst) worst = rel;
        }
        printf("  %-28s worst relative error %.2e over %d samples\n", name, worst, samples);
        reportCheck(name, worst < 1e-2);
    };

    compare("dW", W, dW, 12);
    compare("db", b, db, 12);
    compare("dX", X, dX, 12);

    printSection("What comes next");
    printf("  This module is deliberately naive. The project is to build on it:\n");
    printf("    1. wrap the buffers in the Tensor type sketched in README.md\n");
    printf("    2. replace these GEMMs with the tiled kernel from Phase 3/04,\n");
    printf("       then with cuBLAS, and measure both\n");
    printf("    3. add ReLU and softmax + cross-entropy, each with a gradient check\n");
    printf("    4. add an autograd tape so backward() is generated, not written\n");
    printf("    5. add SGD and Adam, and train something small end to end\n");
    printf("    6. fuse the elementwise chains and measure the traffic saved\n");
    printf("\n  Rule for the whole project: EVERY new layer gets a numerical\n");
    printf("  gradient check before it is used in training. There is no cheaper\n");
    printf("  bug detector in machine learning.\n");

    for (float* p : {d_X, d_W, d_b, d_Y, d_dY, d_dX, d_dW, d_db}) CUDA_CHECK(cudaFree(p));
    return verifySummary();
}
