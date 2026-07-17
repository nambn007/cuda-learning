#!/bin/bash
# ============================================================
# Verify CUDA environment is working correctly
# Usage: verify-cuda
# ============================================================

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔍 CUDA Environment Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Check nvcc
echo "📦 NVCC Compiler:"
nvcc --version | grep release
echo ""

# 2. Check GPU visibility
echo "🖥️  GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ⚠️  nvidia-smi not available (GPU may not be passed through)"
echo ""

# 3. Check CUDA libraries
echo "📚 CUDA Libraries:"
ls -1 ${CUDA_HOME}/lib64/libcublas.so* 2>/dev/null && echo "  ✅ cuBLAS" || echo "  ❌ cuBLAS not found"
ls -1 ${CUDA_HOME}/lib64/libcufft.so* 2>/dev/null && echo "  ✅ cuFFT" || echo "  ❌ cuFFT not found"
ls -1 ${CUDA_HOME}/lib64/libcusparse.so* 2>/dev/null && echo "  ✅ cuSPARSE" || echo "  ❌ cuSPARSE not found"
ls -1 ${CUDA_HOME}/lib64/libcurand.so* 2>/dev/null && echo "  ✅ cuRAND" || echo "  ❌ cuRAND not found"
echo ""

# 4. Check build tools
echo "🛠️  Build Tools:"
echo "  gcc:   $(gcc --version | head -1)"
echo "  g++:   $(g++ --version | head -1)"
echo "  cmake: $(cmake --version | head -1)"
echo "  make:  $(make --version | head -1)"
echo ""

# 5. Compile & run a simple CUDA program
echo "🧪 Compile Test (Hello CUDA):"
TMPDIR=$(mktemp -d)
cat > "${TMPDIR}/test.cu" << 'EOF'
#include <cstdio>

__global__ void helloKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("  ✅ Hello from GPU! Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
    }
}

int main() {
    // Query device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("  ❌ cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("  GPU: %s (SM %d.%d, %d SMs, %zu MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024*1024));

    // Launch kernel
    helloKernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  ❌ Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("  ✅ CUDA compilation & execution: OK\n");
    return 0;
}
EOF

nvcc -arch=sm_75 -o "${TMPDIR}/test" "${TMPDIR}/test.cu" 2>&1 && \
    "${TMPDIR}/test" 2>&1 || echo "  ❌ Compilation or execution failed"

rm -rf "${TMPDIR}" 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Verification complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
