#!/usr/bin/env bash
# ============================================================
# verify-cuda - check that the toolchain and GPU are usable
# ============================================================
# Run this first, before touching any exercise. It compiles and
# runs a tiny kernel, which catches driver/toolkit mismatches that
# `nvcc --version` alone would not.
# ============================================================

echo "------------------------------------------------------------"
echo "  CUDA environment verification"
echo "------------------------------------------------------------"
echo

STATUS=0

echo "Compiler"
if command -v nvcc >/dev/null 2>&1; then
    echo "  nvcc    $(nvcc --version | grep -oP 'release \K[0-9.]+')"
else
    echo "  nvcc    MISSING"
    STATUS=1
fi

echo
echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
                   --format=csv,noheader | sed 's/^/  /'
    else
        echo "  nvidia-smi failed - the GPU is not passed through to this container"
        STATUS=1
    fi
else
    echo "  nvidia-smi not found - the GPU is not visible here"
    STATUS=1
fi

echo
echo "Libraries"
for lib in cublas cufft cusparse curand cusolver; do
    if ls "${CUDA_HOME:-/usr/local/cuda}"/lib64/lib${lib}.so* >/dev/null 2>&1; then
        echo "  ok      ${lib}"
    else
        echo "  MISSING ${lib}"
    fi
done

echo
echo "Developer tools"
for tool in cmake ninja gdb cuda-gdb compute-sanitizer ncu nsys cuobjdump nvdisasm; do
    if command -v "${tool}" >/dev/null 2>&1; then
        echo "  ok      ${tool}"
    else
        echo "  absent  ${tool}"
    fi
done

echo
echo "Compile and run test"
TMPDIR_="$(mktemp -d)"
cat > "${TMPDIR_}/test.cu" << 'EOF'
#include <cstdio>

__global__ void addOne(int* v) { v[threadIdx.x] += 1; }

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("  no CUDA device available\n");
        return 1;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        printf("  cudaGetDeviceProperties failed\n");
        return 1;
    }
    printf("  device: %s (sm_%d%d, %d SMs, %.1f GB)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    int host[32];
    for (int i = 0; i < 32; ++i) host[i] = i;

    int* dev = nullptr;
    if (cudaMalloc(&dev, sizeof(host)) != cudaSuccess) {
        printf("  cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(dev, host, sizeof(host), cudaMemcpyHostToDevice);
    addOne<<<1, 32>>>(dev);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev);
        return 1;
    }
    cudaMemcpy(host, dev, sizeof(host), cudaMemcpyDeviceToHost);
    cudaFree(dev);

    for (int i = 0; i < 32; ++i) {
        if (host[i] != i + 1) {
            printf("  wrong result at %d: %d\n", i, host[i]);
            return 1;
        }
    }
    printf("  kernel executed and produced the expected result\n");
    return 0;
}
EOF

# -arch=native compiles for whatever GPU is installed (CUDA 11.5+).
if nvcc -arch=native -o "${TMPDIR_}/test" "${TMPDIR_}/test.cu" 2>"${TMPDIR_}/err"; then
    if "${TMPDIR_}/test"; then
        echo "  compilation and execution: OK"
    else
        echo "  execution FAILED"
        STATUS=1
    fi
else
    echo "  compilation FAILED"
    sed 's/^/    /' "${TMPDIR_}/err"
    STATUS=1
fi
rm -rf "${TMPDIR_}"

echo
echo "------------------------------------------------------------"
if [ ${STATUS} -eq 0 ]; then
    echo "  Environment is ready. Start with phase-1-foundation/README.md"
else
    echo "  Problems were found - see docs/SETUP.md for troubleshooting"
fi
echo "------------------------------------------------------------"
exit ${STATUS}
