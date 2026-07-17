# 🐳 Docker Development Environment

## Quick Start

```bash
# Build & start container
docker compose up -d --build

# Enter container shell
docker compose exec cuda-dev bash

# Verify CUDA is working
verify-cuda

# Stop container
docker compose down
```

## Cách sử dụng

### Build một bài tập

```bash
# Cách 1: Dùng nvcc trực tiếp
cd phase-2-cuda-fundamentals/exercises/01-hello-cuda
nvcc -arch=sm_75 -o hello_cuda main.cu -I../../../common
./hello_cuda

# Cách 2: Dùng CMake (uncomment exercise trong CMakeLists.txt)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
./bin/hello_cuda
```

### VS Code DevContainer

1. Cài extension **Dev Containers** trong VS Code
2. Mở project folder
3. `Ctrl+Shift+P` → **"Reopen in Container"**
4. VS Code sẽ tự build Docker image và kết nối

### Jupyter Notebook

```bash
# Trong container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Truy cập: http://localhost:8888

## Cấu hình GPU

- **Target GPU**: GTX 1660 SUPER (Compute Capability 7.5)
- **CUDA Architecture**: `sm_75`
- **Driver compatibility**: CUDA 12.6 (forward compatible with driver 580.x)

## Profiling trong Docker

```bash
# Nsight Systems
nsys profile --stats=true ./my_program

# Nsight Compute (cần --privileged hoặc không có MIG)
ncu --set full ./my_program
```

> ⚠️ Nếu `ncu` báo lỗi permission, thêm `--privileged` vào docker-compose.yml
