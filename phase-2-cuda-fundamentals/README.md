# Phase 2: CUDA Fundamentals – Lập trình CUDA cơ bản

## 🎯 Mục tiêu
Viết được CUDA kernel đầu tiên, hiểu execution model và memory model cơ bản.

## 📁 Cấu trúc bài tập

```
exercises/
├── 01-hello-cuda/         # Hello World CUDA kernel
├── 02-vector-add/         # Vector addition
├── 03-saxpy/              # SAXPY operation
├── 04-matrix-multiply/    # Naive matrix multiplication
├── 05-image-blur/         # Box blur on GPU
└── 06-cuda-wrapper/       # CUDA memory management wrapper class
```

## 📋 Checklist
- [ ] Cài đặt CUDA Toolkit thành công
- [ ] Hiểu `__global__`, `__device__`, `__host__`
- [ ] Hiểu thread hierarchy: Thread → Warp → Block → Grid
- [ ] Tính toán được global thread ID
- [ ] Sử dụng `cudaMalloc`, `cudaFree`, `cudaMemcpy`
- [ ] Error handling đúng cách
- [ ] Implement vector operations
- [ ] Implement matrix operations
- [ ] Hoàn thành tất cả bài tập
