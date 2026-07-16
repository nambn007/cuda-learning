# Phase 3: Intermediate CUDA – Tối ưu hóa & Memory

## 🎯 Mục tiêu
Master memory hierarchy, tối ưu hóa kernel, và sử dụng profiling tools.

## 📁 Cấu trúc bài tập

```
exercises/
├── 01-shared-memory-matmul/  # Tiled matrix multiplication
├── 02-parallel-reduction/    # All reduction variants
├── 03-prefix-sum/            # Blelloch scan
├── 04-convolution-2d/        # 2D convolution with shared + constant memory
├── 05-histogram/             # GPU histogram
└── 06-profiling-report/      # Nsight profiling reports
```

## 📋 Checklist
- [ ] Master shared memory (bank conflicts, tiling)
- [ ] Hiểu coalesced memory access
- [ ] Sử dụng constant memory và texture memory
- [ ] Warp-level programming (shuffle, vote)
- [ ] Occupancy optimization
- [ ] Implement reduction, scan, histogram
- [ ] Sử dụng thành thạo Nsight Systems & Nsight Compute
- [ ] Đọc hiểu performance metrics
