# Phase 4: Advanced CUDA – Kỹ thuật nâng cao

## 🎯 Mục tiêu
Master CUDA streams, multi-GPU, dynamic parallelism, và advanced optimization.

## 📁 Cấu trúc bài tập

```
exercises/
├── 01-streams-pipeline/       # Overlap transfers & compute
├── 02-cuda-graphs/            # CUDA Graph pipeline
├── 03-multi-gpu-matmul/       # Multi-GPU matrix multiplication
├── 04-dynamic-parallelism/    # Merge sort with dynamic parallelism
└── 05-lock-free-queue/        # GPU lock-free queue with atomics
```

## 📋 Checklist
- [ ] Master CUDA streams và events
- [ ] Sử dụng CUDA Graphs
- [ ] Pinned memory và async transfers
- [ ] Multi-GPU programming với peer access
- [ ] NCCL collective operations
- [ ] Dynamic parallelism
- [ ] Cooperative groups nâng cao
- [ ] Register optimization & `__launch_bounds__`
- [ ] Đọc PTX/SASS assembly
