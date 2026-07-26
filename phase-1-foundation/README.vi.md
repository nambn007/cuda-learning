<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 1 · Nền tảng

> **3–4 tuần · ⭐ · 8 bài tập** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Vì sao có giai đoạn này

Bạn không thể tối ưu một kernel GPU nếu chưa giải thích được vì sao một vòng lặp
CPU lại chậm. Mọi ý tưởng ở đây đều xuất hiện lại trên GPU dưới một cái tên khác:

| Học ở đây | Xuất hiện lại dưới dạng |
|---|---|
| Bố cục row-major, truy cập stride 1 | Memory coalescing (GĐ 3/01) |
| Cache blocking | Shared-memory tiling (GĐ 3/04) |
| Cache line | Memory sector / transaction (GĐ 3/01) |
| Lane AVX2 | Lane của warp (GĐ 2/01) |
| Nhiều biến tích luỹ | Song song mức lệnh, occupancy (GĐ 3/15) |
| Tràn thanh ghi | `__launch_bounds__`, local memory (GĐ 4/11) |
| Cường độ số học, roofline | Biểu đồ roofline của Nsight Compute (GĐ 3/16) |
| RAII, move semantics | Lớp device buffer (GĐ 2/12) |
| Cấp phát rất đắt | Memory pool, `cudaMallocAsync` |

Chỉ nên bỏ qua giai đoạn này nếu bạn đã có thể nói ngay kích thước cache line,
băng thông DRAM duy trì được và điểm gãy của máy mình.

## Danh sách bài tập

| # | Bài tập | Độ khó | Bạn xây dựng gì |
|---|---|---|---|
| 01 | [matmul-naive-cpu](exercises/01-matmul-naive-cpu/README.vi.md) | ⭐ | Mốc cơ sở mà mọi bản matmul sau này so với, kèm một benchmark trung thực |
| 02 | [matmul-cache-blocking](exercises/02-matmul-cache-blocking/README.vi.md) | ⭐⭐ | Đổi thứ tự `ikj` và tiling — 10 lần chỉ nhờ bố cục bộ nhớ |
| 03 | [memory-pool-allocator](exercises/03-memory-pool-allocator/README.vi.md) | ⭐⭐ | Một arena và một pool allocator với free list nhúng |
| 04 | [modern-cpp-toolkit](exercises/04-modern-cpp-toolkit/README.vi.md) | ⭐⭐ | `Matrix<T>`: RAII, move, template, lambda — chứng minh không tốn chi phí |
| 05 | [cache-and-bandwidth](exercises/05-cache-and-bandwidth/README.vi.md) | ⭐⭐ | Đo cache line, các cấp cache và băng thông DRAM của bạn |
| 06 | [simd-intrinsics](exercises/06-simd-intrinsics/README.vi.md) | ⭐⭐⭐ | Tự viết AVX2; vì sao SIMD vô dụng với mã nghẽn bộ nhớ |
| 07 | [roofline-model](exercises/07-roofline-model/README.vi.md) | ⭐⭐⭐ | Đường roofline của máy bạn, dựng từ số liệu đo |
| 08 | [gpu-device-query](exercises/08-gpu-device-query/README.vi.md) | ⭐ | Đỉnh lý thuyết, điểm gãy và ngân sách occupancy của GPU bạn |

Bài 01–07 là C++ thuần và không cần GPU. Bài 08 thì cần.

## Build và chạy

```bash
./scripts/build.sh
ctest --test-dir build -L p1 --output-on-failure
```

## Checklist

- [ ] Tôi giải thích được vì sao matmul `ijk` chậm hơn `ikj` khoảng 10 lần
- [ ] Tôi biết kích thước cache line của mình **vì tôi đã tự đo**
- [ ] Tôi biết băng thông DRAM duy trì được và điểm gãy đơn nhân của máy mình
- [ ] Tôi viết được allocator không có `free` riêng lẻ, và nói được khi nào chấp nhận được
- [ ] Tôi viết được lớp RAII move được và chứng minh phép move không cấp phát
- [ ] Tôi giải thích được vì sao SIMD giúp bài đa thức nhưng không giúp SAXPY
- [ ] Tôi phân loại được kernel là nghẽn bộ nhớ hay nghẽn tính toán trước khi động vào nó
- [ ] Tôi biết băng thông đỉnh, FP32 đỉnh, điểm gãy và ngân sách thanh ghi/thread của GPU mình

## Cột mốc

Hãy ghi lại, cho chính máy của bạn: kích thước cache line, băng thông duy trì
được, điểm gãy đơn nhân, và bốn con số GPU từ bài 08. Giai đoạn 3 mặc định bạn đã
có chúng trong tay.
