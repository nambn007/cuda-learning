<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 4 · Nâng cao

> **8–10 tuần · ⭐⭐⭐⭐ · 13 bài tập** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Giai đoạn này để làm gì

Giai đoạn 3 làm cho một kernel chạy nhanh. Giai đoạn 4 nói về mọi thứ *xung quanh*
kernel: chồng lấn công việc để GPU không bao giờ rảnh, mở rộng vượt khỏi một thiết
bị, đồng bộ đúng cách giữa các thread, và đọc được mã máy mà trình biên dịch thật
sự sinh ra.

Có một chủ đề đáng gọi tên: **ở mức này, viết đúng khó hơn hẳn.** Stream sinh ra
lỗi thứ tự, atomic sinh ra lỗi trật tự bộ nhớ, còn multi-GPU thì sinh ra cả hai.
`compute-sanitizer --tool racecheck` trở thành công cụ dùng thường xuyên chứ không
phải phương án cuối cùng.

## Danh sách bài tập

| # | Bài tập | Độ khó | Ý tưởng cốt lõi |
|---|---|---|---|
| 01 | pinned-memory | ⭐⭐ | Băng thông truyền: bộ nhớ pageable vs pinned |
| 02 | streams-basics | ⭐⭐⭐ | Chạy đồng thời, và cái bẫy của default stream |
| 03 | streams-pipeline | ⭐⭐⭐ | Chồng lấn H2D → kernel → D2H theo từng khối |
| 04 | events-and-sync | ⭐⭐⭐ | `cudaEvent`, `cudaStreamWaitEvent`, tự dựng đồ thị phụ thuộc |
| 05 | cuda-graphs | ⭐⭐⭐⭐ | Ghi lại một pipeline, xoá bỏ chi phí launch từng lệnh |
| 06 | cooperative-groups | ⭐⭐⭐⭐ | Phân hoạch theo tile, đồng bộ toàn grid |
| 07 | dynamic-parallelism | ⭐⭐⭐⭐ | Kernel gọi kernel (ngữ nghĩa CDP2, CUDA 12+) |
| 08 | multi-gpu-basics | ⭐⭐⭐ | Liệt kê thiết bị, truy cập ngang hàng, sao chép P2P |
| 09 | multi-gpu-matmul | ⭐⭐⭐⭐ | Chia việc trên nhiều thiết bị |
| 10 | lock-free-queue | ⭐⭐⭐⭐⭐ | `atomicCAS`, `__threadfence`, trật tự bộ nhớ trên GPU |
| 11 | register-pressure | ⭐⭐⭐ | `__launch_bounds__`, tràn thanh ghi, đánh đổi occupancy |
| 12 | ptx-and-sass | ⭐⭐⭐⭐ | `cuobjdump`, `nvdisasm`, PTX nội tuyến |
| 13 | persistent-kernel | ⭐⭐⭐⭐ | Megakernel và producer/consumer trên thiết bị |

Bài 08 và 09 tự phát hiện máy chỉ có một GPU và bỏ qua gọn gàng — đó không phải lỗi.

## Build và chạy

```bash
./scripts/build.sh
ctest --test-dir build -L p4 --output-on-failure
```

## Checklist

- [ ] Tôi chồng lấn được một lần truyền dữ liệu với một kernel và chứng minh được trên timeline
- [ ] Tôi biết vì sao default stream làm mọi thứ tuần tự, và cách tránh
- [ ] Tôi biểu diễn được phụ thuộc giữa hai stream mà không cần đồng bộ toàn bộ
- [ ] Tôi chuyển được một chuỗi launch lặp đi lặp lại thành một CUDA graph
- [ ] Tôi giải thích được khi nào cần `__threadfence()` và khi nào không
- [ ] Tôi đọc được danh sách SASS và tìm ra lệnh gây nghẽn
- [ ] Tôi đã dùng `__launch_bounds__` để đánh đổi thanh ghi lấy occupancy, và đo cả hai
- [ ] Tôi đã tìm ra một race thật bằng `compute-sanitizer --tool racecheck`

## Cột mốc

Bạn nhận một pipeline nhiều tầng, chồng lấn được các tầng của nó, giữ mọi GPU sẵn
có đều bận, và biện minh được cho từng điểm đồng bộ trong đó.
