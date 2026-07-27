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
| 01 ✅ | [pinned-and-streams](exercises/01-pinned-and-streams/README.vi.md) | ⭐⭐⭐ | Bộ nhớ ghim, và biến `H2D + kernel + D2H` thành `max(...)` |
| 02 ⚠️ | [cuda-graphs](exercises/02-cuda-graphs/README.vi.md) | ⭐⭐⭐⭐ | Phát lại một DAG đã ghi bằng một lệnh launch; đưa CPU ra khỏi vòng lặp trong |
| 03 ⚠️ | [atomics-and-ordering](exercises/03-atomics-and-ordering/README.vi.md) | ⭐⭐⭐⭐⭐ | Riêng tư hoá, `atomicCAS`, và vì sao `__threadfence` không phải tuỳ chọn |

✅ đã kiểm chứng trên RTX 3060 · ⚠️ mã hoàn chỉnh, chưa chạy trên máy tham chiếu
(nên README của nó không có số đo nào)

### Tạm hoãn

Đã đặc tả trong [docs/ROADMAP.vi.md](../docs/ROADMAP.vi.md) nhưng chưa viết:
**events-and-sync**, **cooperative-groups**, **dynamic-parallelism**,
**multi-gpu-basics**, **multi-gpu-matmul**, **lock-free-queue**,
**register-pressure**, **ptx-and-sass**, **persistent-kernel**.

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
