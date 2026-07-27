<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 3 · Trung cấp

> **6–8 tuần · ⭐⭐⭐ · 16 bài tập** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Giai đoạn này để làm gì

Đây là trái tim của lộ trình: nơi "nó chạy được" biến thành "nó chạy nhanh", và là
nơi mọi thứ bạn đo được ở Giai đoạn 1 được đền đáp.

Thứ tự các bài là có chủ ý. Bộ nhớ đến trước (01–05) vì trên GPU gần như mọi thứ
đều nghẽn bộ nhớ — điểm gãy của GPU bạn, đo ở
[GĐ 1/08](../phase-1-foundation/exercises/08-gpu-device-query/README.vi.md), rơi
vào khoảng 37 FLOP/byte, trong khi phép cộng vector chỉ đạt 0.08. Tiếp đến là các
mẫu song song (06–14) mà mọi kernel thực tế đều được dựng từ đó, và cuối cùng là
kỹ năng tinh chỉnh và profiling (15–16) để biết nên chọn mẫu nào.

## Danh sách bài tập

**Tình trạng: đã làm 7/16.** Bài 01–07 hoàn chỉnh và đã kiểm chứng trên RTX 3060.
Bài 08–16 được đặc tả bên dưới nhưng **chưa được viết** — thư mục của chúng chưa có
trong cây mã, và `scripts/new-exercise.sh` sẽ dựng khung khi bạn hoặc người đóng góp
bắt tay vào.

| # | Bài tập | Độ khó | Ý tưởng cốt lõi |
|---|---|---|---|
| 01 ✅ | memory-coalescing | ⭐⭐⭐ | Yếu tố hiệu năng lớn nhất của GPU, đo tận tay |
| 02 ✅ | shared-memory-basics | ⭐⭐ | `__shared__`, `__syncthreads()`, hợp tác trong một block |
| 03 ✅ | bank-conflicts | ⭐⭐⭐ | 32 bank; vì sao đệm thêm một phần tử sửa được mức chậm 32 lần |
| 04 ✅ | tiled-matmul | ⭐⭐⭐ | Blocking của GĐ 1/02 đặt trong shared memory — và vì sao thắng lợi đến từ register tiling chứ không từ bản thân việc tiling |
| 05 ✅ | transpose-optimized | ⭐⭐⭐ | Đọc *và* ghi đều coalesced nhờ một tile chia sẻ |
| 06 ✅ | reduction-variants | ⭐⭐⭐ | Sáu kernel, cái sau nhanh hơn cái trước — bài nghiên cứu kinh điển |
| 07 ✅ | warp-shuffle | ⭐⭐⭐ | `__shfl_down_sync`, ballot, vote — dùng thanh ghi thay shared memory |
| 08 🚧 | atomics | ⭐⭐ | `atomicAdd`, `atomicCAS`, tranh chấp, atomic float tự viết |
| 09 🚧 | histogram | ⭐⭐⭐ | Riêng tư hoá: tranh chấp toàn cục thành tranh chấp trong shared memory |
| 10 🚧 | scan-hillis-steele | ⭐⭐⭐ | Quét tiền tố bao gồm, trong phạm vi một block |
| 11 🚧 | scan-blelloch | ⭐⭐⭐⭐ | Quét hiệu quả công việc, độ dài tuỳ ý, nhiều block |
| 12 🚧 | stream-compaction | ⭐⭐⭐ | Scan + scatter — xương sống của việc lọc dữ liệu trên GPU |
| 13 🚧 | conv-1d-constant | ⭐⭐ | Bộ nhớ `__constant__` và cache quảng bá của nó |
| 14 🚧 | conv-2d-shared | ⭐⭐⭐ | Stencil 2D với tile chia sẻ có vành halo |
| 15 🚧 | occupancy-tuning | ⭐⭐⭐ | Thanh ghi vs shared memory vs kích thước block; khi occupancy cao lại hại |
| 16 🚧 | profiling-nsight | ⭐⭐⭐ | Nsight Systems và Nsight Compute trên chính kernel của bạn |

## Build và chạy

```bash
./scripts/build.sh
ctest --test-dir build -L p3 --output-on-failure
```

Bài 16 cần `ncu` (đi kèm CUDA toolkit) và tốt nhất là có cả `nsys`.

## Checklist

- [ ] Tôi giải thích được coalescing cho người khác, bằng những con số tôi tự đo
- [ ] Tôi nhìn cách đánh chỉ số của một kernel là thấy được xung đột bank
- [ ] Bản tiled matmul của tôi nhanh hơn bản naive ít nhất 5 lần
- [ ] Tôi viết được reduction đạt trên 80% băng thông đỉnh
- [ ] Tôi dùng warp shuffle thay shared memory ở những chỗ áp dụng được
- [ ] Tôi cài được scan hiệu quả công việc cho độ dài tuỳ ý
- [ ] Tôi đọc được báo cáo Nsight Compute và gọi tên được yếu tố giới hạn
- [ ] Tôi biết một trường hợp mà *giảm* occupancy lại làm kernel nhanh hơn

## Cột mốc

Cho một kernel chậm, bạn profile được nó, gọi tên được yếu tố giới hạn từ các chỉ
số, áp dụng đúng cách sửa, và chứng minh mức cải thiện bằng phép đo trước/sau.
