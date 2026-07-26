<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 2 · CUDA cơ bản

> **4–6 tuần · ⭐⭐ · 12 bài tập** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Giai đoạn này để làm gì

Viết những kernel **đúng**, và đo xem chúng thật sự tốn bao nhiêu. Tối ưu là việc
của Giai đoạn 3. Ở đây mục tiêu là: mô hình thực thi, quản lý bộ nhớ thiết bị, số
học chỉ số, và báo cáo trung thực băng thông hữu hiệu.

Có một kết quả trong giai đoạn này khiến gần như ai cũng bất ngờ, nên nói trước
luôn: **với các phép tính đơn giản theo từng phần tử, tính cả đầu vào lẫn đầu ra
thì GPU chậm hơn CPU**, vì vòng đi-về qua PCIe tốn hơn phần tính toán tiết kiệm
được. Biết chính xác khi nào cán cân đảo chiều là một phần lớn của việc dùng GPU
cho có ích.

## Danh sách bài tập

| # | Bài tập | Độ khó | Ý tưởng cốt lõi |
|---|---|---|---|
| 01 | [hello-cuda](exercises/01-hello-cuda/README.vi.md) | ⭐ | `__global__`, cú pháp launch, chỉ số toàn cục, kiểm tra biên, lỗi bất đồng bộ |
| 02 | [thread-indexing](exercises/02-thread-indexing/README.vi.md) | ⭐ | Grid 1D/2D/3D và grid-stride loop |
| 03 | vector-add | ⭐ | Trọn chu trình bộ nhớ — và vì sao PCIe thường thắng |
| 04 | saxpy | ⭐⭐ | Băng thông hữu hiệu là thước đo *chính* cho kernel nghẽn bộ nhớ |
| 05 | error-handling | ⭐⭐ | Lỗi dính, lỗi đồng bộ vs bất đồng bộ, `compute-sanitizer` |
| 06 | matrix-add-2d | ⭐ | Launch 2D trên dữ liệu 2D thật |
| 07 | matmul-naive | ⭐⭐ | Mốc cơ sở GPU cho mọi bản matmul sau này |
| 08 | transpose-naive | ⭐⭐ | Một kernel đúng nhưng chậm — dọn đường cho Giai đoạn 3 |
| 09 | rgb-to-grayscale | ⭐⭐ | Dữ liệu ảnh, đọc/ghi PPM, song song theo điểm ảnh |
| 10 | box-blur | ⭐⭐ | Stencil, vành halo, xử lý biên |
| 11 | unified-memory | ⭐⭐ | `cudaMallocManaged`, di trú trang, prefetch |
| 12 | device-buffer-class | ⭐⭐ | RAII bọc quanh bộ nhớ thiết bị — áp dụng GĐ 1/04 |

## Build và chạy

```bash
./scripts/build.sh
ctest --test-dir build -L p2 --output-on-failure
```

## Checklist

- [ ] Tôi viết được `blockIdx.x * blockDim.x + threadIdx.x` mà không cần nghĩ
- [ ] Tôi kiểm tra biên trong mọi kernel, và biết chuyện gì xảy ra nếu không làm thế
- [ ] Tôi kiểm tra cả lỗi launch *lẫn* lỗi thực thi, và phân biệt được hai loại
- [ ] Tôi cấp phát, sao chép và giải phóng bộ nhớ thiết bị mà không rò rỉ
- [ ] Tôi dùng grid-stride loop như mặc định
- [ ] Tôi báo cáo băng thông hữu hiệu, không phải mili-giây
- [ ] Tôi giải thích được khi nào thì *không* đáng dùng GPU cho một kernel
- [ ] Tôi đã dùng `compute-sanitizer` trên một kernel thật sự có lỗi

## Cột mốc

Cho một bài toán mới theo từng phần tử hoặc theo 2D, bạn viết được kernel đúng,
kiểm chứng nó với bản tham chiếu CPU, báo cáo băng thông hữu hiệu theo phần trăm
đỉnh của GPU mình, và nói được liệu chi phí truyền dữ liệu có làm cả việc này bõ
công hay không.
