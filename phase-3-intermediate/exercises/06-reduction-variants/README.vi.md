<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 06 - Reduction song song, bảy cách

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~3 giờ | Yêu cầu: [02](../02-shared-memory-basics/README.vi.md), [03](../03-bank-conflicts/README.vi.md), [GĐ 1/07](../../../phase-1-foundation/exercises/07-roofline-model/README.vi.md) | **Cần GPU**

## Mục tiêu

Gom *n* giá trị về một, bảy lần. Mọi phiên bản đều đúng; mỗi cái nhanh hơn cái
trước vì một lý do **khác nhau**. Gọi tên được lý do đó chính là bài tập — đây là
bài luyện tối ưu hay nhất trên GPU.

## Kiến thức nền

Reduction đọc mỗi phần tử đầu vào đúng một lần và gần như không ghi gì, nên nó
thuần tuý nghẽn bộ nhớ. Trần của nó là một phép copy: ~330 GB/s trên RTX 3060, tức
**0.373 ms cho 128 MB**. Mọi biến thể đều được chấm điểm so với cái sàn đó, chứ
không phải so với biến thể liền trước.

| Phiên bản | Sửa được gì |
|---|---|
| v1 interleaved, `tid % (2*s) == 0` | — (mốc cơ sở) |
| v2 interleaved, `index = 2*s*tid` | **phân kỳ warp** |
| v3 sequential addressing | **xung đột bank** |
| v4 cộng ngay khi nạp | **thread nhàn rỗi** |
| v5 bung warp cuối (`__shfl_down_sync`) | **rào chắn đồng bộ** |
| v6 quét grid-stride | đổi **hình dạng** |
| v7 grid-stride, 4 biến tích luỹ | **chuỗi phụ thuộc** |

**Về warp cuối cùng:** mẹo kinh điển là bỏ `__syncthreads()` và đánh dấu mảng
shared là `volatile`, dựa vào việc warp chạy đồng bộ. Điều đó đã **không còn an
toàn từ Volta**, nơi các thread trong warp phân kỳ độc lập. Hãy dùng
`__shfl_down_sync`, nó chuyển giá trị thẳng giữa thanh ghi của các lane.

## Nhiệm vụ của bạn

Mở `main.cu` và viết cả bảy. **Hãy dự đoán trước mỗi phép đo** — đặc biệt là trước
v6: nó trông rõ ràng tốt hơn v5. Có đúng vậy không?

Chỉ nên làm v7 sau khi bạn đã đo v6 và thấy bất ngờ.

## Build và chạy

```bash
cmake --build build --target p3_06_reduction_variants -j
./build/bin/p3/p3_06_reduction_variants
```

## Kết quả mong đợi

RTX 3060, 32 triệu float:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
v1 interleaved, divergent             1.805     74.35     1.00x
v2 interleaved, no divergence         1.291    103.95     1.40x
v3 sequential addressing              1.245    107.79     1.45x
v4 first add during load              0.643    208.71     2.81x
v5 unrolled last warp                 0.398    336.95     4.53x
v6 grid-stride + shuffle              0.483    277.69     3.74x     ← chậm hơn!
v7 grid-stride, 4 accumulators        0.396    338.69     4.56x

  Best variant : v7   0.396 ms  (338.7 GB/s, 94% of peak)
  Floor        :      0.373 ms  (reading 128 MB once at peak)
```

### Điều bất ngờ: v6 chậm hơn v5 1.21 lần

Đây là kết quả hữu ích nhất của bài, và nguyên nhân là thứ bạn đã từng đo — trên
**CPU**, ở [GĐ 1/07](../../../phase-1-foundation/exercises/07-roofline-model/README.vi.md).

Vòng quét của v6 cộng dồn vào **một biến duy nhất**. Mỗi thread thực hiện khoảng
580 phép cộng và mỗi phép đều phải chờ kết quả của phép trước. Lệnh FMA phát được
mỗi chu kỳ nhưng mất vài chu kỳ mới xong, nên một chuỗi phụ thuộc đơn chỉ chạy được
một phần nhỏ so với đỉnh. **Kernel bị nghẽn độ trễ chứ không phải nghẽn băng
thông**, và không phép tinh chỉnh bộ nhớ nào chữa được điều đó.

v7 dùng **bốn biến tích luỹ độc lập** — đúng cách sửa đó — và đạt 94% đỉnh.

## Điểm cốt lõi

- **Phân kỳ, xung đột bank, thread nhàn rỗi, rào chắn và chuỗi phụ thuộc là năm vấn
  đề riêng biệt.** Sửa cái này không sửa cái kia, và mỗi cái có dấu hiệu riêng.
- **Hãy chấm điểm so với cái sàn, không phải so với phiên bản trước.** "Nhanh hơn
  1.21 lần" là vô nghĩa; "94% đỉnh" mới cho bạn biết nên dừng.
- **Nghẽn độ trễ trông giống vấn đề bộ nhớ nhưng không phải.** Cùng một sai lầm một
  biến tích luỹ khiến bạn trả giá trên cả CPU lẫn GPU.
- **Warp không còn được đồng bộ ngầm từ Volta.** `volatile` + bỏ rào chắn là một
  lỗi, không phải một phép tối ưu.
- **Mỗi biến thể cho một tổng hơi khác nhau.** Phép cộng số thực không có tính kết
  hợp. Nếu bạn cần tái lập được kết quả, bạn cần một thứ tự cố định hoặc phép cộng
  bù sai số Kahan, và cả hai đều tốn hiệu năng.

## Đi xa hơn

- Quét kích thước block 64→1024 cho v7. Điểm tối ưu ở đâu?
- Thử 2 và 8 biến tích luỹ. Tới đâu thì thêm nữa không còn giúp ích, và vì sao?
- Cài một reduction chỉ dùng một kernel với atomic cho bước gộp cuối. Nhanh hơn,
  hay chi phí tranh chấp lớn hơn? (Bài 08 nói về atomic.)
- Thay v7 bằng `cub::BlockReduce` (GĐ 5/02) rồi so sánh. Bạn nên đạt xấp xỉ — CUB
  sinh ra gần đúng đoạn này, được tinh chỉnh theo từng kiến trúc. **Hãy tự viết
  một lần để hiểu, rồi thôi không viết lại nữa.**
