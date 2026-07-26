<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 05 - Cache line và băng thông bộ nhớ

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [01](../01-matmul-naive-cpu/), [02](../02-matmul-cache-blocking/)

## Mục tiêu

Tự **đo** — chứ không phải đọc về — ba con số quyết định tốc độ của gần như mọi
chương trình thực tế: **kích thước cache line**, **ranh giới các cấp cache**, và
**băng thông DRAM duy trì được** của máy bạn. Đây là buổi tổng duyệt trên CPU cho
[Giai đoạn 3 bài 01 (memory coalescing)](../../../phase-3-intermediate/exercises/01-memory-coalescing/).

## Kiến thức nền

Sự thật duy nhất đứng sau bài này:

> **Bộ nhớ không được chuyển theo từng byte, mà theo từng khối.**

CPU x86 chuyển các **cache line** 64 byte. GPU NVIDIA chuyển các **sector** 32
byte, gộp thành giao dịch 128 byte. Bạn xin một `float` thì phần cứng vẫn nạp cả
khối. Nếu sau đó bạn chỉ dùng đúng một float ấy, bạn đã vứt đi 15/16 băng thông —
mà băng thông, chứ không phải phép tính, mới là thứ giới hạn hầu hết các kernel.

**Thí nghiệm 1 — quét stride.** Cộng một mảng nhưng chỉ chạm vào mỗi phần tử thứ
`stride`. Báo cáo hai loại băng thông: *useful* (số byte chương trình thật sự cần)
và *DRAM* (số byte nguyên cache line mà bus đã chuyển). Khi stride tăng từ 1 lên
16 float, băng thông useful sụp khoảng 16 lần trong khi băng thông DRAM gần như
không đổi. Bus vẫn bận y như cũ; nó đang chuyển dữ liệu mà bạn vứt đi. Vượt quá
stride 16 thì không mất thêm gì nữa — bạn vốn đã phí trọn một line cho mỗi lần
truy cập rồi.

**Thí nghiệm 2 — quét kích thước tập làm việc.** Đọc đi đọc lại các mảng có kích
thước tăng dần. Mảng nhỏ nằm trong L1, rồi L2, rồi L3, rồi rơi ra DRAM. Những bậc
thang trong cột GB/s *chính là* kích thước cache của bạn. Bạn đọc được chúng mà
không cần tra thông số kỹ thuật của CPU.

**Thí nghiệm 3 — STREAM triad.** `a[i] = b[i] + s*c[i]`: ba mảng, một lượt duyệt,
không tái sử dụng gì. Đây là cách chuẩn để đo băng thông duy trì. Hãy so kết quả
với con số GPU mà bất kỳ bài nào ở Giai đoạn 2 in ra — tỉ lệ đó chính là lý do
tồn tại của lộ trình này.

## Nhiệm vụ của bạn

Mở `main.cpp`:

1. **`strideSum()`** — cộng `data[0], data[stride], data[2*stride], ...`
   Nhớ **trả về** tổng. Một benchmark có kết quả không được dùng sẽ bị trình tối
   ưu xoá sạch, và bạn chỉ đang đo một vòng lặp rỗng.
2. **`triad()`** — `a[i] = b[i] + scalar * c[i]`.
3. **Quét stride** — đo từng stride trong `kStrides` và in cả hai cột băng thông,
   dùng `usefulBytes()` / `dramBytes()` trong `reference.h`.
4. **Benchmark triad** — 3 mảng × 4 byte = 12 byte lưu lượng cho mỗi phần tử.

## Build và chạy

```bash
cmake --build build --target p1_05_cache_and_bandwidth -j
./build/bin/p1/p1_05_cache_and_bandwidth
```

## Kết quả mong đợi

```
--- Stride sweep ---
  stride     time (ms)    useful GB/s      DRAM GB/s  efficiency
  ------------------------------------------------------------------
  1              12.50           21.5           21.5        100%
  2               9.80           13.7           27.4         50%
  4               8.90            7.5           30.2         25%
  8               8.60            3.9           31.2         13%
  16              8.50            2.0           31.5          6%
  32              4.30            2.0           31.2          6%
```

Để ý cột **useful** giảm một nửa sau mỗi bước trong khi cột **DRAM** đứng yên. Con
số tuyệt đối của bạn sẽ khác; *hình dạng* của bảng mới là kết quả.

## Điểm cốt lõi

- Phần cứng chuyển dữ liệu theo **khối cố định**. Dùng nửa vời một khối là phí
  băng thông, và không có thủ thuật tính toán nào lấy lại được.
- Băng thông hữu hiệu là `số byte hữu ích / thời gian`, và đó mới là con số đáng
  quan tâm — một kernel có thể làm nghẽn bus mà chỉ đạt 6% thông lượng hữu ích.
- **Các cấp cache là thứ đo được**, không phải lý thuyết suông. Quét tập làm việc
  rồi đọc ra.
- Trên GPU, khối là sector 32 byte và "stride" là khoảng cách giữa các địa chỉ mà
  các thread liền kề trong một warp chạm tới. Cùng một đồ thị, cùng một bài học.

## Đi xa hơn

- Thêm biến thể truy cập ngẫu nhiên (xáo trộn chỉ số). Nó tệ hơn stride 128 bao
  nhiêu, và vì sao?
- Lặp lại triad với 2, 4, 8 luồng. Băng thông bão hoà từ lâu trước khi bạn dùng
  hết số nhân — giới hạn nằm ở bộ điều khiển bộ nhớ, không phải ở CPU.
- Đọc Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007) —
  đến giờ vẫn là tài liệu chuẩn mực về chủ đề này.
