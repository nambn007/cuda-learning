<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 04 - Nhân ma trận có tiling

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~3 giờ | Yêu cầu: [02](../02-shared-memory-basics/README.vi.md), [03](../03-bank-conflicts/README.vi.md), [GĐ 2/07](../../../phase-2-cuda-fundamentals/exercises/07-matmul-naive/README.vi.md) | **Cần GPU**

## Mục tiêu

Áp dụng shared-memory tiling cho bài matmul ở Giai đoạn 2 — rồi phát hiện rằng
**đáp án trong sách giáo khoa gần như không mang lại gì**, trong khi một kỹ thuật
thứ hai ít được nhắc tới lại mang lại 3 lần. Hiểu *vì sao* còn giá trị hơn bản thân
cái kernel.

## Kiến thức nền

Giai đoạn 2/07 để lại kernel naive ở ~6% đỉnh FP32 của GPU. Cường độ số học của nó
bị cố định ở

```
2N FLOP / (2N × 4 byte) = 0.25 FLOP trên mỗi byte
```

bất kể `N` lớn tới đâu. So với điểm gãy gần 37, đó là vô vọng.

**Tiling lẽ ra phải sửa được điều đó.** Nạp một khối `TILE×TILE` của `A` và `B` vào
shared memory thì mỗi giá trị nạp về phục vụ `TILE` thread thay vì 1:

```
cường độ = 2 × TILE / (2 × 4 byte) = TILE/4 = 8 FLOP/byte với TILE = 32
```

Trên giấy là cải thiện 32 lần. Vòng lặp chính là cache blocking của GĐ 1/02, với
`__shared__` đóng vai cache — chỉ khác là bạn tự đặt dữ liệu.

**Cả hai rào chắn đều cần thiết.** Một cái sau khi nạp tile, một cái sau khi dùng
xong. Bỏ cái thứ hai là lỗi kinh điển: thread nhanh bắt đầu nạp tile kế tiếp trong
khi thread chậm còn đang đọc tile hiện tại. Nó thường vẫn cho kết quả đúng với dữ
liệu nhỏ, và đó chính là chỗ nguy hiểm.

**Ở đây không cần đệm.** Trong vòng lặp trong cùng, với một warp (`tx` biến thiên,
`ty` cố định): `As[ty][k]` là cùng một địa chỉ cho cả 32 thread — broadcast, miễn
phí — còn `Bs[k][tx]` trải trên 32 word liên tiếp — 32 bank khác nhau, miễn phí.
Bài 05 mới là nơi việc đệm trở nên thiết yếu.

## Nhiệm vụ của bạn

Mở `main.cu` và viết ba kernel: `matmulNaive`, `matmulTiled<TILE>`, và
`matmulTiledRegister<TILE, TM>` trong đó mỗi thread giữ `TM` biến tích luỹ trong
thanh ghi và tính `TM` hàng đầu ra.

**Hãy dự đoán trước khi đo.** Tiling đơn thuần cắt số lần nạp global trên mỗi phần
tử đầu ra đi 32 lần; bạn kỳ vọng nhanh hơn bao nhiêu? Register tiling **không** hề
giảm lưu lượng global; bạn kỳ vọng nó mang lại bao nhiêu?

Một trong hai dự đoán đó có lẽ sai nặng. Tìm ra cái nào chính là bài tập.

## Build và chạy

```bash
cmake --build build --target p3_04_tiled_matmul -j
./build/bin/p3/p3_04_tiled_matmul
```

## Kết quả mong đợi

RTX 3060, `n = 1024`:

```
Variant                           Time (ms)    GFLOP/s   Speedup
----------------------------------------------------------------
CPU ikj (1 thread)                  152.269      14.10     1.00x
GPU naive                             2.688     798.92    56.65x
GPU tiled (shared)                    2.235     961.02    68.14x
GPU tiled + register                  0.847    2535.47   179.78x

  variant                       loads/output   FLOP per byte   % of peak
  ----------------------------------------------------------------------
  naive                                 2048            0.25          6%
  tiled (TILE=32)                         64            8.00          7%
  tiled + register (TM=4)                 64            8.00         19%
```

### Kết quả mà sách giáo khoa không chuẩn bị cho bạn

**Tiling đơn thuần chỉ mua được 1.20 lần**, dù đã giảm 32 lần số lệnh nạp global
trên mỗi phần tử đầu ra.

Lý do chính là điều bạn đã gặp ở GĐ 2/07: 2048 lệnh nạp mỗi phần tử của bản naive
vốn chưa bao giờ thật sự xuống tới DRAM. **L1 và L2 đã nắm gần hết phần tái sử dụng
đó rồi.** Viết tiling bằng tay chủ yếu là thay một cache tự động bằng một cache thủ công.

**Register tiling mua được 3.0 lần** — và nó là một *loại thay đổi khác*. Hãy nhìn
bảng: cột `loads/output` giống hệt nhau ở cả hai bản tiled. Nó không hề chuyển ít
dữ liệu hơn. Thứ nó làm là cho mỗi thread nhiều việc độc lập hơn:

- mỗi giá trị đọc từ **shared** memory nay phục vụ 4 phép nhân-cộng
- 4 biến tích luỹ nằm trong thanh ghi, nên pipeline FMA có 4 chuỗi độc lập và thôi
  bị nghẽn bởi chính độ trễ của nó — đúng hiệu ứng bạn đã đo trên CPU ở
  [GĐ 1/07](../../../phase-1-foundation/exercises/07-roofline-model/README.vi.md)
- block thu từ 1024 xuống 256 thread, giải phóng thanh ghi và khe lập lịch

> **Bài học mang theo:** trên GPU hiện đại, shared memory hiếm khi có giá trị vì nó
> "nhanh hơn global" — cache đã cho bạn điều đó rồi. Nó có giá trị vì cho phép bạn
> **tái cấu trúc** phép tính sao cho mỗi thread làm nhiều việc hơn trên mỗi byte nó
> chạm tới. Việc tái cấu trúc mới là điểm mấu chốt; shared memory chỉ là thứ khiến
> điều đó khả thi.

## Điểm cốt lõi

- **Giảm 32 lần số lệnh nạp mà chỉ nhanh hơn 1.2 lần.** Hãy luôn kiểm tra xem cache
  đã giải quyết hộ bạn chưa.
- **Khối lượng việc trên mỗi thread quan trọng ngang lưu lượng trên mỗi thread.**
  Register tiling thêm ILP, và ILP là thứ lấp đầy một pipeline bị nghẽn độ trễ.
- **Hai rào chắn cho mỗi vòng tile**, và lỗi thiếu rào chắn thứ hai im lặng với dữ
  liệu nhỏ.
- 19% đỉnh, so với 80–90% của cuBLAS. Phần còn lại vẫn là ý tưởng đó: register tile
  rộng hơn, nạp `float4`, double buffering, tinh chỉnh theo từng kiến trúc.

## Đi xa hơn

- Quét `TM` = 1, 2, 4, 8. Tới đâu thì nó ngừng giúp ích, và cái gì chặn nó lại?
  (`-DCL_PTXAS_VERBOSE=ON` hiển thị lượng thanh ghi dùng.)
- Thử `TILE` = 16 và 64. Cái gì giới hạn kích thước tile?
- Thêm lệnh nạp `float4` cho phần điền tile.
- Double-buffer các tile: nạp tile `t+1` trong lúc đang tính trên tile `t`.
- **Tiếp theo:** [05 transpose-optimized](../05-transpose-optimized/) dùng shared
  memory để *sắp xếp lại* thay vì để tái sử dụng — và ở đó phần đệm của bài 03 là
  thiết yếu. [GĐ 5/03](../../../phase-5-expert/exercises/03-cublas-gemm/) đo khoảng
  cách tới cuBLAS.
