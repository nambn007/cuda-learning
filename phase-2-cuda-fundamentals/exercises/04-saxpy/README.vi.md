<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 04 - SAXPY và nạp dữ liệu dạng vector

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1 giờ | Yêu cầu: [03](../03-vector-add/README.vi.md) | **Cần GPU**

## Mục tiêu

Học thước đo duy nhất quyết định một kernel nghẽn bộ nhớ có tốt hay không —
**phần trăm băng thông đỉnh** — và làm quen với `float4`, phép tối ưu rẻ nhất
trong CUDA.

## Kiến thức nền

`y[i] = a * x[i] + y[i]`. Đọc 8 byte, ghi 4 byte, làm một phép FMA. Cường độ số
học 2/12 = 0.167 FLOP/byte, nên như mọi thứ trong Giai đoạn 2, nó nghẽn bộ nhớ
hoàn toàn.

Bài 03 đã xác lập rằng những kernel như vậy bị DRAM giới hạn. Câu hỏi tiếp theo
là: **đã bị giới hạn bởi băng thông, vậy ta có đang dùng hết nó không?**

**Báo cáo đúng con số.** Với kernel nghẽn bộ nhớ, mili-giây là vô nghĩa nếu không
kèm kích thước, và GB/s là vô nghĩa nếu không kèm đỉnh phần cứng. Con số hữu ích
là tỉ lệ:

| % so với đỉnh | Kết luận |
|---|---|
| > 80% | Kernel đã xong. Hãy đi tối ưu thứ khác. |
| 50–80% | Hợp lý với kernel có làm việc thật. |
| < 50% | Có gì đó sai — gần như luôn là mẫu truy cập (GĐ 3/01). |

**Nạp dữ liệu dạng vector.** `float4` là struct dựng sẵn gồm bốn float, căn chỉnh
16 byte. Nạp một cái phát ra **một lệnh bộ nhớ 16 byte duy nhất** thay vì bốn lệnh
4 byte. Số byte di chuyển y hệt; số *lệnh* giảm 4 lần. Với một kernel chẳng có gì
để làm trong lúc chờ, điều đó giữ được nhiều dữ liệu đang bay hơn trên mỗi warp.

Hãy kỳ vọng **vài phần trăm**, không phải một bước nhảy vọt. `float4` chỉ đáng
dùng khi mẫu truy cập đã coalesced sẵn — nó làm một kernel tốt tốt hơn chút, và
không giúp gì cho một kernel dở.

**Căn chỉnh là yêu cầu bắt buộc.** `reinterpret_cast<float4*>` chỉ hợp lệ trên con
trỏ căn chỉnh 16 byte. `cudaMalloc` trả về ít nhất 256 byte căn chỉnh nên con trỏ
gốc là an toàn — nhưng `d_x + 1` thì không, và kiểu lỗi là misaligned-address lúc
chạy.

**Một chi tiết nữa:** `a * x[i] + y[i]` biên dịch thành một lệnh FMA duy nhất — một
phép nhân-cộng với *một* lần làm tròn. Nó chính xác hơn chút so với nhân rồi cộng
riêng, và khác chút so với bản tham chiếu CPU. Vì thế mới dùng `rtol = 1e-5`.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`saxpy()`** — bản vô hướng dùng grid-stride loop.
2. **`saxpyVectorised()`** — vẫn phép đó nhưng trên `float4`, kèm một vòng vô
   hướng xử lý phần dư `n % 4`.
3. **Báo cáo phần trăm so với đỉnh**, dùng `theoreticalBandwidthGBs()` trong
   `cuda_helper.h`.

Nhớ đặt lại `y` trước mỗi lần đo — SAXPY cộng dồn vào nó.

## Build và chạy

```bash
cmake --build build --target p2_04_saxpy -j
./build/bin/p2/p2_04_saxpy
```

## Kết quả mong đợi

RTX 3060, `n = 32 triệu`:

```
--- Correctness ---
  [PASS] scalar saxpy (n=33554432, max abs err 2.384e-07)
  [PASS] vectorised saxpy (float4) (n=33554432, max abs err 2.384e-07)

--- Performance ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
CPU (single thread)                  66.243      6.08       1.01     1.00x
GPU scalar                            1.242    324.13      54.02    53.33x
GPU float4                            1.206    333.80      55.63    54.92x

--- Percentage of peak bandwidth ---
  Theoretical peak      : 360.0 GB/s
  Scalar kernel         : 324.1 GB/s  (90% of peak)
  float4 kernel         : 333.8 GB/s  (93% of peak)
```

Đạt 90% đỉnh từ bản *naive*. Đó là dáng vẻ của một kernel nghẽn bộ nhớ có truy cập
coalesced, và là mức chuẩn mà Giai đoạn 3 sẽ đòi hỏi ở mọi kernel.

## Điểm cốt lõi

- **Phần trăm so với đỉnh là thước đo** cho kernel nghẽn bộ nhớ. Chỉ nói mili-giây
  là chưa nói gì cả.
- **Một kernel coalesced đơn giản đã đạt ~90%.** Khi bạn thấy 20%, vấn đề nằm ở
  mẫu truy cập chứ không phải ở phép tính.
- **`float4` mua thêm vài phần trăm** nhờ giảm số lệnh, không phải giảm số byte.
  Nó là nét hoàn thiện, không phải cách chữa bệnh.
- **Căn chỉnh là yêu cầu về tính đúng đắn**, không phải gợi ý về hiệu năng.
- FMA làm đổi những bit cuối của kết quả. Luôn so sánh số thực với một sai số cho
  phép.

## Đi xa hơn

- Thử cả `float2`. Lợi ích có tỉ lệ thuận với bề rộng không?
- Đọc SASS: `cuobjdump -sass build/bin/p2/p2_04_saxpy_sol | grep LDG`. Đếm số
  `LDG.E` so với `LDG.E.128` trong hai kernel.
- Quét kích thước block từ 64 đến 1024. Đường cong phẳng đến đâu? (GĐ 3/15.)
- Cố tình làm `x` lệch căn chỉnh (`d_x + 1`) rồi xem nó lỗi. Sau đó đọc thông báo
  lỗi dưới `compute-sanitizer`.
