<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 03 - Cộng vector, và cái giá của việc đưa dữ liệu sang GPU

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐ | Thời gian: ~1 giờ | Yêu cầu: [02](../02-thread-indexing/README.vi.md) | **Cần GPU**

## Mục tiêu

Viết bài "hello world" của lập trình GPU — rồi phát hiện ra rằng **GPU thua cuộc**.
Đây là kết quả tiêu cực hữu ích nhất trong cả lộ trình: nó dạy bạn GPU thật sự
dùng để làm gì, trước khi bạn mất hàng tuần tối ưu một thứ lẽ ra không nên đưa lên
GPU ngay từ đầu.

## Kiến thức nền

Kernel rất tầm thường: `c[i] = a[i] + b[i]`. Với mỗi phần tử, nó đọc 8 byte, ghi 4
byte, và làm một phép cộng. Cường độ số học là **1/12 = 0.083 FLOP/byte**. Điểm
gãy của GPU bạn (đo ở
[GĐ 1/08](../../../phase-1-foundation/exercises/08-gpu-device-query/README.vi.md))
vào khoảng **37**. Kernel này nằm hẳn ngoài rìa trái của đường roofline — thuần
tuý là lưu lượng bộ nhớ, kèm theo một chút phép tính không đáng kể.

Riêng điều đó thì không sao; GPU có ~360 GB/s so với ~17 GB/s của CPU. Vấn đề nằm
ở chỗ đưa dữ liệu sang. **PCIe chạy khoảng 8–12 GB/s**, chậm hơn bộ nhớ của chính
GPU khoảng 30–40 lần. Nên chuỗi thao tác

```
chép a (64 MB qua PCIe) → chép b (64 MB) → kernel → chép c về (64 MB)
```

dành gần như toàn bộ thời gian cho các mũi tên, chứ không phải cho kernel.

Vì vậy phép đo có ý nghĩa không phải là "kernel nhanh cỡ nào" mà là "toàn bộ việc
này nhanh cỡ nào, tính từ nơi dữ liệu thật sự đang nằm". Các bài blog trích dẫn
con số thứ nhất. Hệ thống thật sống chết bằng con số thứ hai.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`vectorAdd()`** — dùng grid-stride loop.
2. **Bốn bước** — `cudaMalloc` ×3, `cudaMemcpy` H2D ×2, launch, `cudaMemcpy` D2H,
   `cudaFree` ×3. Dùng `size_t` cho số byte; `int n * sizeof(float)` tràn âm thầm
   ở mốc 512 triệu phần tử.
3. **Ba phép đo** — bản CPU, kernel đơn thuần, và trọn vòng đi-về. Rồi tự quyết
   xem bạn sẽ đưa con số nào vào báo cáo.

## Build và chạy

```bash
cmake --build build --target p2_03_vector_add -j
./build/bin/p2/p2_03_vector_add
```

## Kết quả mong đợi

Đo trên RTX 3060 với `n = 16 triệu`:

```
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
CPU (single thread)                  13.071     15.40       1.28     1.00x
GPU kernel only                       0.627    321.25      26.77    20.86x
GPU + PCIe round trip                24.365      8.26       0.69     0.54x

  Kernel efficiency : 321.3 of 360.0 GB/s = 89% of theoretical peak

  host -> device (64 MB)               8.06 ms      8.3 GB/s
  device -> host (64 MB)               7.81 ms      8.6 GB/s
  kernel                               0.63 ms

    The GPU is 1.86x SLOWER than the CPU once transfers are counted,
    even though the kernel itself is 20.9x faster.
```

Hãy đọc kỹ ba dòng đó. Kernel đang chạy ở **89% đỉnh lý thuyết của phần cứng** —
không còn gì để tối ưu nữa, nó đã nhanh bằng tốc độ tối đa mà GPU này chuyển được
bộ nhớ. Vậy mà toàn bộ thao tác vẫn **chậm hơn một luồng CPU 1.86 lần**.

## Điểm cốt lõi

- **Kernel không phải là cả chương trình.** Trích dẫn thời gian kernel đơn thuần
  là cách phổ biến nhất để nói dối bằng benchmark GPU — thường là tự dối mình.
- **PCIe là nút thắt** với bất cứ thứ gì không nằm thường trú trên GPU: ~8 GB/s so
  với ~360 GB/s của GPU.
- **Đạt 89% đỉnh nghĩa là xong.** Khi một kernel đã chạm roofline, hãy ngừng tối ưu
  nó và thay đổi thuật toán hoặc luồng dữ liệu.
- GPU đáng dùng khi: dữ liệu **nằm lại** trên thiết bị qua nhiều kernel; cường độ
  số học **cao** (matmul, GĐ 3/04); truyền dữ liệu **chồng lấn** với tính toán
  (stream, GĐ 4/03); hoặc nhiều thao tác được **gộp** vào một lượt duyệt (GĐ 5/14).

## Đi xa hơn

- Chạy lại với `n = 1 << 20` và `n = 1 << 26`. Tỉ lệ có đổi không? Vì sao không?
- Thay `cudaMalloc` + `cudaMemcpy` bằng `cudaMallocManaged`. Nhanh hơn hay chậm
  hơn? (GĐ 2/11 khảo sát kỹ chuyện này.)
- Dùng bộ nhớ host ghim (`cudaMallocHost`). Băng thông PCIe tăng khoảng gấp đôi —
  đó là nội dung GĐ 4/01.
- Xâu chuỗi mười phép cộng vector trên thiết bị trước khi chép kết quả về. Tới mốc
  nào thì GPU thắng? Con số đó chính là câu trả lời thật cho câu hỏi "cái này có
  nên chạy trên GPU không".
