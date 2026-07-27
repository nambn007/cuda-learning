<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 12 - Lớp device buffer theo RAII

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [GĐ 1/04](../../../phase-1-foundation/exercises/04-modern-cpp-toolkit/README.vi.md), [03](../03-vector-add/README.vi.md) | **Cần GPU**

## Mục tiêu

Áp dụng `Matrix<T>` của Giai đoạn 1 vào bộ nhớ thiết bị, và **chứng minh** — bằng
cách đo VRAM còn trống — rằng bản RAII không thể rò rỉ trong khi bản kiểu C thì có.
Đây là bài cuối của Giai đoạn 2, và lớp bạn xây ở đây sẽ được dùng cho phần còn lại
của lộ trình.

## Kiến thức nền

Bộ nhớ host bị rò sẽ được thu hồi khi tiến trình kết thúc. **Một `cudaFree` bị quên
làm rò bộ nhớ thiết bị mà không gì thu hồi cho tới khi context CUDA bị huỷ** — nên
một dịch vụ rò 1 MB mỗi request rốt cuộc sẽ chết với `cudaErrorMemoryAllocation` mà
không manh mối nào về nơi nó đi mất.

Tệ hơn, cấu trúc kiểu C *mời gọi* chính lỗi đó:

```cuda
float* d = nullptr;
cudaMalloc(&d, bytes);
if (somethingWrong) return -1;      // rò rỉ
...
cudaFree(d);
```

Mọi lệnh return sớm, mọi ngoại lệ được ném ra, mọi `goto fail` đều là một chỗ rò.
RAII loại bỏ khả năng đó thay vì nhắc bạn nhớ.

### Ba quyết định thiết kế đáng tranh luận

**Copy bị xoá, không phải được cài đặt.** Một phép sao chép device-to-device 16 MB
diễn ra ngầm đúng là loại thao tác đắt đỏ không bao giờ nên xảy ra một cách tình
cờ. Hãy cung cấp `clone()` tường minh để chi phí hiện rõ ngay tại chỗ gọi.

**Move phải `noexcept`.** Không phải để trang trí: nếu thiếu, `std::vector` sẽ sao
chép thay vì move khi cấp phát lại — và vì copy đã bị xoá, mã đặt những đối tượng
này vào container sẽ **hoàn toàn không biên dịch được**. Chính `noexcept` làm cho
lớp này dùng được.

**Destructor không dùng `CUDA_CHECK`.** `CUDA_CHECK` gọi `exit()`, mà lúc tiến
trình đang dọn dẹp thì context có thể đã biến mất, khi đó `cudaFree` trả về lỗi một
cách chính đáng. Thoát chương trình từ destructor còn tệ hơn chỗ rò mà nó định báo.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`DeviceBuffer<T>`** — constructor, destructor, xoá copy, move `noexcept`,
   `clone()`, `copyFromHost()`, `copyToHost()`, `zero()`, các hàm truy cập. Ném
   `std::out_of_range` khi sao chép vượt kích thước; cắt cụt âm thầm trên GPU chính
   là cách bạn nhận về dữ liệu hỏng sau ba kernel nữa.
2. **`PinnedBuffer<T>`** — cùng khuôn mẫu, bọc quanh `cudaMallocHost` /
   `cudaFreeHost`.
3. **Chứng minh nó** bằng `cudaMemGetInfo()`: đo VRAM trống trước và sau 20 lời gọi
   rò rỉ, rồi sau 20 phạm vi RAII bị tháo dỡ bởi một ngoại lệ được ném ra.

## Build và chạy

```bash
cmake --build build --target p2_12_device_buffer_class -j
./build/bin/p2/p2_12_device_buffer_class
```

## Kết quả mong đợi

RTX 3060, buffer 16 MB:

```
--- Basic use ---
  [PASS] scale by 3 (n=4194304, max abs err 0.000e+00)

--- Leaks ---
  20 early returns from the raw C-style version leaked 320.0 MB
  [PASS] the raw version really does leak
  20 thrown exceptions through the RAII version leaked 0.0 MB
  [PASS] RAII leaks nothing, even when unwinding

--- Move semantics ---
  [PASS] move transfers the pointer
  [PASS] moved-from buffer is empty
  [PASS] clone allocates separate storage

--- Use in containers ---
  [PASS] vector holds four live buffers

--- Pinned host memory ---
pageable host memory                  1.970      8.52 GB/s     1.00x
pinned host memory                    1.327     12.65 GB/s     1.48x
```

**320 MB so với 0 MB.** Đó không phải chuyện sở thích phong cách, đó là khác biệt
giữa một dịch vụ chạy được cả tuần và một dịch vụ thì không.

## Điểm cốt lõi

- **Rò bộ nhớ thiết bị tệ hơn rò bộ nhớ host.** Không gì thu hồi chúng cho tới khi
  context chết.
- **RAII làm cho việc rò rỉ trở nên bất khả thi**, kể cả trên những đường đi bạn
  chưa nghĩ tới — return sớm, ngoại lệ, đoạn mã ai đó thêm vào sáu tháng sau.
- **Hãy xoá copy với những tài nguyên đắt đỏ.** Làm chi phí hiện rõ bằng `clone()`.
- **`noexcept` trên các phép move là thứ khiến một kiểu dùng được trong container.**
- **Đừng bao giờ `CUDA_CHECK` trong destructor.**
- **Bộ nhớ ghim nhanh hơn ~1.5 lần** và là tài nguyên khan hiếm của cả máy — lý do
  thứ hai khiến RAII quan trọng ở đây.

## Đi xa hơn

- Thêm `DeviceBuffer<T>::resize()` giữ nguyên nội dung. Khi nào thì nó đáng làm so
  với cấp phát một buffer mới?
- Thêm constructor nhận stream, dùng `cudaMallocAsync` (CUDA 11.2+). So sánh chi
  phí cấp phát — đây chính là memory pool của GĐ 1/03, được tích hợp sẵn trong driver.
- Thêm chế độ debug ghi lại mọi vùng cấp phát đang sống cùng vị trí gọi, và in ra
  những gì còn sót lại lúc thoát.
- So sánh với `thrust::device_vector` (GĐ 5/01). Nó làm khác gì, và vì sao?

---

**Đến đây là hết Giai đoạn 2.** Bạn đã viết được kernel CUDA đúng, quản lý bộ nhớ
thiết bị an toàn, và báo cáo được những con số hiệu năng trung thực.
[Giai đoạn 3](../../../phase-3-intermediate/README.vi.md) là nơi chúng trở nên nhanh.
