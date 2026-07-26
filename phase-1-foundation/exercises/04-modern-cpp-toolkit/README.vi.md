<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 04 - Bộ công cụ C++ hiện đại: RAII, move, template, lambda

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [03](../03-memory-pool-allocator/)

## Mục tiêu

Xây một lớp nhỏ, `Matrix<T>`, sử dụng đủ bốn tính năng C++ mà bạn sẽ dựa vào suốt
phần còn lại của lộ trình. Ở Giai đoạn 2 bài 12 bạn viết lại chính lớp này nhưng
bọc quanh `cudaMalloc`/`cudaFree`, nên làm đúng ở đây là lời gấp đôi.

## Kiến thức nền

**RAII** — tài nguyên thuộc sở hữu của một đối tượng, và destructor giải phóng nó.
Không tồn tại đường đi nào của chương trình, kể cả ngoại lệ hay `return` sớm, có
thể gây rò rỉ. Điều này quan trọng với GPU hơn nhiều so với CPU: bộ nhớ host bị rò
sẽ được thu hồi khi tiến trình kết thúc, nhưng một `cudaFree` bị quên trong một
dịch vụ chạy dài sẽ rò bộ nhớ thiết bị cho tới khi context bị huỷ hoàn toàn.

**Move semantics** — phép move lấy con trỏ thay vì nhân bản dữ liệu. Với một
`Matrix`, đó là ba phép gán thay vì sao chép hàng megabyte. Hai quy tắc:

- Đối tượng sau khi bị move phải vẫn **hợp lệ và huỷ được** — đặt kích thước về 0
  và con trỏ về null.
- Đánh dấu các thao tác move là **`noexcept`**. `std::vector` chỉ move các phần tử
  khi cấp phát lại nếu move constructor là `noexcept`; nếu không nó sẽ âm thầm
  sao chép để giữ đảm bảo ngoại lệ mạnh.

**Template** — một bản cài đặt, nhiều kiểu dữ liệu. Trên GPU bạn sẽ viết một lớp
bọc kernel rồi khởi tạo nó cho `float`, `double` và `__half`.

**Lambda** — `apply()` nhận template theo kiểu callable chứ không nhận
`std::function`. Nhờ vậy thân lambda được *inline thẳng vào vòng lặp*: không có
lời gọi ảo, không cấp phát, mã máy y hệt một vòng lặp viết tay. Thrust và CUB dựa
đúng vào cơ chế này để functor trên device không tốn chi phí.

## Nhiệm vụ của bạn

Mở `main.cpp` và cài đặt `Matrix<T>` (TODO 1–7):

1. Constructor — `std::make_unique<T[]>(rows * cols)`, báo cho `AllocationTracker`.
2. Destructor — báo cho tracker (`unique_ptr` lo phần giải phóng bộ nhớ).
3. Copy constructor — sao chép **sâu**.
4. Move constructor — lấy con trỏ rồi làm rỗng nguồn. Nhớ `noexcept`.
5. Copy assignment và move assignment. Xử lý tự gán; copy-and-swap là cách gọn nhất.
6. `operator()(r, c)` — chỉ số row-major.
7. `apply(fn)` — nhận template theo kiểu callable.

Sau đó (TODO 8) viết các kiểm tra chứng minh: truy cập phần tử đúng chiều, phép
move **không** cấp phát, ma trận sau khi bị move là rỗng, và không rò rỉ gì.

## Build và chạy

```bash
cmake --build build --target p1_04_modern_cpp_toolkit -j
./build/bin/p1/p1_04_modern_cpp_toolkit
```

## Kết quả mong đợi

```
--- RAII and element access ---
  [PASS] constructor allocated once
  [PASS] element access round-trips
  [PASS] nothing leaked after scope exit

--- Copy versus move ---
  [PASS] copy allocates new storage
  [PASS] move allocates nothing
  [PASS] moved-from matrix is empty

--- Zero-cost abstraction ---
Variant                           Time (ms)      GB/s ...
raw std::vector loop                   ~6.5      ~5.1
Matrix<T>::apply(lambda)               ~6.5      ~5.1
```

Việc hai dòng cuối bằng nhau *chính là* kết quả cần thấy: lớp trừu tượng không
tốn chi phí nào.

## Điểm cốt lõi

- **RAII nghĩa là không thể rò rỉ**, chứ không phải "không rò nếu bạn nhớ".
- Move là **O(1)**; copy là **O(n)**. Trả về theo giá trị trở nên rẻ ngay khi có
  move constructor.
- **`noexcept` trên các phép move có tác dụng thật**, không phải chú thích cho vui.
- Template + lambda được giải quyết lúc biên dịch — **zero-cost abstraction** là
  nghĩa đen, và bạn có thể chứng minh bằng benchmark.

## Đi xa hơn

- Thêm `Matrix<T>::create(rows, cols)` trả về `std::optional<Matrix<T>>` để có
  đường xử lý lỗi không ném ngoại lệ.
- Xoá hẳn copy constructor rồi xem những chỗ gọi nào không biên dịch được nữa —
  đó chính xác là điều một device buffer nên làm.
- Biên dịch với `-fno-elide-constructors` rồi chạy lại. Bây giờ có bao nhiêu lần
  cấp phát?
- Đọc *Effective Modern C++* (Scott Meyers), mục 23–30 về move semantics và
  perfect forwarding.
