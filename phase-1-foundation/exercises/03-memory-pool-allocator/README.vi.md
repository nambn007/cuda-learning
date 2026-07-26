<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 03 - Bộ cấp phát memory pool

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: con trỏ, `new`/`delete`

## Mục tiêu

Viết hai bộ cấp phát mà mọi hệ thống hiệu năng cao đều cần đến — **bump allocator
(arena)** và **pool allocator** — rồi đo xem chúng nhanh hơn `new`/`delete` bao
nhiêu. Đây là buổi tổng duyệt trên CPU cho một vấn đề còn tệ hơn nhiều trên GPU.

## Kiến thức nền

**Vì sao một khoá học CUDA lại quan tâm đến allocator?** Bởi vì `cudaMalloc` tốn
khoảng **100–500 micro-giây**. Nó đi vào driver, có thể đồng bộ toàn bộ thiết bị,
và có thể sắp xếp lại không gian địa chỉ ảo của GPU. Một kernel chạy 50 µs mà
phải chờ 200 µs cấp phát thì dành 80% thời gian không để tính toán.

Mọi codebase CUDA nghiêm túc đều giải quyết chuyện này theo cùng một cách: cấp
phát một khối lớn từ đầu rồi tự chia nhỏ ra dùng. CUDA 11.2+ đóng gói sẵn dưới
dạng `cudaMallocAsync` với `cudaMemPool_t` phía sau; PyTorch có caching allocator
riêng; TensorRT có workspace riêng. Tất cả đều là hai cấu trúc dữ liệu dưới đây.

**Bump allocator (arena).** Một buffer cộng một offset. `allocate()` căn chỉnh
offset, trả về con trỏ, đẩy offset lên — ba lệnh, không khoá, không tìm kiếm,
không phân mảnh. Cái giá: bạn không thể giải phóng riêng một đối tượng, chỉ có
thể xoá sạch bằng `reset()`. Điều đó chấp nhận được mỗi khi các đối tượng có cùng
vòng đời — đúng với công việc theo từng frame, từng request, từng batch.

**Pool allocator.** Các khối kích thước cố định kèm một free list, nên giải phóng
riêng lẻ được. Chỗ hay nhất: một khối *đang rảnh* không chứa dữ liệu người dùng,
nên vài byte đầu của nó bỏ trống — hãy cất luôn con trỏ "khối rảnh kế tiếp" vào
đó. Free list tốn thêm 0 byte bộ nhớ. `allocate()` lấy phần tử đầu danh sách,
`deallocate()` đẩy trả lại đầu danh sách; cả hai đều O(1).

**Căn chỉnh (alignment).** `alignUp(n, a) = (n + a - 1) & ~(a - 1)` đúng vì
alignment luôn là luỹ thừa của 2. Truy cập lệch căn chỉnh thì chậm trên x86 và là
lỗi cứng trên một số kiến trúc — còn trên GPU, nó âm thầm phá vỡ coalescing.

## Nhiệm vụ của bạn

Mở `main.cpp`:

1. **`BumpAllocator`** — hàm khởi tạo, `allocate(size, alignment)`, `reset()`.
   Trả về `nullptr` khi arena hết chỗ; tuyệt đối không tràn buffer.
2. **`PoolAllocator`** — nối toàn bộ các khối thành free list trong hàm khởi tạo,
   rồi cài `allocate()` và `deallocate()`. Dùng `std::memcpy` để đọc/ghi con trỏ
   nhúng bên trong (cách này hợp lệ ngay cả với vùng nhớ thô).
3. **Đo cả ba** bằng `timeCpuMs()` rồi điền vào `ResultTable`.

Dùng **placement new** (`new (raw) Node{...}`) để khởi tạo đối tượng trên vùng
nhớ mà bạn tự cấp phát.

## Build và chạy

```bash
cmake --build build --target p1_03_memory_pool_allocator -j
./build/bin/p1/p1_03_memory_pool_allocator
```

## Kết quả mong đợi

```
--- Correctness ---
  [PASS] new/delete checksum
  [PASS] bump allocator checksum
  [PASS] pool allocator checksum
  [PASS] pool reuses freed blocks
  [PASS] arena reports exhaustion

--- Performance ---
Variant                           Time (ms)  ...   Speedup
new / delete                          ~14.0            1.00x
bump allocator                         ~1.6            ~9x
pool allocator                         ~2.6            ~5x
```

## Điểm cốt lõi

- **Cấp phát không miễn phí.** `new` phải duyệt free list, có thể phải lấy khoá và
  có thể gọi xuống kernel hệ điều hành; trên GPU thì tệ hơn cỡ 1000 lần.
- **Thu hẹp bài toán để làm nó nhanh.** Arena nhanh *chính vì* nó từ bỏ khả năng
  giải phóng riêng lẻ.
- **Cấu trúc dữ liệu nhúng (intrusive) tốn 0 chi phí.** Cất liên kết free list
  ngay trong khối rảnh chính là toàn bộ mẹo hay.
- Vùng nhớ thô + **placement new** là cách tách bạch cấp phát khỏi khởi tạo — đúng
  kiểu tách bạch giữa `cudaMalloc` và một kernel khởi tạo.

## Đi xa hơn

- Thêm `Arena::create<T>(args...)` an toàn với `reset()`, có ghi nhận destructor.
- Làm `PoolAllocator` an toàn đa luồng bằng `compare_exchange` lock-free trên đầu
  danh sách — rồi so sánh với hàng đợi lock-free trên GPU ở Giai đoạn 4 bài 10.
- Đọc tài liệu CUDA về
  [stream-ordered memory allocation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator).
