<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 02 - Đánh chỉ số thread trong 1D, 2D và 3D

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐ | Thời gian: ~1 giờ | Yêu cầu: [01](../01-hello-cuda/) | **Cần GPU**

## Mục tiêu

Ánh xạ dữ liệu có hình dạng bất kỳ lên các thread, và học **grid-stride loop** —
mẫu launch được dùng trong gần như mọi kernel CUDA thực tế.

## Kiến thức nền

Một lệnh launch tạo ra lưới block 1D/2D/3D, mỗi block chứa khối thread 1D/2D/3D.
Chọn hình dạng đó là quyết định thiết kế đầu tiên của mọi kernel, và tính sai số
học chỉ số là lỗi phổ biến nhất của người mới.

**Quy tắc:** khớp hình dạng grid với hình dạng *dữ liệu*, và đảm bảo các **thread**
liền kề chạm tới các **địa chỉ** liền kề.

```cuda
// 1D
int gid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int i = y * width + x;                    // x biến thiên nhanh nhất

// 3D
int z = blockIdx.z * blockDim.z + threadIdx.z;
int i = (z * height + y) * width + x;
```

**Vì sao `x` phải là trục liên tục trong bộ nhớ.** Các thread trong một warp khác
nhau ở `threadIdx.x` trước tiên. Nếu `x` là chỉ số cột, một warp đọc 32 địa chỉ
liền kề và phần cứng gộp chúng thành vài giao dịch. Đổi chỗ `x` và `y` thì kernel
vẫn cho kết quả đúng — nhưng chậm khoảng **10 lần**. Giai đoạn 3 bài 01 đo chính
xác điều này.

**Grid-stride loop** tách rời kích thước grid khỏi kích thước dữ liệu:

```cuda
int gid    = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = gid; i < n; i += stride) out[i] = i;
```

Bốn lợi ích: một cấu hình launch dùng được cho **mọi** `n`; grid có thể chọn theo
*GPU* (vài block mỗi SM) thay vì theo dữ liệu; truy cập vẫn coalesced vì các thread
liền kề vẫn nhận địa chỉ liền kề ở mỗi vòng lặp; và kiểm tra biên có sẵn trong
điều kiện vòng lặp.

Đôi khi đây còn là lựa chọn *duy nhất* — `gridDim.y`/`z` bị giới hạn ở 65535 và
`blockDim.z` ở 64.

## Nhiệm vụ của bạn

Mở `main.cu` và cài đặt `index1D`, `index2D`, `index3D` và `gridStride`, rồi viết
các lệnh launch. Hình dạng block gợi ý: `256` (1D), `dim3(16,16)` (2D),
`dim3(8,8,4)` (3D) — đều 256 thread, đều là bội của 32.

## Build và chạy

```bash
cmake --build build --target p2_02_thread_indexing -j
./build/bin/p2/p2_02_thread_indexing
```

## Kết quả mong đợi

```
--- 2D - one thread per pixel ---
  1920x1080 image -> grid(120, 68) x block(16, 16) = 2088960 threads
  covers 1920x1088 pixels, 15360 of them idle
  [PASS] 2D indexing (n=2073600, exact match)

--- Grid-stride loop ---
  n = 100000, but launching only <<<112, 256>>> = 28672 threads
  each thread handles about 3.5 elements
  [PASS] grid-stride loop (n=100000, exact match)
  [PASS] same launch config, n = 7 (n=7, exact match)
```

Dòng cuối cùng chính là điểm mấu chốt: cùng một lệnh launch xử lý được cả
`n = 100000` lẫn `n = 7`.

## Điểm cốt lõi

- Số học chỉ số mở rộng theo cùng một cách ở mọi chiều; chỉ có công thức làm phẳng
  là thay đổi.
- **Kiểm tra mọi biên.** Grid 2D phủ ảnh 1920×1080 thực ra phủ 1920×1088.
- **`x` là trục liên tục.** Làm ngược lại tốn ~10 lần thời gian mà vẫn cho kết quả
  đúng, nên không có gì cảnh báo bạn.
- **Hãy ưu tiên grid-stride loop.** Nó độc lập kích thước, chọn theo GPU, vẫn
  coalesced, và tự kiểm tra biên.

## Đi xa hơn

- Đổi chỗ `x` và `y` trong `index2D` rồi đo cả hai. (Giai đoạn 3/01 làm việc này
  một cách bài bản.)
- Dùng `cudaOccupancyMaxPotentialBlockSize()` để chọn grid thay cho `số SM × 4`.
  Nó có đồng ý với bạn không?
- Thử `dim3 block(32, 32)` = 1024 thread. Có launch được không? Có nhanh hơn không?
- Đọc [Programming Guide, "Thread Hierarchy"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)
  và bài blog NVIDIA [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
