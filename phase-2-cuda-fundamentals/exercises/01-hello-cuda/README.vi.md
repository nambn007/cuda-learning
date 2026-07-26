<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 01 - Hello CUDA

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐ | Thời gian: ~1 giờ | Yêu cầu: [Giai đoạn 1/08](../../../phase-1-foundation/exercises/08-gpu-device-query/) | **Cần GPU**

## Mục tiêu

Viết, chạy và kiểm chứng kernel đầu tiên của bạn. Ba điều xảy ra ở đây mà C++
thông thường không bao giờ có: một hàm chạy trên bộ xử lý khác, nó chạy hàng nghìn
lần cùng lúc, và **lời gọi trả về trước khi công việc hoàn thành**.

## Kiến thức nền

**`__global__`** đánh dấu một hàm được biên dịch cho GPU, gọi từ CPU, và bắt buộc
trả về `void`. Mọi thread do lệnh launch tạo ra đều chạy cùng một thân hàm — đó
chính là toàn bộ mô hình SIMT.

**Cú pháp launch** `kernel<<<grid, block>>>(args)` là mảnh cú pháp duy nhất không
thuộc C++ trong CUDA. Nó tạo ra `grid` block, mỗi block `block` thread.

**Bốn biến dựng sẵn** nhìn thấy được bên trong mọi kernel:

| | ý nghĩa | miền giá trị |
|---|---|---|
| `threadIdx.x` | chỉ số thread trong block của nó | `[0, blockDim.x)` |
| `blockIdx.x` | chỉ số block trong grid | `[0, gridDim.x)` |
| `blockDim.x` | số thread mỗi block | |
| `gridDim.x` | số block trong grid | |

Từ đó suy ra **dòng lệnh quan trọng nhất của CUDA**:

```cuda
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

**Kiểm tra biên là bắt buộc.** Với `n = 1000` và 256 thread mỗi block, bạn tạo
1024 thread; 24 thread trong đó có `gid >= n`. Ghi vượt vùng cấp phát thường
*không* làm GPU crash — nó âm thầm phá hỏng bất cứ dữ liệu nào nằm kế bên. Luôn
luôn `if (gid < n)`.

**Launch là bất đồng bộ**, nên lỗi có hai loại và cần hai cách kiểm tra khác nhau:

- `cudaGetLastError()` — lỗi của bản thân *lệnh launch*: quá nhiều thread mỗi
  block, quá nhiều shared memory. Báo ngay lập tức.
- `cudaDeviceSynchronize()` — lỗi trong lúc *thực thi*: địa chỉ không hợp lệ,
  assertion. Chỉ thấy được sau khi đã chờ.

**Thứ tự chạy các block là không xác định.** Block được lập lịch lên các SM theo
thứ tự tài nguyên được giải phóng. Các dòng `printf` sẽ ra lộn xộn, và mọi đoạn mã
phụ thuộc vào thứ tự block đều là mã sai.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`helloCUDA()`** — `printf` chỉ số block và thread.
2. **`writeGlobalId(out, n)`** — tính chỉ số toàn cục, kiểm tra biên, ghi giá trị.
3. **Phần cấp phát bộ nhớ** — `cudaMalloc` → launch → `cudaMemcpy` về host →
   `cudaFree`, rồi kiểm tra `host[i] == i`.

## Build và chạy

```bash
cmake --build build --target p2_01_hello_cuda -j
./build/bin/p2/p2_01_hello_cuda
```

Hoặc không dùng CMake:

```bash
nvcc -arch=native -I common -o hello \
     phase-2-cuda-fundamentals/exercises/01-hello-cuda/solution.cu && ./hello
```

## Kết quả mong đợi

```
--- Kernel printf ---
  Launching <<<2, 4>>> = 2 blocks x 4 threads = 8 threads

  Hello from block 1, thread 0 (global id 4)
  Hello from block 1, thread 1 (global id 5)
  ...
  Hello from block 0, thread 0 (global id 0)

--- Verifiable kernel ---
  n = 1000, block = 256, grid = 4 (1024 threads, 24 idle)
  [PASS] every thread wrote its global id (n=1000, exact match)
```

## Điểm cốt lõi

- `gid = blockIdx.x * blockDim.x + threadIdx.x` — hãy thuộc lòng.
- **Luôn kiểm tra biên.** Ghi ngoài phạm vi trên GPU phá dữ liệu một cách âm thầm.
- Launch là **bất đồng bộ**; phải kiểm tra cả lệnh launch *lẫn* quá trình thực thi.
- **Thứ tự thực thi block là không xác định.** Đừng bao giờ dựa vào nó.
- Hãy để `blockDim` là **bội của 32** — phần cứng lập lịch theo warp 32 thread, và
  một block 100 thread lãng phí 28 chỗ trong warp cuối cùng của nó.
- `printf` chứng minh kernel đã *chạy*; chỉ có kết quả được kiểm chứng mới chứng
  minh nó *đúng*.

## Đi xa hơn

- Thử launch `<<<1, 2048>>>`. Nó thất bại — `cudaGetLastError()` sẽ nói vì sao.
  (Giới hạn là 1024 thread mỗi block.)
- Bỏ phần kiểm tra biên rồi chạy `compute-sanitizer ./hello`. Nó chỉ ra chính xác
  chỗ truy cập vượt phạm vi.
- Thêm `printf("%d\n", warpSize)` bên trong kernel. Giá trị đó từ đâu ra?
- Đọc [CUDA C++ Programming Guide, "Programming Model"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model).
