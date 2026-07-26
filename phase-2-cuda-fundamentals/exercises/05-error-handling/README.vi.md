<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 05 - Xử lý lỗi: phục hồi được, lỗi dính, và bất đồng bộ

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1 giờ | Yêu cầu: [03](../03-vector-add/README.vi.md) | **Cần GPU**

## Mục tiêu

Khác với mọi bài trong kho mã này, mục tiêu ở đây là **cố tình làm CUDA lỗi** và
quan sát từng kiểu lỗi hành xử ra sao. Lỗi GPU khiến người ta khổ hơn lỗi CPU chủ
yếu vì nó được báo ở rất xa đoạn mã gây ra nó — bài này chỉ ra chính xác vì sao.

## Kiến thức nền

### Hai loại lỗi

**Phục hồi được** — được phát hiện *trước khi* có gì chạy: cấu hình launch không
hợp lệ, cấp phát thất bại, tham số sai. `cudaGetLastError()` trả về lỗi, xoá cờ,
và context tiếp tục hoạt động bình thường.

**Lỗi dính (sticky)** — phát sinh *trong lúc* thực thi: địa chỉ không hợp lệ,
assert phía thiết bị. Những lỗi này **phá huỷ context CUDA**. Mọi lời gọi sau đó —
`cudaMalloc`, `cudaMemcpy`, mọi lệnh launch — đều trả về cùng lỗi ấy mãi mãi. Cách
duy nhất để phục hồi là thoát tiến trình.

### Cái bẫy khiến người ta mất cả buổi chiều

Một lời gọi API thất bại **vừa trả về lỗi vừa bật cờ last-error**. Đọc giá trị trả
về *không* xoá cờ:

```cuda
cudaError_t e = cudaMalloc(&p, huge);   // trả về cudaErrorMemoryAllocation
cudaGetLastError();                     // CŨNG trả về nó - cờ đã được bật
cudaGetLastError();                     // tới lúc này mới sạch
```

Nên nếu bạn bỏ qua một lời gọi thất bại, thì `CUDA_CHECK_LAST()` kế tiếp sau một
lệnh launch hoàn toàn không liên quan sẽ báo *chính lỗi đó* — và bạn đi lùng sục
một kernel chưa bao giờ hỏng. **Hãy kiểm tra giá trị trả về của mọi lời gọi, ngay
tại lời gọi.**

### `peek` so với `get`

| Lời gọi | Trả về lỗi | Xoá cờ |
|---|---|---|
| `cudaPeekAtLastError()` | có | **không** |
| `cudaGetLastError()` | có | **có** |

### Tính bất đồng bộ

Lệnh launch trả về trước khi kernel chạy, nên lỗi bên trong kernel chỉ lộ ra ở
điểm đồng bộ kế tiếp — thường là một `cudaMemcpy` vô tội cách đó hàng trăm dòng.
Khi một lỗi trông vô lý:

```bash
CUDA_LAUNCH_BLOCKING=1 ./chương_trình_của_bạn
```

Runtime khi đó sẽ chờ sau mỗi lệnh launch và báo lỗi ngay tại dòng gây ra nó. Nó
chậm; chỉ dùng để chẩn đoán.

### Tràn nhỏ mới là loại nguy hiểm

Ghi vượt vài kilobyte thường **không** bị bắt — runtime cấp phát theo trang lớn,
nên phép ghi rơi vào vùng nhớ tiến trình vốn đã sở hữu và âm thầm phá hỏng nó. Chỉ
có `compute-sanitizer` tìm ra:

```bash
compute-sanitizer ./chương_trình                    # truy cập sai
compute-sanitizer --tool racecheck  ./chương_trình  # race dữ liệu
compute-sanitizer --tool synccheck  ./chương_trình  # __syncthreads sai
compute-sanitizer --tool initcheck  ./chương_trình  # đọc bộ nhớ chưa khởi tạo
```

## Nhiệm vụ của bạn

Mở `main.cu` và làm ba thí nghiệm:

1. **Phục hồi được** — launch với hơn 1024 thread mỗi block, tự đọc lỗi (đừng dùng
   `CUDA_CHECK_LAST()` vì nó gọi `exit`), xác nhận đó là
   `cudaErrorInvalidConfiguration`, rồi chứng minh một lệnh launch hợp lệ vẫn chạy.
2. **Peek vs get** — gây ra một lỗi, rồi gọi mỗi hàm hai lần.
3. **Lỗi dính** — ghi vào vùng nằm xa ngoài phạm vi cấp phát. Kiểm tra lỗi ngay sau
   lệnh launch (đã có lỗi chưa? vì sao chưa?), rồi sau
   `cudaDeviceSynchronize()`, rồi thử một lời gọi bất kỳ không liên quan.
   **Đặt phần này cuối cùng** — sau đó context đã chết.

Sau đó chạy file nhị phân dưới `compute-sanitizer`.

## Build và chạy

```bash
cmake --build build --target p2_05_error_handling -j
./build/bin/p2/p2_05_error_handling
compute-sanitizer ./build/bin/p2/p2_05_error_handling
```

## Kết quả mong đợi

```
--- Recoverable errors ---
  <<<4, 256>>> (valid)                       ok
  <<<1, 2048>>> (too many threads per block) cudaErrorInvalidConfiguration
  <<<4, 256>>> (valid again)                 ok
  cudaMalloc(1 PB)                           cudaErrorMemoryAllocation
  cudaGetLastError() after that              cudaErrorMemoryAllocation
  cudaGetLastError() again                   cudaSuccess

--- Sticky errors (this destroys the context) ---
  launch error       : cudaSuccess          <- lệnh launch không sao cả
  after synchronise  : cudaErrorIllegalAddress
  cudaMalloc(16)     : cudaErrorIllegalAddress    <- context đã chết
```

Chú ý bản thân lệnh launch báo **thành công**. Thiệt hại xảy ra sau đó.

## Điểm cốt lõi

- **Lỗi phục hồi được để context nguyên vẹn; lỗi dính thì không.** Chỉ có thoát
  tiến trình mới phục hồi được khỏi lỗi dính.
- **Lời gọi thất bại vừa bật cờ vừa trả về lỗi.** Một thất bại không kiểm tra sẽ bị
  đổ lỗi cho kernel bạn chạy kế tiếp.
- **`peek` để xem, `get` để lấy và xoá.**
- **Launch là bất đồng bộ**, nên vị trí được báo không phải vị trí thật.
  `CUDA_LAUNCH_BLOCKING=1` khắc phục điều đó trong lúc gỡ lỗi.
- **Tràn nhỏ phá dữ liệu âm thầm.** Hãy chạy `compute-sanitizer` trước khi tin
  rằng một kernel đã đúng.

## Năm quy tắc

1. Bọc mọi lời gọi runtime bằng `CUDA_CHECK(...)`.
2. Sau mỗi lệnh launch, gọi `CUDA_CHECK_LAST()` để bắt lỗi cấu hình.
3. Khi gỡ lỗi, dùng `CUDA_CHECK_KERNEL()` để bắt cả lỗi thực thi ngay tại chỗ
   launch. Nhớ bỏ nó khỏi vòng lặp đo thời gian — nó tốn một lần đồng bộ toàn thiết bị.
4. Khi một lỗi xuất hiện ở chỗ không thể nào có lỗi, hãy nghi ngờ một lỗi trước đó
   chưa được kiểm tra và chạy lại với `CUDA_LAUNCH_BLOCKING=1`.
5. Chạy `compute-sanitizer` trước khi tin bất kỳ kernel nào.

## Đi xa hơn

- Thêm `assert(idx < n)` bên trong kernel rồi kích hoạt nó. Bạn nhận được lỗi gì?
- So sánh `cudaGetErrorName()` với `cudaGetErrorString()`. Khi nào nên dùng cái nào?
- Đọc [tài liệu xử lý lỗi của CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html).
