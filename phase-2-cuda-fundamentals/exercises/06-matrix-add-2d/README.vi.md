<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 06 - Cộng ma trận 2D, và thứ tự trục

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐ | Thời gian: ~1 giờ | Yêu cầu: [02](../02-thread-indexing/README.vi.md), [04](../04-saxpy/README.vi.md) | **Cần GPU**

## Mục tiêu

Viết cùng một kernel theo ba cách — cả ba đều đúng, đều cho ra những con số y hệt
nhau — rồi phát hiện rằng **đổi chỗ hai dòng lệnh làm mất 70% hiệu năng**. Đây là
lần đầu bạn chạm vào hiện tượng mà Giai đoạn 3 dành hẳn năm bài để mổ xẻ.

## Kiến thức nền

Cộng hai ma trận về mặt số học giống hệt cộng hai vector: bố cục row-major vốn là
một mảng liên tục. Vậy tại sao lại launch grid 2D làm gì?

**Chủ yếu là để tiện.** Stencil, tích chập hay chuyển vị đều cần `(row, col)` của
từng phần tử, và tính chúng từ chỉ số phẳng thì tốn một phép chia. Hãy dùng grid
2D khi *kernel* cần toạ độ 2D — chứ không phải vì dữ liệu được vẽ ra hình chữ nhật.

**Thứ không hề mang tính hình thức là bạn ánh xạ trục nào vào `x`.** Thread được
tuyến tính hoá với `x` biến thiên nhanh nhất, nên một warp 32 thread bao trùm
`threadIdx.x = 0..15` với hai giá trị của `threadIdx.y` (khi block là 16×16):

| Ánh xạ | Một warp chạm vào gì | Kết quả |
|---|---|---|
| `x` = cột | 16 cột liên tiếp × 2 hàng kề nhau — hai dải 64 byte liền mạch | vài sector |
| `x` = hàng | 16 địa chỉ cách nhau `cols` float | **16 sector 32 byte riêng biệt**, mỗi cái chỉ dùng 8 byte |

Bản đổi chỗ yêu cầu khoảng **4 lần lưu lượng bộ nhớ** cho đúng cùng một kết quả.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`matrixAddRowMajor()`** — `x` = cột. Kiểm tra cả hai biên.
2. **`matrixAddColumnMajor()`** — `x` = hàng, mọi thứ khác giữ nguyên.
   ⚠️ Kernel đổi chỗ cần **grid** cũng phải đổi chỗ theo, nếu không nó sẽ không
   phủ hết ma trận.
3. **`matrixAddFlat()`** — grid-stride loop 1D phẳng trên `rows * cols`.
4. **Xác minh cả ba khớp nhau**, rồi đo thời gian.

Trước khi chạy: **hãy đoán xem bản đổi chỗ chậm hơn bao nhiêu.** Viết con số đó ra,
rồi mới đo. Khoảng cách giữa dự đoán và kết quả chính là bài học.

## Build và chạy

```bash
cmake --build build --target p2_06_matrix_add_2d -j
./build/bin/p2/p2_06_matrix_add_2d
```

## Kết quả mong đợi

RTX 3060, 4096×4096:

```
--- Correctness ---
  [PASS] row-major mapping (x = column)   (max abs err 0.000e+00)
  [PASS] column-major mapping (x = row)   (max abs err 0.000e+00)
  [PASS] flat 1D grid-stride              (max abs err 0.000e+00)

--- Performance ---
Variant                           Time (ms)      GB/s    Speedup
----------------------------------------------------------------
CPU (single thread)                  12.288     16.38      1.00x
GPU 2D, x = column                    0.605    332.81     20.31x
GPU 2D, x = row (swapped)             1.001    201.06     12.27x
GPU 1D flat grid-stride               0.629    320.21     19.54x

  x = column :   332.8 GB/s  (92% of peak)
  x = row    :   201.1 GB/s  (56% of peak)
  Swapping two lines cost 1.7x.
```

### Vì sao là 1.7× chứ không phải 4×?

**Cache L2.** GPU này có 2304 KB L2, và các block kề nhau đọc lại đúng những
sector mà một block trước đó vừa kéo về, nên cache hấp thụ phần lớn phần lãng phí.
Đổi hình dạng block thành 256×1 thì cache hết đỡ được — Giai đoạn 3 bài 01 làm
đúng như vậy và đo được toàn bộ mức ảnh hưởng.

Vì thế bài học trung thực có hai mặt: hình phạt là có thật và lớn, *đồng thời*
phần cứng cứu bạn một phần. Đừng mặc định điều nào cả.

## Điểm cốt lõi

- **`x` phải ánh xạ vào trục liên tục trong bộ nhớ.** Làm ngược lại thì vẫn đúng
  nhưng chậm, và không có gì cảnh báo bạn.
- **Launch 2D là để tiện, không phải bắt buộc.** Bản 1D phẳng ở đây chỉ kém 4%.
- **L2 che giấu một phần mẫu truy cập tệ**, và đó chính là lý do những mẫu tệ tồn
  tại được trong mã thật — chúng chậm, nhưng chưa thảm hoạ, cho tới khi dữ liệu
  lớn lên.
- Hãy dự đoán rồi mới đo. Một dự đoán sai là kết quả hữu ích nhất bạn có thể thu được.

## Đi xa hơn

- Chạy lại với `dim3 block(256, 1)`. Khoảng cách sẽ giãn ra mạnh — vì sao?
- Dùng `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` trên cả hai
  kernel rồi so trực tiếp số sector.
- Thử ma trận không vuông (4096×1024). Mức phạt có đổi không?
- Đọc [Best Practices Guide về truy cập coalesced](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory).
