<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 07 - Nhân ma trận naive trên GPU

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [06](../06-matrix-add-2d/README.vi.md), [GĐ 1/01](../../../phase-1-foundation/exercises/01-matmul-naive-cpu/README.vi.md) | **Cần GPU**

## Mục tiêu

Viết mốc cơ sở GPU mà Giai đoạn 3 và Giai đoạn 5 sẽ đem ra so — rồi tìm hiểu vì
sao một kernel **coalesced hoàn hảo** vẫn chỉ đạt ~5% đỉnh FLOP/s của GPU. Câu trả
lời chính là lý do shared memory tồn tại.

## Kiến thức nền

Mỗi thread lo một phần tử đầu ra. Thread `(row, col)` duyệt trọn hàng `row` của
`A` và trọn cột `col` của `B`:

```cuda
float acc = 0.0f;
for (int k = 0; k < n; ++k)
    acc += A[row * n + k] * B[k * n + col];
C[row * n + col] = acc;
```

**Hãy kiểm tra mẫu truy cập trước đã.** Trong một warp, các thread khác nhau ở `col`:

| Truy cập | Trên 32 thread | Đánh giá |
|---|---|---|
| `A[row * n + k]` | **cùng** một địa chỉ cho cả 32 | broadcast — phần cứng xử lý tốt |
| `B[k * n + col]` | 32 địa chỉ **liên tiếp** | coalesced hoàn hảo |

Vậy là kernel này *vốn đã* truy cập bộ nhớ đúng cách. Mà nó vẫn chậm.

**Vì sao.** Mỗi phần tử đầu ra tốn `2N` FLOP và phát ra `2N` lệnh nạp global, nên

```
cường độ số học = 2N / (2N × 4 byte) = 0.25 FLOP trên mỗi byte được nạp
```

và con số đó đứng nguyên 0.25 dù `N` lớn tới đâu. So với điểm gãy gần 37, thế là
vô vọng.

Hãy phân biệt cho kỹ: **thuật toán không nghẽn bộ nhớ** — nó làm `N³` phép tính
trên `N²` dữ liệu, nên cường độ nội tại của nó tăng theo `N`. **Kernel này thì có**,
vì mỗi thread đọc lại hàng của `A` và cột của `B` từ bộ nhớ toàn cục thay vì chia
sẻ chúng với 255 thread khác trong cùng block vốn cần đúng những giá trị đó.

Với `n = 1024`, đó là **683 lần số lệnh nạp so với mức thuật toán đòi hỏi**.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`matmulNaive()`** — cộng dồn vào biến cục bộ `float acc`, đúng như trên CPU ở
   GĐ 1/01.
2. **`matmulNaiveSwapped()`** — đổi chỗ `row` và `col`, đo cả hai.
3. **Kiểm chứng** với `rtol = 1e-4` — cộng 1024 số hạng bằng FMA lệch đi đáng kể so
   với phép nhân rồi cộng riêng của CPU.
4. **Phân tích**: GFLOP/s đạt được theo phần trăm đỉnh, số byte yêu cầu so với số
   byte thực sự cần, và tỉ lệ giữa hai con số.

## Build và chạy

```bash
cmake --build build --target p2_07_matmul_naive -j
./build/bin/p2/p2_07_matmul_naive
```

## Kết quả mong đợi

RTX 3060, `n = 1024`:

```
--- Performance ---
Variant                           Time (ms)    GFLOP/s   Speedup
----------------------------------------------------------------
CPU ikj (single thread)             154.834      13.87     1.00x
GPU naive (coalesced B)               2.680     801.36    57.78x
GPU naive (strided B)                12.130     177.03    12.76x

--- Where the performance went ---
  Achieved              : 801.4 GFLOP/s  (6% of this GPU's FP32 peak)
  Bytes REQUESTED       : 8.6 GB per call
  Bytes actually needed : 0.013 GB
  Ratio                 : 683x more loads than the algorithm requires
```

**Nhanh hơn CPU 58 lần, mà vẫn chỉ bằng 6% khả năng của GPU.** Cả hai sự thật đều
quan trọng. Đổi chỗ hai trục làm mất thêm 4.5 lần nữa.

> *Tốc độ* nạp mà chương trình in ra hoá ra gấp mấy lần đỉnh DRAM, điều bất khả thi
> — và đầy thông tin. Nó có nghĩa là cache L1 và L2 đã phục vụ phần lớn số yêu cầu
> đó. Kernel không bị giới hạn bởi băng thông DRAM; nó bị giới hạn bởi tốc độ mà
> đường ống bộ nhớ phát ra và phục vụ được 8.6 GB **yêu cầu** nạp.

## Điểm cốt lõi

- **Coalescing là cần, không đủ.** Nó quyết định một yêu cầu được phục vụ hiệu quả
  ra sao; nó không giảm được số yêu cầu bạn phát ra.
- **"Nghẽn bộ nhớ" là thuộc tính của kernel, không phải của thuật toán.** Cùng một
  phép toán có thể rơi vào hoặc không, tuỳ cách bạn dàn xếp dữ liệu.
- **Truy cập broadcast là ổn.** Cả 32 thread đọc một địa chỉ tốn một giao dịch, chứ
  không phải 32.
- Nhanh hơn 58 lần một luồng CPU nghe ấn tượng cho tới khi bạn so với trần của
  *chính GPU*. Luôn nêu cả hai.

## Đi xa hơn

- Thử `dim3 block(32, 32)` (1024 thread). Nhanh hơn hay chậm hơn? Kiểm tra occupancy
  bằng `ncu --set full`.
- Tính tỉ lệ đó cho `n = 2048`. Con số 683 lần có tăng không?
- Profile với `ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` và xác
  nhận số yêu cầu khớp với mô hình.
- **Tiếp theo:** [GĐ 3/04 tiled-matmul](../../../phase-3-intermediate/exercises/04-tiled-matmul/)
  dàn các tile vào bộ nhớ `__shared__` để mỗi giá trị nạp về phục vụ 16 thread.
  Rồi [GĐ 5/03](../../../phase-5-expert/exercises/03-cublas-gemm/) đem cả hai so với cuBLAS.
