<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 03 - Xung đột bank trong shared memory

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [02](../02-shared-memory-basics/README.vi.md) | **Cần GPU**

## Mục tiêu

Shared memory có bộ quy tắc truy cập riêng, hoàn toàn tách biệt với coalescing của
bộ nhớ toàn cục. Vi phạm chúng thì một chỉ số đặt sai chỗ tốn **15 lần**. Sửa chúng
chỉ bằng một cột dư.

## Kiến thức nền

Shared memory không phải một khối nhớ phẳng. Nó được chia thành **32 bank**, mỗi
bank rộng 4 byte, các word liên tiếp nằm ở các bank liên tiếp:

```
bank = (địa chỉ / 4) % 32

word:  0   1   2  ...  31  32  33 ...
bank:  0   1   2  ...  31   0   1 ...
```

Mỗi bank phục vụ **một địa chỉ mỗi chu kỳ**:

| Truy cập của một warp | Chi phí |
|---|---|
| 32 thread → 32 bank khác nhau | 1 chu kỳ ✅ |
| 32 thread → **cùng một địa chỉ** | 1 chu kỳ ✅ (broadcast) |
| *k* thread → các địa chỉ khác nhau trong một bank | *k* chu kỳ ❌ |

Với stride `s`, một warp chạm vào `32/gcd(s,32)` bank khác nhau, nên truy cập bị
tuần tự hoá thành `gcd(s,32)` chu kỳ. **Thứ quan trọng là stride modulo 32, không
phải độ lớn của nó** — stride 33 không hề xung đột trong khi stride 32 là trường
hợp tệ nhất.

### Cái bẫy kinh điển, và cách sửa kinh điển

```cuda
__shared__ float tile[32][32];
tile[threadIdx.x][i]     // địa chỉ tid*32 + i  →  bank i % 32
                         // MỌI thread cùng một bank: xung đột 32 chiều
```

```cuda
__shared__ float tile[32][33];      // thêm một cột
tile[threadIdx.x][i]     // địa chỉ tid*33 + i  →  bank (tid + i) % 32
                         // đủ 32 bank: không xung đột
```

Phí 128 byte mỗi tile. Bạn sẽ dùng lại mẹo này ở
[04 tiled-matmul](../04-tiled-matmul/) và
[05 transpose-optimized](../05-transpose-optimized/).

## Nhiệm vụ của bạn

Mở `main.cu` và viết năm kernel: `sharedStride`, `tileRowAccess`,
`tileColumnAccess`, `tileColumnPadded`, `sharedBroadcast`.

**Hãy tính số bank ra giấy trước khi viết từng kernel**, và dự đoán: điều gì xảy ra
ở stride 33? Broadcast có phải xung đột 32 chiều không?

## Build và chạy

```bash
cmake --build build --target p3_03_bank_conflicts -j
./build/bin/p3/p3_03_bank_conflicts
```

## Kết quả mong đợi

RTX 3060, một warp mỗi block:

```
--- Experiment 1 - stride ---
  stride      time (ms) predicted ways  vs stride 1  banks hit
  --------------------------------------------------------------------
  1               0.181              1        1.00x         32
  2               0.182              2        1.00x         16
  4               0.348              4        1.92x          8
  8               0.700              8        3.86x          4
  16              1.370             16        7.56x          2
  32              2.739             32       15.11x          1
  33              0.181              1        1.00x         32

--- Experiment 2 - a 2D tile ---
tile[32][32], by row                  0.179     1.00x
tile[32][32], by column               2.746     0.07x
tile[32][33], by column (padded)      0.166     1.08x

  Column access costs 15.34x. One extra column - 128 bytes per tile -
  recovers 101% of it.

--- Experiment 3 - broadcast ---
  Every thread reads the SAME address: 0.166 ms (0.92x vs stride 1)
```

### Đọc kết quả một cách trung thực

**Mức chậm đo được chỉ bằng khoảng một nửa số chiều xung đột dự đoán** (32 chiều
tốn ~15 lần chứ không phải 32), và stride 2 không cho thấy mức phạt nào. Đó không
phải mô hình sai — mỗi vòng lặp còn làm một phép cộng, một phép mask và một lệnh rẽ
nhánh, và vài chu kỳ shared memory dôi ra nấp sau chúng. Chỉ khi mức xung đột đủ
lớn thì shared memory mới trở thành ràng buộc quyết định.

**Hãy dự đoán TỈ LỆ từ mô hình; còn chi phí tuyệt đối thì phải đo.**

## Điểm cốt lõi

- **Xung đột bank là hiện tượng của shared memory, không liên quan tới coalescing.**
  Một kernel có thể coalesce hoàn hảo mà vẫn mất 15 lần ở đây.
- **`gcd(stride, 32)` là toàn bộ mô hình.** Stride 33 thắng stride 32.
- **Broadcast miễn phí.** 32 thread cùng muốn *một* địa chỉ là trường hợp tốt nhất,
  không phải tệ nhất. Xung đột cần 32 địa chỉ *khác nhau* trong một bank.
- **Hãy đệm thêm một phần tử** mỗi khi một tile được ghi theo hàng và đọc theo cột.
  Đó là cách sửa rẻ nhất trong CUDA.

## Đi xa hơn

- `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` — số xung đột
  đo được có khớp `gcd(s, 32)` không?
- Thử tile kiểu `double` (8 byte). Ánh xạ bank thay đổi thế nào?
- Đệm thêm 2, 4 và 8 thay vì 1. Mức đệm nào hiệu quả, và vì sao?
- Tăng block lên 256 thread. Hiệu ứng còn không, hay có nhiều warp hơn thì nó bị che đi?
