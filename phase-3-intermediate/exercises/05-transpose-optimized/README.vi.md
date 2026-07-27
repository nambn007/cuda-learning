<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 05 - Chuyển vị với tile trong shared memory

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [02](../02-shared-memory-basics/README.vi.md), [03](../03-bank-conflicts/README.vi.md), [GĐ 2/08](../../../phase-2-cuda-fundamentals/exercises/08-transpose-naive/README.vi.md) | **Cần GPU**

## Mục tiêu

Sửa cái kernel mà Giai đoạn 2 đã chứng minh là không thể sửa bằng cách sắp xếp lại
chỉ số — và trong quá trình đó, học được khi nào một xung đột bank mới thật sự tốn
kém.

## Kiến thức nền

GĐ 2/08 đã xác lập rằng `out[col][row] = in[row][col]` không thể coalesce cả hai
phía: dù warp biến thiên theo chỉ số nào thì một phía cũng rời rạc. **Sự rời rạc đó
*chính là* phép chuyển vị.**

Cách sửa là công dụng thứ hai của shared memory
([bài 02](../02-shared-memory-basics/README.vi.md)): **sắp xếp lại**. Shared memory
không có yêu cầu coalescing, nên:

1. đọc một khối `TILE×TILE` từ bộ nhớ toàn cục — **coalesced**
2. ghi nó vào một tile chia sẻ
3. `__syncthreads()`
4. đọc tile chia sẻ theo **chiều chuyển vị** — miễn phí, không có quy tắc coalescing
5. ghi ra bộ nhớ toàn cục — **coalesced**

Cả hai truy cập global giờ đều liên tục. Phép chuyển vị diễn ra hoàn toàn trên chip.

**Ngoại trừ** việc bước 4 duyệt một *cột* của tile chia sẻ:

```cuda
tile[threadIdx.x][threadIdx.y + j]
  địa chỉ = tx * 32 + (ty + j)
  bank    = (ty + j) % 32     ← giống hệt nhau cho cả 32 thread → xung đột 32 chiều
```

Đệm thành `tile[32][33]` làm bank trở thành `(tx + ty + j) % 32` — đủ 32 bank, tốn
128 byte mỗi tile. Đó là cách sửa của bài 03, áp dụng đúng chỗ.

**Một điểm dễ sai:** ở bước 3, tile đầu ra nằm ở *vị trí chuyển vị trong GRID*,
chứ không chỉ chuyển vị bên trong tile. Hãy tính lại `x` và `y` từ `blockIdx.y` và
`blockIdx.x` đổi chỗ cho nhau.

## Nhiệm vụ của bạn

Mở `main.cu` và viết bốn kernel: `copyKernel`, `transposeNaive`,
`transposeSharedNoPad`, `transposeSharedPadded`.

**Dự đoán trước khi đo:** ở bài 03 đúng xung đột 32 chiều này tốn **15 lần**. Bạn
có kỳ vọng điều tương tự ở đây không? Nếu không, kernel này khác ở chỗ nào?

## Build và chạy

```bash
cmake --build build --target p3_05_transpose_optimized -j
./build/bin/p3/p3_05_transpose_optimized
```

## Kết quả mong đợi

RTX 3060, 4096×4096:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
copy (speed of light)                 0.407    329.82     1.00x
transpose naive                       1.440     93.22     0.28x
transpose shared, no padding          0.440    304.82     0.92x
transpose shared, padded              0.417    322.17     0.98x

--- As a fraction of copy speed ---
  copy                               329.8 GB/s    100%
  naive                               93.2 GB/s     28%
  shared, no padding                 304.8 GB/s     92%
  shared, padded                     322.2 GB/s     98%

  naive -> shared          3.27x
  shared -> shared padded  1.06x
  naive -> shared padded   3.46x
```

### Xung đột bank tốn 6%, không phải 15 lần

Dàn dữ liệu qua shared memory mới là thắng lợi lớn: **3.27 lần**, từ 28% lên 92%
tốc độ copy.

Bản không đệm *thật sự* có xung đột bank 32 chiều — phép tính ở trên không sai.
Nhưng loại bỏ nó chỉ đáng **1.06 lần** ở đây, so với **15 lần** ở bài 03.

Khác biệt nằm ở chỗ **kernel bị giới hạn bởi cái gì**. Bài 03 là lưu lượng shared
memory thuần trong một vòng lặp chặt, nên mọi chu kỳ bị tuần tự hoá đều hiện ra
trong tổng thời gian. Kernel này đẩy 128 MB qua DRAM, và mấy chu kỳ shared memory
dôi ra nấp sau độ trễ đó.

> **Xung đột bank chỉ tốn kém khi shared memory là nút thắt.**

Cứ đệm — 128 byte đổi lấy 6% là món hời, và ngay khi bạn tối ưu phía global thêm
nữa thì xung đột sẽ thôi bị che. Nhưng **hãy đo trước khi cho rằng đó là vấn đề của
mình.**

### Vì sao là 98% chứ không phải 100%

Ba lý do mang tính cấu trúc: hai lệnh `__syncthreads()` mỗi tile mà phép copy không
cần; phép ghi rơi vào một hàng 4096 float khác nhau ứng với mỗi cột tile, làm mất
tính cục bộ trang DRAM mà một phép copy thẳng được hưởng; và shared memory giới hạn
số block mỗi SM.

Bù nốt 2% cuối cần sắp xếp lại block theo đường chéo hoặc dùng lệnh nạp vector hoá
— hoặc không chuyển vị nữa, vì nhiều thuật toán có thể tiêu thụ ma trận theo chiều
nào cũng được nếu bạn cho phép.

## Điểm cốt lõi

- **Shared memory có thể biến một truy cập không coalesce được thành coalesced**,
  bằng cách đưa phần rời rạc lên chip. Đây là công dụng thứ hai của nó, khác với
  tái sử dụng.
- **Vị trí grid của tile đầu ra cũng chuyển vị**, chứ không chỉ nội dung của nó.
- **Xung đột bank quan trọng tỉ lệ với mức độ bạn bị nghẽn ở shared memory.** Cùng
  một xung đột 32 chiều tốn 15 lần ở kernel này và 6% ở kernel kia.
- **Hãy đo xem tài nguyên nào đang ràng buộc trước khi tối ưu nó.** Đây là bài học
  được lặp lại nhiều nhất trong Giai đoạn 3.

## Đi xa hơn

- Profile cả hai bản shared với
  `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` — số xung đột
  chênh nhau rất lớn dù thời gian chạy gần như không đổi.
- Thử `kBlockRows` = 4, 8, 16, 32. Cái gì giới hạn nó?
- Cài sắp xếp block theo đường chéo (đổi ánh xạ `blockIdx`) để trải đều truy cập
  trang DRAM. Nó lấy lại được bao nhiêu trong 2% cuối?
- Thêm lệnh nạp `float4`. Tile có còn vừa vặn không?
