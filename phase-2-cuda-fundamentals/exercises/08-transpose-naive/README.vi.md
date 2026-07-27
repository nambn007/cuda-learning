<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 08 - Chuyển vị naive: kernel không thể coalesce

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [06](../06-matrix-add-2d/README.vi.md) | **Cần GPU**

## Mục tiêu

Gặp kernel đầu tiên mà **cẩn thận thôi là chưa đủ**. Chuyển vị không làm phép tính
nào — mỗi phần tử một lần đọc, một lần ghi, y hệt phép copy — vậy mà nó chỉ chạy
bằng một phần ba tốc độ copy. Không cách sắp xếp chỉ số nào chữa được. Đó chính là
thứ dọn đường cho shared memory ở Giai đoạn 3.

## Kiến thức nền

```cuda
out[col][row] = in[row][col];
```

Một warp biến thiên theo một chỉ số. Chọn chỉ số nào thì cũng có một bên liên tục
và bên kia rời rạc:

| Warp biến thiên | Đọc `in[row*cols + col]` | Ghi `out[col*rows + row]` |
|---|---|---|
| `col` | 32 float liên tiếp ✅ | 32 địa chỉ cách nhau `rows` ❌ |
| `row` | 32 địa chỉ cách nhau `cols` ❌ | 32 float liên tiếp ✅ |

**Sự rời rạc đó không phải lỗi trong cách bạn đánh chỉ số. Sự rời rạc đó *chính là*
phép chuyển vị.**

**Hãy đo so với phép copy, đừng so với đỉnh.** Copy di chuyển đúng bằng ấy byte,
nên nó là tốc độ tối đa của bài toán này. "93 GB/s" tự nó chẳng nói lên gì; "29%
của copy" mới là con số cho biết còn bao nhiêu dư địa.

**Khi buộc phải để một bên rời rạc, hãy để bên đọc rời rạc.** Sự bất đối xứng này
là thật:

- Một lần **đọc** rời rạc vẫn có thể trúng L1/L2 — một warp lân cận có thể đã kéo
  sector đó về rồi, và những lần trượt được che bởi các warp khác đang thường trú.
- Một lần **ghi** rời rạc thì không có lối thoát nào. Nó rốt cuộc phải xuống DRAM,
  và ghi một phần sector là phí phần còn lại của sector đó, không có tái sử dụng
  nào bù lại.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`copyKernel()`** — mốc cơ sở.
2. **`transposeReadCoalesced()`** — `x` → `col`.
3. **`transposeWriteCoalesced()`** — `x` → `row`. Thân hàm giống hệt, chỉ khác hình
   dạng grid.
4. **Báo cáo mỗi bản chuyển vị theo phần trăm băng thông của copy.**

Block gợi ý: `dim3(32, 8)` = 256 thread, rộng 32 để một warp trải đúng một hàng của
block.

Rồi, trước khi đọc lời giải: **hãy thử sắp xếp lại chỉ số sao cho cả hai bên đều
coalesced.** Tự thuyết phục mình về câu trả lời.

## Build và chạy

```bash
cmake --build build --target p2_08_transpose_naive -j
./build/bin/p2/p2_08_transpose_naive
```

## Kết quả mong đợi

RTX 3060, 4096×4096:

```
Variant                           Time (ms)      GB/s   Speedup
---------------------------------------------------------------
copy (speed of light)                 0.415    323.63     1.00x
transpose, coalesced read             1.442     93.11     0.29x
transpose, coalesced write            0.990    135.54     0.42x
CPU transpose                       160.717      0.84     0.00x

  copy                               323.6 GB/s  ( 90% of DRAM peak)
  transpose, coalesced read           93.1 GB/s    29% of copy
  transpose, coalesced write         135.5 GB/s    42% of copy
```

Ba điều đáng chú ý:

1. **Copy đạt 90% đỉnh DRAM** — phần cứng không có vấn đề gì, vấn đề là mẫu truy cập.
2. **Bản chuyển vị tốt nhất đạt 42% của copy.** Hơn một nửa băng thông bị vứt đi
   vào các sector dùng dở.
3. **CPU chỉ được 0.84 GB/s** — chênh 190 lần, vì phép ghi có bước nhảy cũng phá
   cache của nó y như vậy, thậm chí tệ hơn.

## Điểm cốt lõi

- **Có những mẫu truy cập không thể coalesce bằng cách sắp xếp lại chỉ số.** Sự
  rời rạc là bản chất của phép toán.
- **Hãy đo so với mốc đúng.** So GB/s với đỉnh DRAM là tâng bốc phép chuyển vị; so
  với copy mới là nói thật.
- **Nếu buộc phải chọn, hãy giữ phần ghi coalesced.** Phần đọc còn có L2 và cơ chế
  che độ trễ để nương vào; phần ghi thì không.
- Cùng hiện tượng đó phá bản CPU còn nặng hơn — đây chính là thí nghiệm stride ở
  GĐ 1/05, diễn ra trên cả hai bộ xử lý cùng lúc.

## Đi xa hơn

- Thử các hình dạng block `(32,8)`, `(16,16)`, `(32,32)`. Thứ hạng có đổi không?
- Chạy `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` trên cả hai
  bản chuyển vị: đếm số sector ghi rồi so với mô hình.
- Chuyển vị ma trận không vuông (4096×1024). Có gì thay đổi về mặt cấu trúc không?
- **Tiếp theo:** [GĐ 3/05 transpose-optimized](../../../phase-3-intermediate/exercises/05-transpose-optimized/)
  dàn một tile vào bộ nhớ `__shared__` để cả hai bên đều liên tục — rồi đụng phải
  xung đột bank, tức [GĐ 3/03](../../../phase-3-intermediate/exercises/03-bank-conflicts/).
