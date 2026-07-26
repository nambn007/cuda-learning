<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 07 - Mô hình roofline

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [05](../05-cache-and-bandwidth/), [06](../06-simd-intrinsics/)

## Mục tiêu

Dựng đường roofline của chính máy bạn từ số liệu đo được, và học cách trả lời câu
hỏi phải đến **trước** mọi việc tối ưu: *kernel này bị giới hạn bởi bộ nhớ hay bởi
phép tính?* Đoán sai nghĩa là tối ưu nhầm chỗ, và chỉ riêng mô hình này đã ngăn
được phần lớn công sức lãng phí trong suốt phần còn lại của lộ trình.

## Kiến thức nền

```
GFLOP/s đạt được = min( GFLOP/s đỉnh , cường độ số học × GB/s đỉnh )
```

**Cường độ số học (arithmetic intensity, AI)** = số FLOP thực hiện trên mỗi byte
di chuyển. Vẽ AI trên trục hoành và hiệu năng đạt được trên trục tung, bạn sẽ có
hai đường thẳng: một đường dốc (bị giới hạn băng thông) và một đường ngang (bị
giới hạn tính toán). Chỗ chúng gặp nhau là **điểm gãy (ridge point)**.

```
 GFLOP/s
   ^
   |            mái phẳng = đỉnh tính toán
   |        ____________________________
   |       /
   |      /   <- mái dốc = AI × băng thông đỉnh
   |     /
   |    /
   +---+------------------------------------> cường độ số học
      điểm gãy
```

- **AI dưới điểm gãy → nghẽn bộ nhớ.** Cách duy nhất đi lên là di chuyển ít byte
  hơn: bố cục tốt hơn, coalescing, tiling, gộp kernel. Thêm phép tính là *miễn phí*.
- **AI trên điểm gãy → nghẽn tính toán.** Lúc này lệnh mới quan trọng: SIMD, FMA,
  fast math, Tensor Core.

Điểm gãy điển hình: CPU desktop ~10 FLOP/byte, RTX 3060 ~37 FLOP/byte, A100 hơn 50.
**Điểm gãy dâng lên qua mỗi thế hệ phần cứng**, bởi phép tính rẻ đi nhanh hơn bộ
nhớ. Đó là lý do gần như mọi kernel bạn từng profile đều hoá ra nghẽn bộ nhớ — và
là lý do Giai đoạn 3 của lộ trình này chủ yếu nói về bộ nhớ.

## Nhiệm vụ của bạn

Mở `main.cpp`:

1. **`intensityKernel<TILE>(in, out, n, fmaCount)`** — đọc một float, thực hiện
   `fmaCount` phép FMA, ghi một float. Lưu lượng cố định ở 8 byte/phần tử trong
   khi khối lượng tính toán tăng dần, nên quét `fmaCount` chính là quét AI.
   Hãy tổ chức nó thành một **register tile**: giữ `TILE` phần tử với `TILE` biến
   tích luỹ độc lập. Một biến tích luỹ duy nhất phải chờ chính kết quả trước đó
   của nó và chỉ đạt một phần tư đỉnh; nhiều biến sẽ giữ pipeline FMA luôn đầy.
   `TILE` là tham số template để trình biên dịch bung hết vòng lặp và giữ được
   tile trong thanh ghi.
2. **`triad()`** — `a[i] = b[i] + s*c[i]` để đo mái băng thông.
3. **Mái băng thông** — chạy triad trên mảng 64 MB.
4. **Mái tính toán** — chạy `fmaCount = 256` trên mảng vừa cache với
   `TILE` = 4, 8, 16, 32, 64, 128 rồi lấy giá trị tốt nhất. Quan sát thông lượng
   leo lên, rồi tụt xuống khi tile không còn vừa thanh ghi.
5. **Quét** `kIntensitySweep` (dùng tile thắng cuộc) rồi so kết quả đạt được với
   `rooflineBound()`.

## Build và chạy

```bash
cmake --build build --target p1_07_roofline_model -j
./build/bin/p1/p1_07_roofline_model
```

## Kết quả mong đợi

Đo đơn luồng trên một CPU desktop:

```
--- Roof 2 - FMA throughput (and the register cliff) ---
  tile width            GFLOP/s   independent FMA chains
  --------------------------------------------------------------
  4                         9.9   4
  8                        17.2   8
  16                       31.1   16
  32                       64.2   32
  64                      121.8   64
  128                     118.3   128

  Best: tile 64 at 121.8 GFLOP/s.

--- The roofline ---
  peak bandwidth               17.043 GB/s
  peak throughput             121.842 GFLOP/s
  ridge point                   7.149 FLOP/byte

--- Intensity sweep ---
  FMAs         AI     achieved     roofline  of roof  attainable performance
  --------------------------------------------------------------------------------
  1           0.2          3.6          4.3      84%  |            | memory bound
  4           1.0         14.0         17.0      82%  |###         | memory bound
  16          4.0         48.0         68.0      70%  |#####       | memory bound
  64         16.0        118.0        121.8      97%  |###########| compute bound
  256        64.0        119.0        121.8      98%  |###########| compute bound
```

Hai điều đáng chú ý. Bảng quét tile làm thông lượng tăng gấp đôi bốn lần liên tiếp
— đó thuần tuý là **song song mức lệnh**, không hề làm thêm việc gì. Và biểu đồ
cột trong phần quét cường độ *chính là* đường roofline: đi lên khi còn nghẽn bộ
nhớ, rồi nằm ngang.

> **Cả hai mái ở đây đều là đơn luồng.** Mái tính toán của cả một con chip tăng
> theo số nhân trong khi mái băng thông thì không (mọi nhân dùng chung một bộ điều
> khiển bộ nhớ), nên điểm gãy của toàn chip cao gấp vài lần con số ~7 mà bạn đo
> được. RTX 3060 nằm ở khoảng **37** FLOP/byte.

## Điểm cốt lõi

- **Hãy đo các mái, đừng tin thông số kỹ thuật.** Băng thông đạt được thường chỉ
  bằng 60–80% giá trị lý thuyết.
- **Phân loại trước khi tối ưu.** Kernel nghẽn bộ nhớ sẽ không nhanh lên nhờ lệnh
  tốt hơn, và kernel nghẽn tính toán sẽ không nhanh lên nhờ bố cục tốt hơn.
- **Hiệu suất được tính so với mái, không phải so với đỉnh.** Một kernel chạy
  4 GFLOP/s vẫn có thể đang đạt 98% trần của nó — và như vậy là *xong*.
- **Độ trễ cần công việc độc lập.** Một chuỗi phụ thuộc chỉ đạt một phần tư đỉnh;
  cách sửa là thêm chuỗi, cho tới khi hết thanh ghi.
- Điểm gãy dâng lên mỗi thế hệ, nên **tỉ lệ kernel bị nghẽn bộ nhớ ngày càng tăng**.

## Đi xa hơn

- Thêm mái thứ ba cho dữ liệu nằm trong L2: chạy lại phép quét trên mảng vừa
  cache. Bạn được một *hệ phân cấp* roofline, đúng cách Nsight Compute trình bày
  cho GPU.
- Tính AI của các biến thể matmul ở bài 02 rồi đặt chúng lên biểu đồ. Blocking có
  đẩy kernel sang phải không?
- Đọc Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
  Model* (CACM 2009).
- Ở Giai đoạn 3 bài 16, Nsight Compute sẽ tự vẽ biểu đồ này cho các kernel GPU của bạn.
