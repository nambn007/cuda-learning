<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 09 - RGB sang thang xám: AoS so với SoA

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [04](../04-saxpy/README.vi.md), [06](../06-matrix-add-2d/README.vi.md) | **Cần GPU**

## Mục tiêu

Làm việc trên dữ liệu ảnh thật, và kiểm chứng một câu "kinh nghiệm dân gian" của
GPU — *"hãy ưu tiên Structure of Arrays"* — bằng cách đo đạc. Kết quả đi ngược lại
lời khuyên đó, và hiểu vì sao sẽ cho bạn quy tắc đúng thay vì một khẩu hiệu.

## Kiến thức nền

Phép tính chỉ một dòng (hệ số luma ITU-R BT.601; mắt người nhạy với xanh lá hơn
xanh dương rất nhiều, nên lấy trung bình cộng đơn thuần sẽ trông sai):

```
gray = 0.299·R + 0.587·G + 0.114·B
```

**Câu hỏi về bố cục.** Ảnh được lưu xen kẽ — `RGBRGBRGB...` — tức Array of
Structures. Cách khác là ba mặt phẳng riêng, `RRRR...GGGG...BBBB` — Structure of
Arrays. Lời khuyên thường gặp nói SoA nhanh hơn trên GPU. Đo ở đây thì hai bên
ngang nhau:

| Bố cục | Một warp đọc gì | Số sector |
|---|---|---|
| AoS | 96 byte liên tiếp | 3 |
| SoA | ba dải 32 byte | 3 |

Cùng lưu lượng, cùng số lệnh, cùng tốc độ. **Kernel này cần cả ba kênh của mọi
điểm ảnh**, nên chẳng có gì để SoA tiết kiệm cả.

**Chỗ AoS thật sự gây hại** là khi bạn chỉ cần *một phần* của mỗi cấu trúc. Trích
riêng kênh đỏ thì AoS trở thành phép đọc stride 3: một warp chạm 96 byte mà chỉ
dùng 32, vứt đi hai phần ba mỗi sector. Bài tập đo cả trường hợp này.

> **Quy tắc phát biểu cho đúng:** AoS so với SoA quan trọng tỉ lệ thuận với việc
> bạn thật sự đọc bao nhiêu phần của mỗi cấu trúc. Cần mọi trường? Hai bố cục hoà.
> Cần một trường trong nhiều trường? SoA thắng xấp xỉ đúng bằng tỉ lệ đó.

Đó là lý do một hệ mô phỏng hạt với struct 32 byte hưởng lợi lớn từ SoA khi kernel
chỉ chạm `position` — và vì sao kernel thang xám này thì không.

**Thứ thật sự giúp ích ở đây** là vector hoá: cho mỗi thread lo bốn điểm ảnh, biến
mười hai lệnh nạp byte thành ba lệnh nạp 4 byte. 4 điểm ảnh = 12 byte = đúng ba
`uchar4`. Đó là thắng lợi về **số lệnh**, không phải về bố cục, và đáng khoảng 1.3 lần.

**Một điều nữa: đừng so ảnh bằng phép bằng tuyệt đối.** `nvcc` mặc định gộp
`0.299f*r + 0.587f*g + 0.114f*b` thành lệnh FMA, làm tròn một lần thay vì hai lần.
Khi giá trị thật nằm gần `x.5`, host và device làm tròn ra hai số nguyên khác nhau
— khoảng 1 trên 50.000 điểm ảnh lệch đúng một mức xám. Phép kiểm tra đúng là
"không điểm ảnh nào lệch quá một mức". Hãy biên dịch với `-fmad=false` nếu bạn thật
sự cần khớp từng bit với bản tham chiếu trên host.

## Nhiệm vụ của bạn

Mở `main.cu` và viết năm kernel:

1. **`grayInterleaved()`** — AoS.
2. **`grayPlanar()`** — SoA.
3. **`grayInterleavedVec4()`** — AoS với lệnh nạp `uchar4`. Hãy tự suy ra thành
   phần nào của `uchar4` nào ứng với kênh nào; làm sai sẽ ra một tấm ảnh trông sai
   thấy rõ.
4. **`redFromInterleaved()` / `redFromPlanar()`** — thí nghiệm phân định câu hỏi về
   bố cục.

**Trước khi chạy, hãy viết ra dự đoán của bạn** cho (a) AoS vs SoA ở thang xám,
(b) `uchar4` vs vô hướng, (c) AoS vs SoA ở kênh đỏ. Ít nhất một trong ba câu trả
lời có lẽ sẽ làm bạn bất ngờ.

Kiểm chứng bằng `checkImage()`, không dùng `checkArrayExact()`.

## Build và chạy

```bash
cmake --build build --target p2_09_rgb_to_grayscale -j
cd build/testrun && ../bin/p2/p2_09_rgb_to_grayscale
```

Chương trình tự tạo `input.ppm` nếu bạn không cung cấp, và ghi ra `output_gray.pgm`.
Hãy mở nó ra xem — một kernel xử lý ảnh sai tinh vi thường *trông* sai ngay, điều
mà không phép assert nào cho bạn miễn phí.

## Kết quả mong đợi

RTX 3060, 4096×2048:

```
--- Correctness ---
  [PASS] interleaved (AoS)              (max diff 1 level, 0.002% of pixels differ)
  [PASS] planar (SoA)                   (max diff 1 level, 0.002% of pixels differ)
  [PASS] interleaved with uchar4 loads  (max diff 1 level, 0.002% of pixels differ)

--- Performance ---
Variant                           Time (ms)      GB/s   Speedup
----------------------------------------------------------------
CPU (single thread)                  18.215      1.84     1.00x
GPU interleaved (AoS)                 0.142    236.38   128.32x
GPU interleaved, uchar4               0.106    315.08   171.04x
GPU planar (SoA)                      0.148    225.99   122.68x

--- When AoS really does hurt: reading one channel ---
red channel from AoS (stride 3)       0.136    123.19     1.00x
red channel from SoA (planar)         0.104    160.63     1.30x
```

**SoA chậm hơn AoS một chút ở thang xám, và nhanh hơn 1.30 lần ở một kênh đơn.**
Cả hai con số đều từ cùng dữ liệu, trên cùng GPU.

## Điểm cốt lõi

- **Hãy kiểm chứng những câu truyền miệng.** "Ưu tiên SoA" là một quy tắc kinh
  nghiệm có điều kiện kèm theo, và chính điều kiện đó mới quan trọng.
- **Bố cục quan trọng tỉ lệ thuận với phần cấu trúc mà bạn thật sự đọc.**
- **Số lệnh là một trục riêng, tách khỏi lưu lượng.** `uchar4` chuyển đúng bằng ấy
  byte mà nhanh hơn 1.3 lần.
- **Việc gộp FMA khiến host và device lệch nhau một bit**, mà với ảnh 8 bit thì là
  một mức xám. Hãy so sánh có sai số cho phép, hoặc tắt `-fmad`.
- Kernel xử lý ảnh có thêm một phép kiểm tra miễn phí: nhìn vào kết quả.

## Đi xa hơn

- Chuyển ảnh của chính bạn: `convert anh.jpg input.ppm`, rồi chạy lại.
- Thêm bản `uchar4` cho kernel planar. Lúc đó SoA có thắng AoS không?
- Thử một struct 8 trường mà kernel chỉ đọc một trường. Tỉ lệ thay đổi thế nào?
- Chạy `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` trên hai
  kernel kênh đỏ — xác nhận tỉ lệ sector 3:1.
