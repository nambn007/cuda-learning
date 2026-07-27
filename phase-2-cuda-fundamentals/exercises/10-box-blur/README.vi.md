<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 10 - Box blur: stencil, halo và tính khả tách

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [09](../09-rgb-to-grayscale/README.vi.md) | **Cần GPU**

## Mục tiêu

Viết **stencil** đầu tiên của bạn — mẫu nằm sau tích chập, bộ lọc ảnh, bộ giải PDE
và automat tế bào — và học rằng thắng lợi lớn nhất có được ở đây hoàn toàn không
phải một kỹ thuật GPU. Đó là đại số.

## Kiến thức nền

Box blur thay mỗi điểm ảnh bằng trung bình của `(2R+1)²` điểm ảnh xung quanh. Mọi
stencil đều chung ba vấn đề:

**1. Tái sử dụng.** Các điểm đầu ra kề nhau đọc những vùng đầu vào chồng lấn. Với
bán kính 4, kernel naive đọc mỗi điểm ảnh đầu vào tới **81 lần**. Cache bù lại
được kha khá, nhưng đây vẫn là chi phí chủ đạo.

**2. Biên.** Điểm ảnh ở rìa không có hàng xóm ở một phía. Clamp (lặp lại điểm ảnh
biên) là cách xử lý chuẩn, và nó không miễn phí — bài này đo chính xác nó tốn bao
nhiêu, và con số lớn hơn hầu hết mọi người đoán.

**3. Tính khả tách.** Box filter là *tích* của một box ngang và một box dọc, nên
làm mờ theo hàng rồi theo cột cho ra **đúng** cùng kết quả — không phải xấp xỉ —
mà chỉ đọc `2(2R+1)` điểm ảnh thay vì `(2R+1)²`:

| Bán kính | Số tap naive | Số tap khả tách | Tỉ lệ |
|---|---|---|---|
| 2 | 25 | 10 | 2.5× |
| 4 | 81 | 18 | **4.5×** |
| 8 | 289 | 34 | 8.5× |

Đây là thắng lợi từ **đại số**, không phải từ mẹo GPU nào. Nó áp dụng được trên cả
CPU, nó kết hợp được với mọi phép tối ưu khác, và nó lớn dần theo bán kính.
**Hãy luôn tìm tính khả tách trước khi nghĩ tới shared memory.**

**Một lưu ý về độ chính xác:** mọi thứ ở đây là số học số nguyên, nên host và
device khớp nhau từng bit và bạn dùng được `checkArrayExact()` — khác với phép
luma dấu phẩy động ở bài 09. Tuy nhiên bộ đệm trung gian phải rộng hơn một byte:
tổng ngang của chín điểm ảnh có thể lên tới 2295.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **`blur2D()`** — stencil naive `(2R+1)²` với `clampDev()` ở biên. Làm tròn bằng
   `(sum + window/2) / window` để khớp CPU chính xác.
2. **`blurHorizontal()` / `blurVertical()`** — hai lượt khả tách. Hãy tự suy ra vì
   sao `tmp` phải là `unsigned short`.
3. **`blur2DNoBoundary()`** — chỉ xử lý phần trong, không clamp. Nó cố tình sai ở
   gần biên; nó tồn tại chỉ để đo xem việc clamp tốn bao nhiêu.
4. **Báo cáo số lần đọc trên mỗi điểm ảnh** bên cạnh thời gian đo được, để thấy
   bao nhiêu phần trong mức 4.5× dự đoán thật sự thành hiện thực.

## Build và chạy

```bash
cmake --build build --target p2_10_box_blur -j
cd build/testrun && ../bin/p2/p2_10_box_blur
```

Chương trình ghi ra `output_blur.pgm` — hãy mở nó. Bàn cờ phải mềm đi và đường
tròn phải có viền mờ. Nếu phần rìa trông sai thì phần clamp đang sai.

## Kết quả mong đợi

RTX 3060, 2048×2048, bán kính 4:

```
--- Correctness ---
  [PASS] naive 2D stencil (n=8388608, exact match)
  [PASS] separable (two 1D passes) (n=8388608, exact match)

--- Performance ---
Variant                           Time (ms)      GB/s   Speedup
----------------------------------------------------------------
CPU naive 2D                        263.170      2.61     1.00x
GPU naive 2D                          1.888    364.29   139.37x
GPU 2D, no boundary handling          0.918    748.93   286.53x
GPU separable (2 passes)              0.588    285.44   447.74x

  Reads per output pixel
    naive 2D    : 81       separable : 18       ratio : 4.5x fewer
    measured    : 3.21x faster
```

Hai kết quả đáng ngẫm:

- **Tính khả tách mang lại 3.21× trên mức 4.5× dự đoán.** Phần chênh là chi phí
  launch thêm một kernel và một vòng đi-về qua bộ đệm trung gian.
- **Xử lý biên tốn 2.06×.** Không phải do phân kỳ warp — chỉ vài warp ở đúng rìa
  mới phân kỳ — mà do số lượng: **162 cặp min/max số nguyên trên mỗi điểm ảnh đầu
  ra**, tất cả đều nằm trong vòng lặp trong cùng của một kernel vốn gần như thuần
  đọc bộ nhớ.

## Điểm cốt lõi

- **Hãy tìm cấu trúc đại số trước.** Tính khả tách đánh bại mọi phép vi tối ưu có
  sẵn ở giai đoạn này, và ở bán kính 8 nó sẽ đáng giá 8.5 lần.
- **Xử lý biên là chi phí thật, không phải chi tiết vụn.** Cách sửa trong sản xuất
  là đệm thêm `radius` điểm ảnh quanh đầu vào để khỏi cần clamp, hoặc tách lệnh
  launch thành một kernel cho phần trong và một kernel mỏng cho phần biên.
- **Số học số nguyên cho kết quả khớp từng bit** với bản tham chiếu trên host. Hãy
  dùng nó khi có thể.
- Cache che giấu rất nhiều phần đọc dư thừa — đó là lý do bản naive vẫn nhanh hơn
  CPU 139 lần dù đọc mọi thứ tới 81 lần.

## Đi xa hơn

- Chạy lại với bán kính 2, 8 và 16. Mức tăng tốc của bản khả tách có bám theo
  `(2R+1)/2` không?
- Box blur có một cách viết dùng **tổng chạy (running sum)** với chi phí O(1) mỗi
  điểm ảnh bất kể bán kính. Hãy tự dẫn ra, cài đặt, rồi so sánh.
- Ba lần box blur liên tiếp xấp xỉ một Gaussian. Thử và nhìn kết quả.
- **Tiếp theo:** [GĐ 3/14 conv-2d-shared](../../../phase-3-intermediate/exercises/14-conv-2d-shared/)
  dàn một tile *cùng vành halo của nó* vào bộ nhớ `__shared__` để mỗi điểm ảnh đầu
  vào chỉ được đọc từ bộ nhớ toàn cục đúng một lần, còn
  [GĐ 3/13](../../../phase-3-intermediate/exercises/13-conv-1d-constant/) đặt các
  trọng số bộ lọc vào bộ nhớ `__constant__`.
