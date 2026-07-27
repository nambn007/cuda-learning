<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 02 - Cache blocking và thứ tự vòng lặp

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [01](../01-matmul-naive-cpu/)

## Mục tiêu

Làm cho đúng những phép tính đó chạy nhanh gấp nhiều lần mà không đổi một công
thức nào — chỉ đổi thứ tự truy cập bộ nhớ. Đây là bài học quan trọng nhất của cả
lộ trình, bởi vì *chính ý tưởng này* sẽ quay lại dưới dạng shared-memory tiling ở
Giai đoạn 3.

## Kiến thức nền

Vòng lặp naive `ijk` đọc `B[k][j]` với vòng `k` nằm trong cùng, tức là nó duyệt
**dọc theo một cột** của `B` — stride `N`. Mỗi lần truy cập như vậy kéo về một
cache line 64 byte mới mà chỉ dùng đúng 4 byte. 60 byte còn lại bị đẩy ra khỏi
cache trước khi vòng lặp quay lại cần đến chúng.

**Cách sửa 1 — đổi thứ tự thành `ikj`.** Chuyển `k` ra giữa:

```
for i:
  for k:
    a = A[i][k]                  // không phụ thuộc j, đưa ra ngoài
    for j:
      C[i][j] += a * B[k][j]     // B và C đều stride 1
```

Giờ vòng trong cùng là một phép cộng vector có hệ số trên vùng nhớ liên tục. Hai
thứ cải thiện cùng lúc: mọi cache line đều được dùng hết, và vòng lặp có dạng mà
trình biên dịch tự động vector hoá được thành lệnh AVX FMA.

**Cách sửa 2 — chia khối (blocking/tiling).** `ikj` vẫn quét toàn bộ `B` một lần
cho mỗi hàng của `A`. Với `N = 768`, `B` chiếm 2.25 MB, lớn hơn một lát L2 điển
hình, nên mỗi lượt quét lại phải lấy từ DRAM. Blocking giới hạn tập dữ liệu đang
hoạt động: khi tính một tile `T×T` của `C`, chỉ có một tile `T×T` của `A` và một
tile `T×T` của `B` là đang sống. Ba tile 64×64 float là 48 KB — chúng nằm gọn
trong cache, và mỗi phần tử của `B` chỉ còn bị đọc `N/T` lần thay vì `N` lần.

Câu cuối đáng đọc lại. **Blocking biến lưu lượng bộ nhớ thành tái sử dụng.**

### Một cảnh báo thành thật về kết quả

Trên CPU desktop hiện đại, blocking thường **gần như không đem lại gì** — đôi khi
còn chậm hơn một chút. Đó không phải lỗi code của bạn, và bài tập có in ra một
bảng quét kích thước để bạn thấy chính xác điểm giao cắt nằm ở đâu. Lý do:

- CPU vốn đã có 4–32 MB L3 cùng một bộ prefetch phần cứng rất hung hăng. Với
  `n ≤ 1024` nó đang làm việc blocking *hộ bạn* rồi.
- Bản blocked phải trả thêm chi phí vòng lặp và làm vòng trong ngắn lại, khiến bộ
  vector hoá kém hiệu quả hơn — đủ để triệt tiêu phần lợi thu được.
- Chỉ khi các ma trận vượt hẳn L3 (`n = 2048`, `B` = 16 MB) thì blocking thủ công
  mới bắt đầu thắng — và cũng chỉ thắng khiêm tốn.

**Sự giằng co này xuất hiện lại trên GPU, và câu trả lời ở đó thú vị hơn.** Giai
đoạn 3 bài 04 viết lại đúng cấu trúc vòng lặp này bằng bộ nhớ `__shared__` rồi đo:
tiling thủ công tự nó chỉ mua được ~1.2 lần, đúng vì lý do bạn đang thấy ở đây —
cache phần cứng vốn đã nắm được phần tái sử dụng. Thắng lợi lớn (3 lần) đến từ một
tầng tiling *thứ hai*, xuống thanh ghi, thứ mà cache không làm hộ bạn được. Hãy nhớ
điều đó khi bạn tới bài ấy.

Vậy bài học rút ra ở đây thật ra gồm hai điều: **thứ tự vòng lặp là một chiến
thắng lớn và miễn phí ở mọi nơi**, và **blocking thủ công chỉ đáng làm khi phần
cứng ngừng cache hộ bạn** — mà trên GPU thì điều đó luôn đúng.

## Nhiệm vụ của bạn

Mở `main.cpp`:

1. **`matmulIKJ()`** — vòng lặp đã đổi thứ tự. Nhớ zero `C` trước; phiên bản này
   cộng dồn.
2. **`matmulBlocked()`** — sáu vòng lặp: `ii, kk, jj` duyệt các tile, rồi `i, k, j`
   bên trong tile. Dùng `std::min(ii + tile, n)` làm cận trên để kích thước không
   chia hết cho tile vẫn chạy đúng.
3. **Đo thời gian** cả hai và thêm vào `ResultTable`.

## Build và chạy

```bash
cmake --build build --target p1_02_matmul_cache_blocking -j
./build/bin/p1/p1_02_matmul_cache_blocking
```

## Kết quả mong đợi

Đo trên một CPU desktop 12 luồng tại `n = 1024`:

```
--- Performance ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
naive ijk (baseline)               1619.395         -       1.33     1.00x
ikj order                           154.051         -      13.94    10.51x
blocked ikj                         200.687         -      10.70     8.07x
blocked + OpenMP (12 threads)        42.458         -      50.58    38.14x

--- When blocking starts to matter ---
  n              B size     ikj (ms) blocked (ms)    speedup
  --------------------------------------------------------------
  256            0.2 MB          2.3          2.4      0.94x
  512            1.0 MB         19.3         19.6      0.98x
  1024           4.0 MB        155.7        208.6      0.75x
  2048          16.0 MB       2201.1       2067.0      1.06x
```

Hãy đọc kỹ bảng thứ hai — đó mới là nội dung thật của bài này. Blocking *thua* cho
tới khi tập dữ liệu vượt quá cache, rồi mới chuyển thành thắng. Điểm giao cắt cụ
thể là thuộc tính của **máy bạn**.

## Điểm cốt lõi

- **Thứ tự vòng lặp thay đổi hiệu năng 10 lần với cùng một lượng phép tính.** Nút
  thắt là bố cục bộ nhớ, không phải số FLOP.
- Vòng trong cùng nên có **stride 1** — vừa vì cache, vừa vì bộ vector hoá. Thực
  ra phần lớn lợi ích của `ikj` đến từ auto-vectorisation.
- **Blocking thủ công chỉ có lời khi phần cứng ngừng cache hộ bạn.** Trên CPU có
  L3 lớn nghĩa là phải `n` thật lớn; trên GPU nghĩa là *luôn luôn*.
- Tồn tại một kích thước tile tối ưu, và nó là thuộc tính của *cache* chứ không
  phải của thuật toán. Hãy đo, đừng đoán.
- Đa nhân cho thêm khoảng 4 lần nữa — và vẫn còn kém GPU cả trăm lần, như bạn sẽ
  thấy ở Giai đoạn 3.

## Đi xa hơn

- Chạy lại với `n = 2048`. Khoảng cách giữa naive và blocked rộng ra — vì sao?
- Thêm `-march=native` rồi so sánh. Bao nhiêu phần trong lợi ích của `ikj` đến từ
  vector hoá? (`g++ -O3 -march=native -fopt-info-vec`)
- So kết quả tốt nhất của bạn với `sgemm` của OpenBLAS. Dự kiến còn kém 5–20 lần;
  phần chênh còn lại đến từ register blocking, packing và microkernel viết tay.
- Đọc bài ["How to optimize a CUDA matmul kernel"](https://siboehm.com/articles/22/CUDA-MMM)
  của Simon Boehm — đúng tiến trình này, nhưng trên GPU.
