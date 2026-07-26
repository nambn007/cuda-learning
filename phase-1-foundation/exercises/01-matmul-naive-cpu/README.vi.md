<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 01 - Nhân ma trận naive trên CPU

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐ | Thời gian: ~45 phút | Yêu cầu: không

## Mục tiêu

Viết bản cài đặt tham chiếu mà mọi bài tập sau này sẽ so sánh với nó, và — quan
trọng hơn — học cách **đo đạc** một cách trung thực. Kết thúc bài này bạn có thể
phát biểu kết quả bằng GFLOP/s và giải thích được vì sao gọi một hàm `clock()`
bao quanh đoạn code chưa phải là benchmark.

## Kiến thức nền

Nhân ma trận là ví dụ xuyên suốt toàn bộ lộ trình này. Nó xuất hiện lại dưới dạng
CUDA kernel naive (Giai đoạn 2), kernel tiled dùng shared memory (Giai đoạn 3),
kernel multi-GPU (Giai đoạn 4) và bản so sánh với cuBLAS/Tensor Core (Giai đoạn 5).
Có một baseline CPU đáng tin khiến tất cả những con số kia trở nên có ý nghĩa.

Hai ý tưởng từ bài này đi theo bạn tới tận GPU:

**Bố cục row-major.** `C[r][c]` nằm tại `C[r * N + c]`. Các phần tử cùng một *hàng*
nằm liền nhau trong bộ nhớ; các phần tử cùng một *cột* cách nhau `N` float. Phần
cứng chuyển bộ nhớ theo cache line (64 byte = 16 float trên x86), nên duyệt với
stride 1 lấy được 16 float hữu ích mỗi line, còn duyệt với stride `N` chỉ lấy được 1.
Trên GPU đúng nguyên lý này xuất hiện với tên gọi *memory coalescing*.

**Đếm FLOP.** Câu lệnh trong cùng `acc += A[i][k] * B[k][j]` gồm một phép nhân và
một phép cộng: 2 phép toán dấu phẩy động. Nó chạy `N³` lần, nên thuật toán thực
hiện `2N³` FLOP bất kể bạn viết thế nào. Chia cho thời gian chạy ta được một tốc
độ có thể đem so với đỉnh lý thuyết của phần cứng — và so với GPU sau này.

Vì sao lấy trung vị (median) của nhiều lần chạy thay vì một lần đo duy nhất? Lần
chạy đầu phải trả giá cho page fault trên buffer đầu ra vừa cấp phát và cho việc
CPU vẫn đang ở xung nhịp nghỉ. Các lần sau có thể bị bộ lập lịch của hệ điều hành
ngắt quãng. Vòng warmup loại bỏ hiệu ứng thứ nhất; lấy trung vị loại bỏ hiệu ứng
thứ hai.

## Nhiệm vụ của bạn

Mở `main.cpp` và hoàn thành ba TODO:

1. **`matmulNaive()`** — vòng lặp ba tầng `i, j, k`. Cộng dồn vào biến cục bộ
   `float acc` bên trong vòng `j`, không cộng thẳng vào `C[i * n + j]`.
2. **Đo thời gian** bằng `timeCpuMs(số_lần, callable)` trong `common/timer.h`.
3. **Báo cáo kết quả** — `ResultTable` sẽ biến thời gian của bạn cùng với
   `matmulFlops(n)` thành GFLOP/s.

`reference.h` chứa kết quả chuẩn dùng để kiểm tra. Riêng bài đầu tiên này nó
chính là thuật toán bạn đang viết, nên hãy tự làm trước khi mở ra xem.

## Build và chạy

```bash
cmake --build build --target p1_01_matmul_naive_cpu -j
./build/bin/p1/p1_01_matmul_naive_cpu

# Lời giải tham chiếu
cmake --build build --target p1_01_matmul_naive_cpu_sol -j
./build/bin/p1/p1_01_matmul_naive_cpu_sol
```

## Kết quả mong đợi

```
============================================================
  Phase 1 / 01 - Naive matrix multiplication
============================================================
  Matrix size: 512 x 512 (1.0 MB per matrix)

--- Correctness ---
  [PASS] C == A * B (n=262144, max abs err ..., max rel err ...)

--- Performance ---
CPU matrix multiplication
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
--------------------------------------------------------------------------
naive ijk                           ~100        ~0.03      ~2.7      1.00x
```

Con số tuyệt đối phụ thuộc CPU của bạn; nằm trong khoảng 1–5 GFLOP/s là bình
thường với thứ tự vòng lặp này.

## Điểm cốt lõi

- Row-major nghĩa là **stride 1 dọc theo hàng**; duyệt theo cột chạm vào một cache
  line mới ở mỗi phần tử.
- Hãy báo cáo **tốc độ** (GFLOP/s, GB/s) chứ không phải mili-giây thô — tốc độ mới
  so sánh được giữa các kích thước bài toán và giữa các máy.
- Một benchmark cần **warmup + lặp lại + lấy trung vị**, nếu không bạn chỉ đang đo nhiễu.
- Kiểm tra tính đúng đắn *trước* khi đo hiệu năng, và đừng bao giờ so sánh số thực
  bằng `==`.

## Đi xa hơn

- Thử `n = 1024` rồi `n = 2048`. Thời gian tăng theo `N³`, nên gấp đôi kích thước
  là gấp 8 lần khối lượng tính toán — kết quả đo có khớp không?
- Xem mã assembly sinh ra: `g++ -O3 -S -march=native`. Trình biên dịch có vector
  hoá vòng lặp trong cùng không? Vì sao không?
- Đọc *Computer Organization and Design* (Patterson & Hennessy), chương 5, về
  phân cấp bộ nhớ.
