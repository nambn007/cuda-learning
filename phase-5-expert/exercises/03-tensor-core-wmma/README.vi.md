<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 03 - Tensor Core qua WMMA

> **Giai đoạn 5 · Chuyên sâu** | Độ khó: ⭐⭐⭐⭐⭐ | Thời gian: ~3 giờ | Yêu cầu: [02](../02-cublas-gemm/README.vi.md) | **Cần sm_70+** | ✅ đã kiểm chứng trên RTX 3060

## Mục tiêu

Dùng đơn vị phần cứng khiến GPU hiện đại nhanh với học sâu — và đo xem nó tốn gì về
độ chính xác, thay vì coi như miễn phí.

## Kiến thức nền

Một Tensor Core tính một phép nhân-cộng ma trận nhỏ trong **một lệnh**:

```
D = A × B + C     A, B: 16×16 half     C, D: 16×16 float
```

Đó là 16·16·16·2 = **8192 FLOP mỗi lệnh**, và đó là lý do một card có Tensor Core
công bố con số FP16 gấp nhiều lần con số FP32 của nó.

**API WMMA** (`nvcuda::wmma`) cung cấp chúng ở mức **warp**:

```cuda
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16,16,16, float>             c;
wmma::fill_fragment(c, 0.0f);
wmma::load_matrix_sync(a, ptrA, lda);
wmma::mma_sync(c, a, b, c);
wmma::store_matrix_sync(ptrC, c, ldc, wmma::mem_row_major);
```

**Ba quy tắc mà API áp đặt:**

1. **Cả warp** cùng làm một tile. Mọi thread phải gọi mọi hàm `wmma`, một cách đồng
   nhất — một lời gọi có phân kỳ là hành vi không xác định.
2. **Đừng bao giờ truy cập chỉ số bên trong fragment.** Bố cục thanh ghi của nó cố
   ý không được đặc tả; bạn chỉ nạp, nhân và ghi.
3. Con trỏ cần căn chỉnh 256 bit, và leading dimension phải là bội của 16 với `half`.

**Độ chính xác là một đánh đổi thật.** Đầu vào là FP16 (~3 chữ số thập phân); biến
tích luỹ là FP32. Với GEMM thì thường ổn vì phần tích luỹ chi phối sai số — nhưng
bài này *đo* nó thay vì nói qua loa. BF16 (sm_80+) đánh đổi bit phần định trị lấy
dải mũ của FP32, an toàn hơn nhiều khi huấn luyện; TF32 là tự động trên Ampere cho
GEMM FP32 trừ khi bạn tắt đi.

## Nhiệm vụ của bạn

Viết `gemmFp32` (mốc cơ sở) và `gemmWmma`. Launch WMMA với `blockDim(32, 4)` — 32
thread theo x đúng bằng một warp.

Kiểm chứng với sai số tương đối ~2%. **Nếu thấy mình phải nới rộng thêm nữa, hãy
dừng lại và nghĩ xem FP16 có hợp với dữ liệu của bạn không**, thay vì nới lỏng phép
kiểm tra cho tới khi nó pass.

## Build và chạy

```bash
cmake --build build --target p5_03_tensor_core_wmma -j
./build/bin/p5/p5_03_tensor_core_wmma
```

## Kết quả mong đợi

RTX 3060 (sm_86), n = 1024:

```
FP32, one thread per element          2.656 ms      808.45 GFLOP/s     1.00x
FP16 Tensor Core (WMMA)               0.378 ms     5683.34 GFLOP/s     7.03x
```

**7 lần** — và lưu ý đây là kernel *dạy học*. Nó đọc A và B thẳng từ bộ nhớ toàn
cục, không dàn qua shared memory, nên phần lớn thông lượng của Tensor Core bị tiêu
vào việc chờ bộ nhớ. Hãy dàn tile như ở
[GĐ 3/04](../../../phase-3-intermediate/exercises/04-tiled-matmul/README.vi.md) và
nó sẽ đi xa hơn đáng kể. Đó chính là điều CUTLASS làm, qua khoảng hai chục tầng.

## Điểm cốt lõi

- **Một lệnh, 8192 FLOP.** Đó là toàn bộ câu chuyện vì sao các con số thông lượng
  FP16 trông như vậy.
- **WMMA ở mức warp và phải đồng nhất.** Fragment cố ý được thiết kế mờ đục.
- **Mức tăng tốc không miễn phí** — bạn đã đổi độ chính xác đầu vào. Hãy đo sai số
  trên dữ liệu *của bạn*, không phải trên số ngẫu nhiên đều.
- **Đơn vị tính toán vô dụng nếu bộ nhớ không nuôi kịp.** Một kernel WMNA ngây thơ
  để phần lớn phần cứng ngồi không.

## Đi xa hơn

- Thêm dàn shared memory quanh vòng WMMA rồi đo lại.
- Thử `__nv_bfloat16` thay cho `half` và so sai số.
- So với `cublasGemmEx` dùng `CUBLAS_COMPUTE_16F`.
- Đọc cấu trúc GEMM của [CUTLASS](https://github.com/NVIDIA/cutlass) và đếm số tầng tiling.
