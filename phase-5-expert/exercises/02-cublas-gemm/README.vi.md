<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 02 - cuBLAS GEMM, và khoảng cách tới nó

> **Giai đoạn 5 · Chuyên sâu** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [GĐ 3/04](../../../phase-3-intermediate/exercises/04-tiled-matmul/README.vi.md) | **Cần GPU**
>
> ⚠️ **Chưa kiểm chứng trên máy tham chiếu** — không có số đo bên dưới.

## Mục tiêu

Giai đoạn 3/04 đưa bản tiled matmul lên ~19% đỉnh FP32 của card. Hãy đo phần khoảng
cách còn lại, và học một quy ước API khiến ai cũng vấp.

## Kiến thức nền

### Cái bẫy column-major

cuBLAS thừa hưởng quy ước của Fortran: phần tử `(i, j)` nằm ở `A[i + j*ld]`, không
phải `A[i*ld + j]`. Đưa cho nó ma trận row-major thì nó tính ra **ma trận chuyển vị
của thứ bạn muốn — âm thầm, không báo lỗi**.

Cách sửa không di chuyển dữ liệu nào cả:

```
C   = A * B        (row-major)
  cùng vùng byte với
C^T = B^T * A^T    (column-major)
```

Nên hãy truyền **B trước, A sau**, với `CUBLAS_OP_N` cho cả hai, và cuBLAS sẽ ghi
đúng ma trận `C` row-major mà bạn muốn. Chỉ thứ tự tham số thay đổi.

> Nếu kết quả của bạn trông giống ma trận chuyển vị của đáp án mong đợi, nghĩa là
> bạn đã đảo nhầm đúng một thứ.

### Khoảng cách còn lại gồm những gì

Đại khái theo thứ tự công sức để rút ngắn: register tile rộng hơn (8×8 mỗi thread,
không phải 4×1) · nạp `float4` vector hoá · double buffering · thêm một tầng dàn
shared memory · tinh chỉnh mọi kích thước tile theo từng kiến trúc · Tensor Core
nếu kiểu dữ liệu cho phép ([bài 03](../03-tensor-core-wmma/)).

**Đạt 80% cuBLAS bằng tay là kết quả xuất sắc và là dự án nhiều tuần. Đạt 100% là
công việc toàn thời gian của một người nào đó.** Biết được con số mới giúp bạn
quyết định có nên thử hay không.

## Nhiệm vụ của bạn

1. Mang kernel tiled + register từ Giai đoạn 3/04 sang.
2. Gọi `cublasSgemm` trên dữ liệu row-major bằng đẳng thức ở trên. Hãy suy luận ra
   `m`, `n`, `k` và ba giá trị `ld`, đừng thử hoán vị bừa.
3. So sánh theo GFLOP/s, theo phần trăm đỉnh FP32, và theo phần trăm so với cuBLAS.
   Dùng cuBLAS làm mốc kiểm tra tính đúng đắn — không có bản chuẩn CPU ở n = 2048.

## Build và chạy

```bash
cmake --build build --target p5_02_cublas_gemm -j
./build/bin/p5/p5_02_cublas_gemm
```

## Điểm cốt lõi

- **cuBLAS là column-major.** Mẹo hoán đổi không tốn gì và là cách chuẩn để dùng nó
  từ C++ row-major.
- **Hãy biết phần trăm của bạn so với thư viện**, không chỉ so với đỉnh. Nó cho biết
  làm thêm có đáng không.
- Thư viện thắng nhờ nhiều tầng của chính những ý tưởng bạn đã biết, được tinh chỉnh
  theo từng kiến trúc.
