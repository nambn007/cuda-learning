<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 5 · Chuyên sâu

> **6–8 tuần · ⭐⭐⭐⭐⭐ · 14 bài tập** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Giai đoạn này để làm gì

Mọi thứ từ đầu đến giờ dạy bạn cách viết kernel. Giai đoạn này dạy bạn **khi nào
đừng viết**. Một hàm `SGEMM` của cuBLAS đã được tinh chỉnh có hàng năm trời công
sức tối ưu theo từng kiến trúc đằng sau; một kernel viết tay đạt 80% của nó đã là
kết quả xuất sắc, còn đạt 100% là công việc toàn thời gian.

Nên mục tiêu ở đây là khả năng phán đoán: biết hệ sinh thái cung cấp những gì,
biết gọi cho đúng, biết đo khoảng cách, và biết vài tình huống hiếm hoi mà tự viết
thật sự thắng — thường là **fusion (gộp kernel)**, khi một kernel tự viết tránh
được một vòng đi-về qua DRAM mà lời gọi thư viện không tránh được.

## Danh sách bài tập

| # | Bài tập | Độ khó | Ý tưởng cốt lõi |
|---|---|---|---|
| 01 | thrust-basics | ⭐⭐ | Thuật toán kiểu STL trên thiết bị |
| 02 | cub-primitives | ⭐⭐⭐ | Nguyên thuỷ mức block/device so với bản tự viết của bạn |
| 03 | cublas-gemm | ⭐⭐⭐ | Cái bẫy column-major; bản tiled matmul của bạn còn cách bao xa |
| 04 | cublas-batched | ⭐⭐⭐ | GEMM theo lô và theo lô có bước nhảy |
| 05 | curand-monte-carlo | ⭐⭐⭐ | Sinh số ngẫu nhiên trên thiết bị, và "ngẫu nhiên song song" nghĩa là gì |
| 06 | cufft-convolution | ⭐⭐⭐ | Tích chập bằng FFT |
| 07 | cusparse-spmv | ⭐⭐⭐⭐ | SpMV dạng CSR — bất quy tắc, nghẽn bộ nhớ, không tránh được |
| 08 | mixed-precision | ⭐⭐⭐⭐ | FP16/BF16/TF32, vector hoá `half2`, đánh đổi độ chính xác và tốc độ |
| 09 | tensor-core-wmma | ⭐⭐⭐⭐⭐ | API WMMA (cần sm_70+) |
| 10 | cutlass-gemm | ⭐⭐⭐⭐⭐ | GEMM dạng template, gộp epilogue *(phụ thuộc tuỳ chọn)* |
| 11 | pytorch-extension | ⭐⭐⭐⭐ | Toán tử CUDA tự viết gọi được từ Python *(phụ thuộc tuỳ chọn)* |
| 12 | nvrtc-jit | ⭐⭐⭐⭐ | Biên dịch lúc chạy và driver API |
| 13 | cuda-opengl-interop | ⭐⭐⭐⭐ | Trực quan hoá không sao chép *(phụ thuộc tuỳ chọn)* |
| 14 | kernel-fusion | ⭐⭐⭐⭐ | Ít lượt duyệt bộ nhớ hơn — phép tối ưu còn lại có đòn bẩy lớn nhất |

Các bài ghi *phụ thuộc tuỳ chọn* chỉ được build khi chạy:

```bash
./scripts/build.sh --optional
```

Bài 09 cần compute capability ≥ 7.0 và sẽ bỏ qua gọn gàng trên card cũ hơn.

## Build và chạy

```bash
./scripts/build.sh
ctest --test-dir build -L p5 --output-on-failure
```

## Checklist

- [ ] Tôi tìm đến Thrust/CUB trước khi tự viết một scan hay một phép sắp xếp
- [ ] Tôi biết cuBLAS là column-major và gọi đúng được trên dữ liệu row-major
- [ ] Tôi biết bản GEMM của mình đạt bao nhiêu phần trăm so với cuBLAS
- [ ] Tôi dùng được FP16 mà không âm thầm phá hỏng độ chính xác, và giải thích được vì sao
- [ ] Tôi đã chạy một kernel Tensor Core và đo mức tăng tốc so với FP32
- [ ] Tôi đưa được một kernel CUDA ra cho Python gọi
- [ ] Tôi nhận diện được một chuỗi kernel đáng gộp, và đo được lưu lượng tiết kiệm

## Cột mốc

Cho một bài toán mới, trong vài phút bạn nói được thư viện nào phủ nó, gọi đúng
thư viện đó, và định lượng được một kernel tự viết sẽ phải vượt qua ngưỡng nào.
