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
| 01 ✅ | [thrust-and-cub](exercises/01-thrust-and-cub/README.vi.md) | ⭐⭐ | Hiệu chuẩn: bản reduction GĐ 3 của bạn ngang bằng CUB |
| 02 ⚠️ | [cublas-gemm](exercises/02-cublas-gemm/README.vi.md) | ⭐⭐⭐ | Cái bẫy column-major, và GEMM của bạn còn cách bao xa |
| 03 ✅ | [tensor-core-wmma](exercises/03-tensor-core-wmma/README.vi.md) | ⭐⭐⭐⭐⭐ | 8192 FLOP trong một lệnh — và cái giá về độ chính xác |

✅ đã kiểm chứng trên RTX 3060 · ⚠️ mã hoàn chỉnh, chưa chạy trên máy tham chiếu

### Tạm hoãn

Đã đặc tả trong [docs/ROADMAP.vi.md](../docs/ROADMAP.vi.md) nhưng chưa viết:
**cublas-batched**, **curand-monte-carlo**, **cufft-convolution**, **cusparse-spmv**,
**mixed-precision**, **cutlass-gemm**, **pytorch-extension**, **nvrtc-jit**,
**cuda-opengl-interop**, **kernel-fusion**.

Bài 03 cần compute capability ≥ 7.0 và sẽ bỏ qua gọn gàng trên card cũ hơn.

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
