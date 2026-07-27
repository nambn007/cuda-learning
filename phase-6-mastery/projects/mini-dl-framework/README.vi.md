<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Dự án: Mini deep-learning framework

> **Giai đoạn 6 · Làm chủ** | ⭐⭐⭐⭐⭐ | ✅ module lõi đã kiểm chứng trên RTX 3060

## Bạn sẽ xây gì

Một framework huấn luyện được một mạng nhỏ từ đầu đến cuối trên GPU: tensor, layer,
autograd, optimizer.

## Cái gì có sẵn

`main.cu` là một **lớp linear chạy được** — lan truyền xuôi (`Y = XW + b`) và cả ba
gradient (`dX = dY Wᵀ`, `dW = Xᵀ dY`, `db = Σ dY`) — được đối chiếu với **vi phân
số**.

Phép kiểm tra đó là công cụ giá trị nhất của cả dự án. Một chỉ số bị chuyển vị hay
một phép tổng bị thiếu sẽ lộ ra trong vài giây; không có nó, việc huấn luyện chỉ
hội tụ kém hơn một chút và không bao giờ nói cho bạn biết vì sao.

> **Quy tắc cho cả dự án: mọi lớp mới đều phải qua kiểm tra gradient bằng vi phân số
> trước khi được dùng để huấn luyện.** Không có công cụ phát hiện lỗi nào rẻ hơn
> trong học máy.

## Vì sao chọn dự án này

**Thiết kế API dưới áp lực hiệu năng**, cộng với lan truyền ngược phải là đạo hàm
*chính xác* của lan truyền xuôi. Làm đúng phần toán mới là một nửa; giữ cho lớp
trừu tượng không ngốn của bạn 3 lần tốc độ là nửa còn lại.

## Các mốc công việc

| # | Sản phẩm | Giới thiệu điều gì |
|---|---|---|
| 1 | `Tensor` với RAII, shape, view | Áp dụng GĐ 2/12 ở quy mô lớn |
| 2 | Thay GEMM naive bằng bản GĐ 3/04, rồi bằng cuBLAS | Đo cả hai — giờ bạn đã biết khoảng cách |
| 3 | ReLU, softmax + cross-entropy, mỗi cái đều kiểm tra gradient | Gộp elementwise, softmax ổn định số học |
| 4 | Một băng ghi autograd | `backward()` được sinh ra, không phải viết tay |
| 5 | SGD và Adam; huấn luyện MNIST đạt >97% | Cả hệ thống thật sự chạy |
| 6 | Gộp chuỗi elementwise; một memory pool | GĐ 1/03 và phần fusion của GĐ 5, dùng thật |

## Tiêu chí đánh giá

- **Đúng đắn:** mọi lớp đều qua kiểm tra gradient số với sai số tương đối 1e-2.
- **Hội tụ:** MNIST đạt trên 97% độ chính xác trên tập kiểm tra.
- **Hiệu năng:** báo cáo thời gian mỗi epoch và phần trăm GEMM của bạn so với
  cuBLAS. Nêu rõ lớp trừu tượng tốn bao nhiêu so với gọi thẳng kernel.

## Cách làm

1. **Kiểm tra gradient trước, tối ưu sau.** Một gradient sai nhưng nhanh sẽ huấn
   luyện tới một cực trị tệ hơn mà trông vẫn ổn.
2. **Giữ lại các kernel naive.** Chúng là bản tham chiếu khi các kernel nhanh bất đồng.
3. **Profile cả một epoch**, không phải một kernel. Chi phí framework nằm ở khoảng
   giữa các kernel, và `nsys` là nơi bạn nhìn thấy nó.

## Tài liệu

- Karpathy, [micrograd](https://github.com/karpathy/micrograd) — autograd trong 100 dòng
- *Deep Learning* (Goodfellow và cộng sự), chương 6 — dẫn xuất lan truyền ngược
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — một
  framework thật được tổ chức ra sao
- Micikevicius và cộng sự, *Mixed Precision Training* (2018) — cho khi bạn thêm FP16
