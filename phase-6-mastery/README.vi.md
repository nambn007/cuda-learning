<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Giai đoạn 6 · Làm chủ

> **Liên tục · 🏆 · 5 dự án** — [Lộ trình](../docs/ROADMAP.vi.md) · [← Kho mã](../README.vi.md)

## Giai đoạn này để làm gì

Bài tập thì có đáp án sẵn. Dự án thì không. Giai đoạn này là nơi bạn biết được mình
có đưa nổi một codebase GPU từ một thư mục rỗng thành thứ dám đem cho nhà tuyển
dụng xem hay không.

Mỗi dự án đi kèm:

- một **đặc tả** — nó phải làm được gì, và thế nào là "xong"
- một **tài liệu kiến trúc** — bố cục dữ liệu, cách chia kernel, những quyết định
  thiết kế đáng cân nhắc sớm
- các **mốc công việc** — ba đến năm chặng, mỗi chặng chạy độc lập được, để lúc
  nào bạn cũng có một thứ hoạt động được
- **tiêu chí đánh giá** — những con số hiệu năng mà một bản cài đặt tốt phải đạt
- một **mô-đun lõi làm mẫu** — một kernel không tầm thường đã được cài đặt và giải
  thích, để bạn bắt đầu từ một nền móng chạy được thay vì một file trống

Phần còn lại là của bạn. Đó chính là mục đích.

## Các dự án

| Dự án | Độ khó | Bạn xây gì | Kỹ thuật chính |
|---|---|---|---|
| [gpu-ray-tracer](projects/gpu-ray-tracer/) | ⭐⭐⭐⭐ | Path tracer cho ra ảnh sạch nhiễu | Duyệt BVH, xử lý phân kỳ, RNG cho mỗi thread, khử nhiễu |
| [mini-dl-framework](projects/mini-dl-framework/) | ⭐⭐⭐⭐⭐ | Huấn luyện một mạng nhỏ từ đầu đến cuối | Autograd, kernel lan truyền ngược, GEMM, gộp toán tử, memory pool |
| [fluid-simulation](projects/fluid-simulation/) | ⭐⭐⭐⭐ | Mô phỏng chất lỏng tương tác thời gian thực | Stencil, giải red-black, băm không gian, interop OpenGL |
| [gpu-database](projects/gpu-database/) | ⭐⭐⭐⭐ | Truy vấn phân tích trên dữ liệu dạng cột | Nén dòng, phân hoạch radix, hash join, atomic |
| [llm-inference-engine](projects/llm-inference-engine/) | ⭐⭐⭐⭐⭐ | Phục vụ một transformer nhỏ | Phân trang KV-cache, kernel attention, gom lô, lượng tử hoá |

## Chọn dự án nào

Hãy chọn cái mà *kiểu thất bại* của nó là thứ bạn muốn học:

- **Ray tracer** — phân kỳ warp và truy cập bộ nhớ bất quy tắc. Mỗi tia đi một
  hướng khác nhau.
- **DL framework** — thiết kế API dưới áp lực hiệu năng, cộng thêm lan truyền ngược
  phải khớp chính xác với lan truyền xuôi.
- **Mô phỏng chất lỏng** — băng thông duy trì và tối ưu stencil, kèm kết quả trực
  quan khiến lỗi lộ ra ngay.
- **Cơ sở dữ liệu** — song song phụ thuộc dữ liệu. Bạn không biết trước kích thước
  đầu ra.
- **LLM engine** — dung lượng bộ nhớ là ràng buộc quyết định, và là hiện trạng
  công nghệ hôm nay.

Hai dự án hoàn chỉnh là mức chuẩn cho giai đoạn này. Một dự án làm xong hơn hẳn
năm dự án dở dang.

## Cách làm việc

1. **Làm cho đúng trước đã.** Viết bản tham chiếu CPU trước khi viết kernel; bạn sẽ
   cần nó để gỡ lỗi.
2. **Profile trước khi tối ưu.** Tiêu chí đánh giá của mỗi dự án được diễn đạt bằng
   chỉ số profiler là có lý do.
3. **Ghi nhật ký.** Bạn đã thử gì, đo được gì, kết luận gì. Đây mới là thứ bạn thật
   sự đem cho người khác xem — và là thứ biến một dự án thành bài blog hay bài nói
   hội thảo.
4. **So với bản tốt nhất hiện có.** cuBLAS, cuDNN, OptiX, vLLM. Biết mình đang ở
   mức 60% so với hiện trạng công nghệ vẫn hơn là không biết gì.

## Vượt ra ngoài các dự án

- Đóng góp cho một dự án mã nguồn mở của NVIDIA (CUTLASS, cuDF, RAPIDS)
- Viết lại một phép tối ưu bạn đã làm, kèm số liệu
- Trả lời câu hỏi trên diễn đàn NVIDIA Developer
- Theo dõi các phiên GTC và ghi chú phát hành CUDA; phần cứng vẫn đang chạy tiếp
- Học thêm một mô hình thứ hai — SYCL, HIP, hoặc Triton — để thấy ý tưởng nào của
  CUDA là nền tảng và ý tưởng nào chỉ thuộc về riêng hãng
