<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Dự án: LLM inference engine

> **Giai đoạn 6 · Làm chủ** | ⭐⭐⭐⭐⭐ | ✅ module lõi đã kiểm chứng trên RTX 3060

## Bạn sẽ xây gì

Một engine phục vụ một transformer nhỏ: giải mã theo lô, KV cache phân trang,
attention được gộp kernel.

## Cái gì có sẵn

`main.cu` cài đặt phép softmax nằm ở trái tim của attention, **hai lần** — bản ba
lượt và bản một lượt kiểu **online** — cả hai đều được kiểm chứng với bản tham
chiếu độ chính xác kép, bao gồm cả tính chất thật sự quan trọng ở các bước sau: mọi
hàng đều có tổng bằng 1.

Nó minh hoạ hai điều:

**Trừ đi giá trị lớn nhất của hàng.** `exp(x)` tràn số khi `x > 88` với float, và
điểm attention thường xuyên vượt mức đó. Trừ đi giá trị lớn nhất về mặt toán học
không thay đổi gì, nhưng lại là ranh giới giữa chạy được và trả về NaN. Thử xoá nó đi.

**Mẹo tái tỉ lệ online.**

```
m_new = max(m, x)
s_new = s·exp(m − m_new) + exp(x − m_new)
```

cho phép bạn duy trì một softmax đúng trong khi chỉ nhìn mỗi hàng **một lần**. Đó
chính xác là thứ khiến FlashAttention khả thi: nó tính attention trên một chuỗi mà
ma trận điểm số của nó không bao giờ được lưu đầy đủ trong bộ nhớ.

## Vì sao chọn dự án này

**Dung lượng bộ nhớ, không phải số FLOP, mới là ràng buộc quyết định.** Mọi mốc
dưới đây đều xoay quanh việc di chuyển hoặc lưu trữ ít hơn — điều đó khiến nó là
bài toán thời sự nhất trong lộ trình, và cũng là nơi hiện trạng công nghệ thay đổi
nhanh nhất.

## Các mốc công việc

| # | Sản phẩm | Giới thiệu điều gì |
|---|---|---|
| 1 | `QKᵀ` dùng tiled GEMM của GĐ 3/04 | GEMM theo lô trên các head |
| 2 | Gộp scores → softmax → `×V` vào một kernel | Không bao giờ hiện thực hoá ma trận điểm số |
| 3 | KV cache | Giải mã chuyển thành nghẽn bộ nhớ, không phải nghẽn tính toán |
| 4 | Paged attention | Chuỗi dài ngắn khác nhau chia sẻ bộ nhớ mà không phân mảnh |
| 5 | Giải mã theo lô | Nhiều chuỗi, mỗi chuỗi một bước; bài toán lập lịch |
| 6 | Trọng số INT8 hoặc FP8 | Lượng tử hoá, và đo mất mát chất lượng một cách trung thực |

## Tiêu chí đánh giá

- **Đúng đắn:** logits khớp bản tham chiếu PyTorch tới 1e-3 với cùng bộ trọng số.
- **Hiệu năng:** số token/giây ở batch 1 và batch 32; báo cáo phần trăm *băng thông*
  đỉnh đạt được lúc giải mã (nó nghẽn bộ nhớ — hãy so với số liệu GĐ 3/01).
- **Bộ nhớ:** VRAM đỉnh theo độ dài chuỗi. Paged attention phải làm con số này gần
  tuyến tính thay vì bậc hai theo số chuỗi chạy đồng thời.
- **Chất lượng (mốc 6):** perplexity trước và sau lượng tử hoá. Một con số tăng tốc
  công bố mà không kèm chỉ số chất lượng thì không phải một kết quả.

## Cách làm

1. **Nạp trọng số thật từ sớm** — một GPT-2 nhỏ hoặc TinyLlama. Trọng số ngẫu nhiên
   che giấu những lỗi số học mà phân phối thật sẽ phơi bày.
2. **So với PyTorch theo từng lớp**, không phải đầu-cuối. Kết quả sai sau 12 block
   chẳng nói cho bạn biết block nào hỏng.
3. **Đo băng thông, không phải FLOP**, trong lúc giải mã. Giải mã batch 1 phải đọc
   toàn bộ ma trận trọng số chỉ để sinh ra một token.

## Tài liệu

- Dao và cộng sự, *FlashAttention* (2022) và *FlashAttention-2* (2023)
- Kwon và cộng sự, *Efficient Memory Management for LLM Serving with PagedAttention* (2023)
- [vLLM](https://github.com/vllm-project/vllm) — bản cài đặt tham chiếu
- [llm.c](https://github.com/karpathy/llm.c) — điểm khởi đầu đầu-cuối dễ đọc
