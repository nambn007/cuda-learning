<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# Dự án: GPU ray tracer

> **Giai đoạn 6 · Làm chủ** | ⭐⭐⭐⭐ | ✅ module lõi đã kiểm chứng trên RTX 3060

## Bạn sẽ xây gì

Một path tracer dựng được ảnh sạch nhiễu của một cảnh không tầm thường, đủ nhanh để
lặp đi lặp lại khi chỉnh sửa.

## Cái gì có sẵn

`main.cu` là **module lõi chạy được**, không phải khung rỗng: hình cầu, RNG riêng
cho từng thread, phản xạ khuếch tán lấy mẫu theo cosine, mã hoá gamma, xuất PPM, và
các kiểm tra theo tính chất (không đồng đều, không đen, deterministic). Nó dựng ảnh
800×450 ở 64 spp trong ~3 ms trên RTX 3060 — 7.4 tỷ tia sơ cấp mỗi giây.

Mọi thứ sau đó là của bạn.

## Vì sao chọn dự án này

**Phân kỳ.** Các điểm ảnh kề nhau nảy theo hướng khác nhau và kết thúc sau số lần
nảy khác nhau, nên các thread trong một warp làm những việc thật sự khác nhau. Đây
là bài toán phân kỳ khó nhất trong lộ trình — mọi bài trước đều có sẵn một mẫu truy
cập đều đặn nếu bạn chịu tìm. Ở đây thì không có.

## Các mốc công việc

| # | Sản phẩm | Giới thiệu điều gì |
|---|---|---|
| 1 | Lưới tam giác, đọc file `.obj` | Dữ liệu bất quy tắc, gián tiếp |
| 2 | Dựng BVH (host) + duyệt (device) | Ngăn xếp trong thanh ghi, bài toán phân kỳ đầy đủ |
| 3 | Vật liệu: kim loại, điện môi, phát sáng | Shading nhiều rẽ nhánh, và cái giá của nó |
| 4 | Kết thúc kiểu Russian roulette, lấy mẫu quan trọng | Giảm phương sai — ít mẫu hơn cho cùng chất lượng |
| 5 | Tái cấu trúc theo wavefront | Sắp xếp tia theo vật liệu để lấy lại tính đồng nhất |
| 6 | Khử nhiễu (À-Trous hoặc tương tự) | 16 spp mà trông như 1024 |

## Tiêu chí đánh giá

- **Đúng đắn:** hộp Cornell hội tụ về ảnh tham chiếu trong sai số vài phần trăm.
- **Hiệu năng:** báo cáo Mrays/s, cùng chỉ số warp execution efficiency của `ncu`
  trước và sau mốc 5. Nếu bạn không nói được cái giá của phân kỳ bằng một con số,
  nghĩa là bạn chưa đo.
- **Chất lượng:** ảnh so sánh cạnh nhau ở 16 / 64 / 1024 spp, có và không khử nhiễu.

## Cách làm

1. **Giữ một bản render tham chiếu trên CPU.** Chậm, hiển nhiên đúng, và là cách
   duy nhất để phân biệt lỗi dựng hình với lỗi lấy mẫu.
2. **Nhìn ảnh sau mỗi thay đổi.** Kiểm tra theo tính chất bắt được crash; mắt bạn
   bắt được cái sai.
3. **Profile trước khi tái cấu trúc.** Mốc 5 là một lần viết lại lớn — chỉ làm sau
   khi đã đo được phân kỳ thật sự tốn của bạn bao nhiêu.

## Tài liệu

- *Ray Tracing in One Weekend* (Shirley) — bắt đầu từ đây nếu phần toán còn lạ
- *Physically Based Rendering* (Pharr, Jakob, Humphreys) — sách tham chiếu
- Laine và cộng sự, *Megakernels Considered Harmful* (2013) — lý lẽ cho mốc 5
- [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) — thứ bạn đang
  cạnh tranh, và điều mà RT core làm được còn bạn thì không
