<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 01 - Bộ nhớ ghim và chồng lấn bằng stream

> **Giai đoạn 4 · Nâng cao** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [GĐ 2/03](../../../phase-2-cuda-fundamentals/exercises/03-vector-add/README.vi.md) | **Cần GPU** | ✅ đã kiểm chứng trên RTX 3060

## Mục tiêu

Giai đoạn 2/03 kết thúc không vui: với công việc đơn giản theo từng phần tử thì GPU
*thua*, vì vòng đi-về qua PCIe tốn hơn phần tính toán tiết kiệm được. Đây là câu
trả lời thực sự đầu tiên cho chuyện đó.

## Kiến thức nền

**Bộ nhớ host ghim (page-locked).** Vùng cấp phát thông thường có thể bị swap ra,
nên driver không thể đưa địa chỉ của nó cho DMA engine — nó phải trung chuyển qua
một buffer ghim nội bộ trước. `cudaMallocHost` cho bạn vùng nhớ mà hệ điều hành
không được di chuyển, và DMA engine đọc thẳng từ đó.

Nó cũng là **điều kiện bắt buộc để bất đồng bộ**: `cudaMemcpyAsync` trên bộ nhớ
*pageable* âm thầm hành xử đồng bộ. Một "pipeline" dựng trên bộ nhớ pageable không
chồng lấn được gì cả — mà vẫn trông như đúng.

**Stream.** Stream là hàng đợi công việc GPU có thứ tự; các stream khác nhau có thể
chạy đồng thời. GPU có các copy engine riêng cho từng chiều cộng với các SM, nên
với đủ stream thì H2D, tính toán và D2H có thể cùng bay một lúc:

```
tuần tự    =  H2D + kernel + D2H
chồng lấn  →  max(H2D, kernel, D2H) + độ trễ của một khối
```

**Cái bẫy.** Default stream đồng bộ với mọi stream khác. Một lệnh launch hay copy
quên tham số stream sẽ làm sập toàn bộ timeline — mà kết quả vẫn đúng, nên không
test nào bắt được.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. Cấp phát bộ nhớ ghim, so băng thông H2D của nó với một `std::vector`.
2. Dựng pipeline tuần tự.
3. Dựng pipeline chia khối trên `kStreams` stream. **Mọi** thao tác đều phải ghi rõ
   stream. Nhớ zero buffer đầu ra trước khi kiểm chứng, để một pipeline âm thầm
   không làm gì không thể pass nhờ dùng lại kết quả cũ.
4. Tái hiện bẫy default-stream bằng cách xoá tham số stream *chỉ ở lệnh launch
   kernel*, rồi giải thích vì sao mức phạt đo được lại có độ lớn như vậy.
5. Xác nhận chồng lấn bằng `nsys profile --stats=true`. Kết quả đúng không chứng
   minh được điều gì về tính đồng thời.

## Build và chạy

```bash
cmake --build build --target p4_01_pinned_and_streams -j
./build/bin/p4/p4_01_pinned_and_streams
nsys profile --stats=true ./build/bin/p4/p4_01_pinned_and_streams
```

## Kết quả mong đợi

RTX 3060, 256 MB mỗi buffer, 2 copy engine:

```
--- Transfer bandwidth ---
pageable host -> device              33.248      8.07 GB/s     1.00x
pinned host -> device                21.711     12.36 GB/s     1.53x

--- Summary ---
sequential                                    44.228 ms     1.00x
overlapped, 4 streams                         26.939 ms     1.64x
overlapped, but one default-stream call       29.158 ms     1.52x

  kernel alone     :  1.79 ms
  sequential total : 44.23 ms
  overlapped total : 26.94 ms  (1.64x)
```

### Hãy đọc kỹ dòng default-stream

Cái bug này chỉ tốn **8%** ở đây — điều đó *không* có nghĩa nó là bug nhẹ. Kernel
chỉ chiếm 1.79 ms trên 44 ms, nên phần truyền dữ liệu áp đảo, và H2D vẫn chồng lấn
được với D2H ngay cả khi kernel thì không. Hãy làm cho kernel thành phần đắt đỏ và
đúng cái bug đó sẽ lấy đi của bạn mọi thứ.

> **Mức phạt đo được nhỏ không có nghĩa là bug nhỏ.** Nó chỉ có nghĩa là khối lượng
> công việc cụ thể này không nhạy với nó.

## Điểm cốt lõi

- **Bộ nhớ ghim nhanh hơn ~1.5 lần và là *điều kiện bắt buộc* cho sao chép bất đồng
  bộ**, không phải một phép tối ưu. `cudaMemcpyAsync` trên bộ nhớ pageable là đồng
  bộ một cách âm thầm.
- **Chồng lấn biến một phép cộng thành phép lấy cực đại.** 1.64 lần ở đây, và nó
  tăng theo mức cân bằng giữa ba giai đoạn.
- **Default stream là một rào chắn.** Một tham số stream bị thiếu làm mọi thứ tuần
  tự hoá, một cách âm thầm.
- **Hãy kiểm chứng tính đồng thời trên timeline.** Kết quả đúng chẳng nói gì về
  việc có thứ gì chồng lấn hay không.

## Đi xa hơn

- Quét `kChunks` = 2, 4, 8, 16, 64. Quá ít thì không có gì để chồng lấn; quá nhiều
  thì chi phí launch từng lệnh áp đảo.
- Làm kernel nặng gấp 10 lần rồi chạy lại. Cả mức tăng tốc lẫn mức phạt
  default-stream thay đổi thế nào?
- Thử `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` — nó thay đổi gì với
  cái bẫy kia?
- Ghim thật nhiều bộ nhớ (vài GB) rồi xem máy chậm đi. Trang bị ghim là tài nguyên
  của cả máy.
