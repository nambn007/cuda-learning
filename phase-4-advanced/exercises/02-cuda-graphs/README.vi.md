<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 02 - CUDA graphs

> **Giai đoạn 4 · Nâng cao** | Độ khó: ⭐⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [01](../01-pinned-and-streams/README.vi.md) | **Cần GPU**
>
> ⚠️ **Chưa được kiểm chứng trên máy tham chiếu.** Mã build được và logic đã được rà
> soát, nhưng bài này chưa chạy trên RTX 3060, nên README này cố ý **không có số đo
> nào**. Hãy chạy và tự điền số của bạn.

## Mục tiêu

Đưa CPU ra khỏi vòng lặp trong cùng. Khi một chương trình chạy hàng chục kernel nhỏ
lặp lại hàng nghìn lần, chính driver — chứ không phải GPU — mới là nút thắt.

## Kiến thức nền

Mỗi lệnh launch kernel tốn của CPU vài micro-giây công việc phía driver: kiểm tra
tham số, ghi gói lệnh, gõ chuông báo. Con số đó vô hình bên cạnh một kernel 2 ms và
**áp đảo** bên cạnh một kernel 5 µs.

Suy luận học sâu, bộ giải lặp và bước mô phỏng vật lý đều có cùng hình dạng — hàng
chục kernel nhỏ, chạy theo cùng thứ tự, hàng nghìn lần. GPU rốt cuộc ngồi không
giữa các kernel, chờ CPU theo kịp.

Một **CUDA graph** ghi lại chuỗi đó một lần dưới dạng DAG các node kèm phụ thuộc,
rồi phát lại bằng **một** lệnh launch duy nhất. Driver kiểm tra công việc một lần,
lúc instantiate, thay vì mỗi lần chạy.

**Hai cách dựng graph:**

| | |
|---|---|
| **Stream capture** | đặt stream vào chế độ ghi, phát lệnh y như bình thường, kết thúc ghi. Gần như không phải sửa mã — lựa chọn thông thường. |
| **API tường minh** | `cudaGraphAddKernelNode` v.v. Nhiều việc hơn, nhưng bạn kiểm soát trực tiếp cấu trúc phụ thuộc. |

**Cái bẫy:** graph ghi lại **con trỏ và giá trị, không phải ý định**. Nếu vòng lặp
kế tiếp lẽ ra phải dùng buffer khác, việc phát lại cùng graph sẽ vui vẻ chạy trên
buffer *cũ* — âm thầm, không báo lỗi. Hoặc giữ con trỏ cố định (double-buffer vào
các ô cố định), hoặc cập nhật tham số node bằng
`cudaGraphExecKernelNodeSetParams`, vốn rẻ hơn nhiều so với instantiate lại.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. Mốc cơ sở: `kKernelsPerIteration` lệnh launch trong một stream, lặp
   `kIterations` lần. Kiểm chứng với `chainCPU()` trước đã.
2. Ghi lại đúng chuỗi đó bằng `cudaStreamBeginCapture` / `cudaStreamEndCapture` /
   `cudaGraphInstantiate`. In ra số node.
3. Đo cả hai và báo cáo **micro-giây trên mỗi lần launch**, không phải tổng
   mili-giây. Hãy dự đoán chi phí mỗi lần launch trước khi đo.
4. Tái hiện lỗi con trỏ: ghi graph trên một buffer, phát lại và kỳ vọng buffer thứ
   hai được cập nhật, quan sát thấy không có gì xảy ra và cũng không có lỗi. Rồi sửa
   bằng `cudaGraphExecKernelNodeSetParams`.

## Build và chạy

```bash
cmake --build build --target p4_02_cuda_graphs -j
./build/bin/p4/p4_02_cuda_graphs
nsys profile --stats=true ./build/bin/p4/p4_02_cuda_graphs
```

## Kết quả mong đợi

Chương trình in ra phần kiểm tra tính đúng đắn (cả hai biến thể phải khớp chuỗi
CPU), rồi một bảng so lệnh launch riêng lẻ với graph replay, cùng chi phí mỗi lần
launch tính bằng micro-giây.

**Hãy tự điền số của bạn vào đây** sau khi chạy. Điều cần chú ý: khối lượng công
việc trên GPU là như nhau ở cả hai, nên mọi chênh lệch đều là chi phí launch thuần
tuý phía CPU. Khoảng cách sẽ giãn ra khi bạn tăng `kKernelsPerIteration` và thu hẹp
khi bạn làm mỗi kernel lớn hơn.

## Điểm cốt lõi

- **Chi phí launch là chi phí có thật**, đo bằng micro-giây thời gian *CPU* cho mỗi
  kernel, và nó vô hình cho tới khi kernel của bạn trở nên nhỏ.
- **Graph chuyển việc kiểm tra từ mỗi lần launch sang một lần instantiate.**
- **Stream capture gần như không cần sửa mã** — cứ phát lệnh như thường giữa begin
  và end.
- **Graph đóng băng con trỏ.** Lỗi phổ biến nhất là phát lại với buffer cũ, và nó
  hỏng một cách âm thầm.
- Graph có lời với **nhiều kernel nhỏ trong một chuỗi lặp lại**; nó chẳng giúp gì
  cho vài kernel lớn.

## Đi xa hơn

- Quét `kKernelsPerIteration` từ 1 tới 100. Từ đâu thì graph bắt đầu thắng?
- Tăng số phần tử cho tới khi mỗi kernel mất hàng mili-giây. Lợi thế có biến mất không?
- Dựng cùng graph đó bằng API node tường minh rồi so sánh mã.
- Ghi một graph chứa cả memcpy và event chứ không chỉ kernel.
- So sánh timeline trong `nsys` — khoảng trống giữa các kernel chính là thứ graph xoá bỏ.
