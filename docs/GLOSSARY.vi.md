<!-- Ngôn ngữ: [English](GLOSSARY.md) | **Tiếng Việt** -->

# Thuật ngữ

[← quay lại kho mã](../README.vi.md)

Cột tiếng Việt là thuật ngữ được dùng trong các tài liệu `.vi.md`. Thuật ngữ tiếng
Anh là thứ bạn sẽ gặp trong tài liệu NVIDIA, nên **hãy học khái niệm bằng ngôn ngữ
nào bạn thấy dễ, nhưng nhớ giữ lấy tên tiếng Anh** — đó là từ khoá bạn sẽ tra cứu.

---

## Mô hình thực thi

| English | Tiếng Việt | Ý nghĩa |
|---|---|---|
| Thread | Thread / luồng | Đơn vị thực thi nhỏ nhất. Có thanh ghi và con trỏ lệnh riêng. |
| Warp | Warp | 32 thread cùng phát ra một lệnh. Đơn vị lập lịch thực sự. |
| Thread block | Block / khối thread | Nhóm thread chạy trên một SM, dùng chung được `__shared__` và đồng bộ được với nhau. |
| Grid | Grid / lưới | Toàn bộ các block do một lệnh launch tạo ra. |
| Kernel | Kernel | Hàm `__global__` chạy trên GPU. |
| Launch | Launch / khởi chạy | `kernel<<<grid, block>>>(args)`. Bất đồng bộ. |
| SM (Streaming Multiprocessor) | SM / bộ đa xử lý dòng | Nhân phần cứng mà một block chạy trên đó. Một GPU có vài chục cái. |
| SIMT | SIMT | Single Instruction, Multiple Threads — biến thể SIMD của NVIDIA. |
| Warp divergence | Phân kỳ warp | Các thread trong warp rẽ nhánh khác nhau; cả hai nhánh chạy tuần tự. |
| Occupancy | Occupancy / độ chiếm dụng | Số warp thường trú mỗi SM ÷ mức tối đa. Nhiều warp che được nhiều độ trễ hơn. |
| Grid-stride loop | Vòng lặp grid-stride | Mỗi thread duyệt mảng theo bước bằng tổng số thread. |

## Bộ nhớ

| English | Tiếng Việt | Ý nghĩa |
|---|---|---|
| Global memory | Bộ nhớ toàn cục | DRAM của thiết bị. Lớn, chậm, mọi thread đều thấy. |
| Shared memory | Shared memory / bộ nhớ chia sẻ | Vùng nhớ nháp trên chip, riêng cho mỗi block. Là cache do bạn tự quản lý. |
| Register | Thanh ghi | Nơi lưu trữ nhanh nhất, riêng cho từng thread. Rất khan hiếm. |
| Local memory | Bộ nhớ cục bộ | Vùng tràn riêng cho mỗi thread nhưng thực chất nằm trong DRAM. Chậm. |
| Constant memory | Bộ nhớ hằng | Vùng chỉ đọc 64 KB có cache quảng bá. |
| Texture memory | Bộ nhớ texture | Đường chỉ đọc có nội suy phần cứng và tối ưu cho tính cục bộ 2D. |
| Unified / managed memory | Bộ nhớ hợp nhất | Một con trỏ dùng được ở cả host lẫn device; trang được di trú khi cần. |
| Pinned (page-locked) memory | Bộ nhớ ghim | Bộ nhớ host mà hệ điều hành không được swap, nên DMA đọc trực tiếp được. |
| Coalescing | Coalescing / gộp truy cập | Gộp các địa chỉ của một warp thành ít giao dịch nhất có thể. |
| Bank conflict | Xung đột bank | Hai thread trong warp chạm vào hai hàng khác nhau của cùng một bank shared memory; bị tuần tự hoá. |
| Cache line / sector | Cache line / sector | Khối kích thước cố định mà bộ nhớ thực sự di chuyển theo. |
| Bandwidth | Băng thông | Số byte trên giây. |
| Latency | Độ trễ | Thời gian tới khi một yêu cầu trả về. Được che bằng song song, chứ không giảm đi. |

## Hiệu năng

| English | Tiếng Việt | Ý nghĩa |
|---|---|---|
| Throughput | Thông lượng | Khối lượng công việc hoàn thành trên một đơn vị thời gian. |
| Arithmetic intensity | Cường độ số học | Số FLOP trên mỗi byte di chuyển. Trục hoành của roofline. |
| Roofline model | Mô hình roofline | `min(đỉnh tính toán, cường độ × băng thông đỉnh)`. |
| Ridge point | Điểm gãy | Nơi hai mái gặp nhau: cường độ mà tại đó kernel thôi bị nghẽn bộ nhớ. |
| Memory bound | Nghẽn bộ nhớ | Bị giới hạn bởi việc di chuyển dữ liệu. Sửa bằng cách chuyển ít byte hơn. |
| Compute bound | Nghẽn tính toán | Bị giới hạn bởi phép tính. Sửa bằng lệnh tốt hơn. |
| FLOP / GFLOP/s | FLOP / GFLOP/s | Phép toán dấu phẩy động; tỉ phép toán mỗi giây. |
| Effective bandwidth | Băng thông hữu hiệu | Số byte hữu ích ÷ thời gian — khác với số byte mà bus đã chuyển. |
| Speedup | Mức tăng tốc | Thời gian mốc cơ sở ÷ thời gian sau tối ưu. |
| Register spilling | Tràn thanh ghi | Quá nhiều giá trị đang sống, trình biên dịch phải cất chúng vào local memory. |
| ILP | Song song mức lệnh | Nhiều lệnh độc lập cùng chạy; thuốc chữa cho độ trễ. |

## Các mẫu song song

| English | Tiếng Việt | Ý nghĩa |
|---|---|---|
| Reduction | Reduction / rút gọn | Từ nhiều giá trị về một (tổng, cực đại, ...). |
| Scan / prefix sum | Scan / tổng tiền tố | Tổng luỹ tiến dọc theo mảng. |
| Stream compaction | Nén dòng | Chỉ giữ lại các phần tử thoả điều kiện. |
| Stencil | Stencil | Mỗi giá trị đầu ra đọc một vùng lân cận cố định của đầu vào. |
| Tiling / blocking | Tiling / chia khối | Xử lý từng khối vừa cache hoặc vừa shared memory để tạo ra tái sử dụng. |
| Halo | Halo / vành biên | Phần viền phụ mà một tile phải nạp thêm cho stencil. |
| Privatisation | Riêng tư hoá | Tạo bản sao riêng cho mỗi block để biến tranh chấp toàn cục thành tranh chấp cục bộ. |
| Kernel fusion | Gộp kernel | Ghép các kernel để tránh một vòng đi-về qua DRAM. |
| Atomic operation | Phép toán nguyên tử | Đọc-sửa-ghi mà không thread nào chen ngang được. |

## Công cụ và API

| English | Tiếng Việt | Ý nghĩa |
|---|---|---|
| Compute capability | Compute capability | Mức tính năng/tập lệnh của GPU, ví dụ `sm_86`. Không phải tốc độ. |
| PTX | PTX | Hợp ngữ ảo khả chuyển, được driver biên dịch JIT. |
| SASS | SASS | Mã máy thật cho một kiến trúc cụ thể. |
| Stream | Stream | Hàng đợi công việc GPU có thứ tự. Các stream khác nhau có thể chồng lấn. |
| Event | Event | Dấu mốc trong một stream, dùng để đo thời gian và tạo phụ thuộc giữa các stream. |
| CUDA Graph | CUDA Graph | Đồ thị thao tác đã ghi lại, phát lại gần như không tốn chi phí launch. |
| Cooperative groups | Cooperative groups | API đồng bộ các tập con thread, lên tới toàn bộ grid. |
| Dynamic parallelism | Song song động | Một kernel khởi chạy một kernel khác. |
| Peer access (P2P) | Truy cập ngang hàng | Một GPU đọc thẳng bộ nhớ của GPU khác. |
| Nsight Systems | Nsight Systems | Profiler timeline toàn ứng dụng (`nsys`). |
| Nsight Compute | Nsight Compute | Profiler chuyên sâu từng kernel (`ncu`). |
| compute-sanitizer | compute-sanitizer | Phát hiện truy cập bộ nhớ sai, race và lỗi đồng bộ. |

## Thư viện

| Tên | Công dụng |
|---|---|
| cuBLAS | Đại số tuyến tính dày đặc (GEMM và họ hàng) |
| cuDNN | Nguyên thuỷ học sâu (tích chập, pooling, chuẩn hoá) |
| cuFFT | Biến đổi Fourier nhanh |
| cuRAND | Sinh số ngẫu nhiên |
| cuSPARSE | Ma trận thưa |
| cuSOLVER | Giải hệ tuyến tính và bài toán trị riêng |
| Thrust | Thuật toán kiểu STL chạy trên thiết bị |
| CUB | Nguyên thuỷ mức block và mức device, lớp nằm dưới Thrust |
| CUTLASS | Khối xây dựng GEMM dạng template, tinh chỉnh được |
| NCCL | Truyền thông tập thể giữa nhiều GPU |
| TensorRT | Tối ưu và triển khai suy luận |

---

## Quy ước đặt tên trong kho mã này

| Ký hiệu | Ý nghĩa |
|---|---|
| `d_x` | con trỏ tới bộ nhớ **thiết bị** (device) |
| `h_x` | con trỏ tới bộ nhớ **máy chủ** (host) |
| `p2_03_vector_add` | giai đoạn 2, bài 03 — bản **khởi đầu** |
| `p2_03_vector_add_sol` | cùng bài đó — **lời giải tham chiếu** |
| `TODO` | phần bạn cần tự viết |
| `[PASS]` / `[FAIL]` / `[SKIP]` | kết quả kiểm chứng; `[SKIP]` không phải lỗi |
