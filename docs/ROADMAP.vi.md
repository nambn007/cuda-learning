<!-- Ngôn ngữ: [English](ROADMAP.md) | **Tiếng Việt** -->

# Lộ trình

> Từ số 0 đến chuyên gia GPU trong 6–18 tháng. 63 bài tập, 5 dự án, và một luận
> điểm xuyên suốt: **bộ nhớ mới là nút thắt, và bạn phải đo mới biết.**

[← quay lại kho mã](../README.vi.md)

---

## Lộ trình được xây thế nào

Mọi bài tập đều theo cùng một khuôn — **đọc lý thuyết, điền các `TODO`, kiểm chứng
với bản tham chiếu CPU, đo đạc, so sánh.** Không có khẳng định nào mà chương trình
không tự chứng minh.

Ba mạch ý tưởng chạy suốt lộ trình, và đáng được gọi tên vì nhận ra chúng chính là
phần lớn việc học:

1. **Khối, chứ không phải byte.** Phần cứng chuyển dữ liệu theo đơn vị kích thước
   cố định. Cache line của CPU (GĐ 1/05) chính là memory sector của GPU (GĐ 3/01).
   Dùng một phần khối là phí phần còn lại.
2. **Tái sử dụng thắng băng thông.** Cache blocking (GĐ 1/02) *chính là*
   shared-memory tiling (GĐ 3/04) *chính là* kernel fusion (GĐ 5/14).
3. **Độ trễ được che bằng song song.** Nhiều biến tích luỹ trên CPU (GĐ 1/06–07)
   trở thành nhiều warp thường trú trên GPU (GĐ 3/15).

Thời gian giả định ~20 giờ/tuần. Chia đôi nếu bạn vốn đã viết C++ hệ thống.

---

## Giai đoạn 1 · Nền tảng

**3–4 tuần · ⭐ · [thư mục](../phase-1-foundation/README.vi.md)**

Bạn không thể tối ưu một kernel GPU nếu chưa giải thích được vì sao một vòng lặp
CPU lại chậm. Mọi khái niệm ở đây đều xuất hiện lại trên GPU dưới một cái tên khác.

| # | Bài tập | Ý tưởng cốt lõi |
|---|---|---|
| 01 | [matmul-naive-cpu](../phase-1-foundation/exercises/01-matmul-naive-cpu/README.vi.md) | Bố cục row-major, đếm FLOP, benchmark trung thực (warmup + trung vị) |
| 02 | [matmul-cache-blocking](../phase-1-foundation/exercises/02-matmul-cache-blocking/README.vi.md) | Thứ tự vòng lặp đáng giá 10 lần; blocking chỉ có lời khi cache bó tay |
| 03 | [memory-pool-allocator](../phase-1-foundation/exercises/03-memory-pool-allocator/README.vi.md) | Arena và pool allocator — vì sao phải tránh `cudaMalloc` trong đường nóng |
| 04 | [modern-cpp-toolkit](../phase-1-foundation/exercises/04-modern-cpp-toolkit/README.vi.md) | RAII, move semantics, template, lambda — bộ đồ nghề cho lớp device buffer |
| 05 | [cache-and-bandwidth](../phase-1-foundation/exercises/05-cache-and-bandwidth/README.vi.md) | Tự đo cache line, các cấp cache, băng thông DRAM của máy bạn |
| 06 | [simd-intrinsics](../phase-1-foundation/exercises/06-simd-intrinsics/README.vi.md) | Lane AVX2 ≈ lane warp; SIMD chỉ giúp mã nghẽn tính toán |
| 07 | [roofline-model](../phase-1-foundation/exercises/07-roofline-model/README.vi.md) | Tự dựng roofline; tìm vách đá thanh ghi |
| 08 | [gpu-device-query](../phase-1-foundation/exercises/08-gpu-device-query/README.vi.md) | Đỉnh lý thuyết, điểm gãy và ngân sách occupancy của GPU bạn |

**Cột mốc.** Bạn phát biểu được, cho chính máy mình: kích thước cache line, băng
thông DRAM duy trì được, điểm gãy đơn nhân, và — cho GPU của bạn — băng thông
đỉnh, FP32 đỉnh, điểm gãy và ngân sách thanh ghi mỗi thread.

---

## Giai đoạn 2 · CUDA cơ bản

**4–6 tuần · ⭐⭐ · [thư mục](../phase-2-cuda-fundamentals/README.vi.md)**

Viết kernel đúng và hiểu một lệnh launch thật sự tốn gì. Tối ưu tính sau; đúng đắn
và đo đạc trung thực đến trước.

| # | Bài tập | Ý tưởng cốt lõi |
|---|---|---|
| 01 | [hello-cuda](../phase-2-cuda-fundamentals/exercises/01-hello-cuda/README.vi.md) | `__global__`, cú pháp launch, chỉ số toàn cục, kiểm tra biên, lỗi bất đồng bộ |
| 02 | [thread-indexing](../phase-2-cuda-fundamentals/exercises/02-thread-indexing/README.vi.md) | Grid 1D/2D/3D, và grid-stride loop |
| 03 | vector-add | Trọn chu trình `cudaMalloc`/`Memcpy`/`Free` — và vì sao PCIe thường thắng |
| 04 | saxpy | Băng thông hữu hiệu là thước đo cho kernel nghẽn bộ nhớ |
| 05 | error-handling | Lỗi dính (sticky), lỗi đồng bộ vs bất đồng bộ, `compute-sanitizer` |
| 06 | matrix-add-2d | Launch 2D trên dữ liệu 2D thật |
| 07 | matmul-naive | Mốc cơ sở GPU mà mọi bản matmul sau này sẽ so với |
| 08 | transpose-naive | Một kernel *đúng* nhưng *chậm* — dọn đường cho Giai đoạn 3 |
| 09 | rgb-to-grayscale | Dữ liệu ảnh, đọc/ghi PPM, song song theo từng điểm ảnh |
| 10 | box-blur | Stencil, vùng biên (halo), xử lý biên |
| 11 | unified-memory | `cudaMallocManaged`, di trú trang, `cudaMemPrefetchAsync` |
| 12 | device-buffer-class | RAII bọc quanh bộ nhớ thiết bị — áp dụng GĐ 1/04 |

**Cột mốc.** Bạn viết được kernel đúng cho một bài toán mới, quản lý bộ nhớ thiết
bị không rò rỉ, báo cáo được băng thông hữu hiệu, và giải thích được vì sao phép
cộng vector lại *chậm hơn* trên GPU khi tính cả thời gian truyền dữ liệu.

---

## Giai đoạn 3 · Trung cấp

**6–8 tuần · ⭐⭐⭐ · [thư mục](../phase-3-intermediate/README.vi.md)**

Trái tim của lộ trình. Đây là nơi "nó chạy được" biến thành "nó chạy nhanh", và là
nơi công sức bỏ ra ở Giai đoạn 1 được đền đáp.

| # | Bài tập | Ý tưởng cốt lõi |
|---|---|---|
| 01 | memory-coalescing | Yếu tố hiệu năng lớn nhất của GPU, được đo tận tay |
| 02 | shared-memory-basics | `__shared__`, `__syncthreads__`, hợp tác trong phạm vi block |
| 03 | bank-conflicts | 32 bank, vì sao đệm thêm một phần tử sửa được mức chậm 32 lần |
| 04 | tiled-matmul | Blocking của GĐ 1/02, đặt trong shared memory — ở đây nó đáng giá 5–10 lần |
| 05 | transpose-optimized | Đọc *và* ghi đều coalesced nhờ một tile chia sẻ |
| 06 | reduction-variants | Sáu kernel, cái sau nhanh hơn cái trước — bài nghiên cứu kinh điển |
| 07 | warp-shuffle | `__shfl_down_sync`, ballot, vote — dùng thanh ghi thay shared memory |
| 08 | atomics | `atomicAdd`, `atomicCAS`, tranh chấp, atomic float tự viết |
| 09 | histogram | Riêng tư hoá: biến tranh chấp toàn cục thành tranh chấp trong shared memory |
| 10 | scan-hillis-steele | Quét tiền tố bao gồm, trong phạm vi một block |
| 11 | scan-blelloch | Quét hiệu quả công việc, độ dài tuỳ ý, nhiều block |
| 12 | stream-compaction | Scan + scatter, xương sống của việc lọc dữ liệu trên GPU |
| 13 | conv-1d-constant | Bộ nhớ `__constant__` và cache quảng bá của nó |
| 14 | conv-2d-shared | Stencil 2D với tile chia sẻ có vành halo |
| 15 | occupancy-tuning | Thanh ghi vs shared memory vs kích thước block; khi occupancy cao lại hại |
| 16 | profiling-nsight | Nsight Systems và Nsight Compute trên chính kernel của bạn |

**Cột mốc.** Cho một kernel chậm, bạn profile được nó, gọi tên được yếu tố giới
hạn từ các chỉ số, áp dụng đúng cách sửa, và chứng minh được mức cải thiện.

---

## Giai đoạn 4 · Nâng cao

**8–10 tuần · ⭐⭐⭐⭐ · [thư mục](../phase-4-advanced/README.vi.md)**

Vượt ra ngoài một kernel: chồng lấn công việc, mở rộng quy mô, và đọc được thứ mà
trình biên dịch thật sự sinh ra.

| # | Bài tập | Ý tưởng cốt lõi |
|---|---|---|
| 01 | pinned-memory | Băng thông truyền: bộ nhớ pageable vs pinned |
| 02 | streams-basics | Chạy đồng thời, và cái bẫy của default stream |
| 03 | streams-pipeline | Chồng lấn H2D → kernel → D2H, chia theo khối |
| 04 | events-and-sync | `cudaEvent`, `cudaStreamWaitEvent`, tự dựng đồ thị phụ thuộc |
| 05 | cuda-graphs | Ghi lại một pipeline; xoá bỏ chi phí launch từng lệnh |
| 06 | cooperative-groups | Phân hoạch theo tile, đồng bộ toàn grid |
| 07 | dynamic-parallelism | Kernel gọi kernel (ngữ nghĩa CDP2, CUDA 12+) |
| 08 | multi-gpu-basics | Liệt kê thiết bị, truy cập ngang hàng, sao chép P2P |
| 09 | multi-gpu-matmul | Chia việc trên nhiều thiết bị |
| 10 | lock-free-queue | `atomicCAS`, `__threadfence`, trật tự bộ nhớ trên GPU |
| 11 | register-pressure | `__launch_bounds__`, tràn thanh ghi, đánh đổi với occupancy |
| 12 | ptx-and-sass | `cuobjdump`, `nvdisasm`, PTX nội tuyến |
| 13 | persistent-kernel | Megakernel và mô hình producer/consumer trên thiết bị |

Bài 08 và 09 tự phát hiện máy chỉ có một GPU và bỏ qua một cách gọn gàng.

**Cột mốc.** Bạn chồng lấn được truyền dữ liệu với tính toán, giữ nhiều GPU cùng
bận, đọc SASS để giải thích một điểm nghẽn, và lập luận được về trật tự bộ nhớ
giữa các thread.

---

## Giai đoạn 5 · Chuyên sâu

**6–8 tuần · ⭐⭐⭐⭐⭐ · [thư mục](../phase-5-expert/README.vi.md)**

Thôi tự viết mọi thứ: hãy biết hệ sinh thái, và biết khi nào một thư viện sẽ thắng
bạn (thường là vậy).

| # | Bài tập | Ý tưởng cốt lõi |
|---|---|---|
| 01 | thrust-basics | Thuật toán kiểu STL chạy trên thiết bị |
| 02 | cub-primitives | Khối xây dựng mức block và mức device so với bản tự viết của bạn |
| 03 | cublas-gemm | Cái bẫy column-major; bản tiled matmul của bạn còn cách bao xa |
| 04 | cublas-batched | GEMM theo lô và theo lô có bước nhảy |
| 05 | curand-monte-carlo | Sinh số ngẫu nhiên trên thiết bị, và "ngẫu nhiên song song" nghĩa là gì |
| 06 | cufft-convolution | Tích chập bằng FFT |
| 07 | cusparse-spmv | SpMV dạng CSR — bất quy tắc, nghẽn bộ nhớ, không tránh được |
| 08 | mixed-precision | FP16/BF16/TF32, vector hoá `half2`, đánh đổi độ chính xác và tốc độ |
| 09 | tensor-core-wmma | API WMMA (cần sm_70+) |
| 10 | cutlass-gemm | GEMM dạng template, gộp epilogue *(phụ thuộc tuỳ chọn)* |
| 11 | pytorch-extension | Một toán tử CUDA tự viết gọi được từ Python *(phụ thuộc tuỳ chọn)* |
| 12 | nvrtc-jit | Biên dịch lúc chạy và driver API |
| 13 | cuda-opengl-interop | Trực quan hoá không sao chép *(phụ thuộc tuỳ chọn)* |
| 14 | kernel-fusion | Ít lượt duyệt bộ nhớ hơn — phép tối ưu còn lại có đòn bẩy lớn nhất |

Các bài có phụ thuộc tuỳ chọn chỉ được build khi bật `-DCL_ENABLE_OPTIONAL=ON`.

**Cột mốc.** Bạn tìm đến đúng thư viện trước tiên, gọi được một kernel Tensor Core,
và giải thích được khi nào thì tự viết là chính đáng.

---

## Giai đoạn 6 · Làm chủ

**Liên tục · 🏆 · [thư mục](../phase-6-mastery/README.vi.md)**

Mỗi dự án đi kèm một tài liệu kiến trúc, bảng phân rã mốc công việc, tiêu chí đánh
giá, và một mô-đun lõi được cài đặt sẵn làm ví dụ mẫu. Phần còn lại là của bạn.

| Dự án | Bạn sẽ xây gì |
|---|---|
| gpu-ray-tracer | Path tracing với duyệt BVH, nhiều lần dội, khử nhiễu |
| mini-dl-framework | Autograd, kernel xuôi/ngược, SGD/Adam, gộp toán tử |
| fluid-simulation | Navier–Stokes hoặc SPH, thời gian thực, kèm trực quan hoá |
| gpu-database | Lưu trữ theo cột, filter/join/aggregate trên GPU, biên dịch truy vấn |
| llm-inference-engine | Quản lý KV-cache, paged attention, giải mã theo lô |

**Cột mốc.** Hai dự án hoàn chỉnh, một báo cáo profiler cho mỗi dự án, và một bài
viết thuật lại bạn đã tối ưu cái gì và vì sao.

---

## Theo dõi tiến độ

| Mức | Tiêu chí | Bằng chứng |
|---|---|---|
| Beginner | Kernel đúng, quản lý bộ nhớ thiết bị an toàn | Xong Giai đoạn 1–2 |
| Intermediate | Profile, chẩn đoán, tối ưu | Xong Giai đoạn 3 |
| Advanced | Stream, multi-GPU, thực thi bất đồng bộ, SASS | Xong Giai đoạn 4 |
| Expert | Thư viện, Tensor Core, toán tử tự viết cho framework | Xong Giai đoạn 5 |
| Master | Dự án hoàn thiện, đóng góp cộng đồng | Giai đoạn 6 liên tục |

Các mốc benchmark để tự đối chiếu:

| Kernel | Beginner | Expert |
|---|---|---|
| SGEMM 4096³ | < 5% cuBLAS | > 80% cuBLAS |
| Reduction | < 10% băng thông đỉnh | > 90% băng thông đỉnh |
| Tích chập 2D | naive | > 70% cuDNN |
| Sao chép bộ nhớ | < 30% lý thuyết | > 85% lý thuyết |

---

## Nhịp học gợi ý theo tuần

| Ngày | Trọng tâm | Số giờ |
|---|---|---|
| Thứ 2 | Lý thuyết — đọc README của bài và tài liệu tham chiếu | 2–3 |
| Thứ 3 | Viết mã — điền các `TODO` | 3–4 |
| Thứ 4 | Lý thuyết + viết mã | 2–3 |
| Thứ 5 | Viết mã — hoàn thiện, kiểm chứng, đo đạc | 3–4 |
| Thứ 6 | Profile và tối ưu; so với lời giải tham chiếu | 2–3 |
| Thứ 7 | Làm dự án | 4–6 |
| Chủ nhật | Ôn tập, đọc paper, đọc blog | 2–3 |

---

## Đọc thêm

**Sách, theo thứ tự nên đọc**

1. *CUDA by Example* — Sanders & Kandrot (nhập môn)
2. *Programming Massively Parallel Processors*, tái bản lần 4 — Kirk & Hwu (giáo trình chuẩn)
3. *Professional CUDA C Programming* — Cheng, Grossman & McKercher
4. *The CUDA Handbook* — Nicholas Wilt (tra cứu)

**Tài liệu chính thức**

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

**Bài báo**

- Volkov, *Understanding Latency Hiding on GPUs* (2016)
- Micikevicius và cộng sự, *Mixed Precision Training* (2018)
- Dao và cộng sự, *FlashAttention* (2022) và *FlashAttention-2* (2023)
- Williams, Waterman & Patterson, *Roofline* (CACM 2009)

**Khoá học**

- [NVIDIA DLI — Fundamentals of Accelerated Computing with CUDA C/C++](https://www.nvidia.com/en-us/training/)
- [Coursera — GPU Programming Specialization (Johns Hopkins)](https://www.coursera.org/specializations/gpu-programming)
- [Udacity CS344 — Intro to Parallel Programming](https://github.com/udacity/cs344) (đã lưu trữ, vẫn rất hay)

**Blog**

- [NVIDIA Technical Blog](https://developer.nvidia.com/blog)
- [Simon Boehm — How to optimise a CUDA matmul kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Blog của Lei Mao](https://leimao.github.io/)

**Công cụ**

- [Nsight Systems](https://developer.nvidia.com/nsight-systems) · [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [Compiler Explorer](https://godbolt.org/) — có hỗ trợ CUDA, tiện để đọc PTX
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Các liên kết bổ sung được gom trong [`resources/README.md`](../resources/README.md).

---

> **Lời khuyên quan trọng nhất.** Không thể học CUDA bằng lý thuyết. Hãy viết mã
> mỗi ngày, profile mọi thứ, và luôn so với bản cài đặt tốt nhất hiện có (cuBLAS,
> cuDNN, CUB). Khoảng cách giữa một kernel *chạy được* và một kernel *chạy nhanh*
> chính là nơi việc học thật sự diễn ra.
