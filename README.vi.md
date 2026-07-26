<div align="center">

# CUDA Learning

**Lộ trình học CUDA thực hành — từ nền tảng C++ đến làm chủ GPU**

[🇬🇧 English](README.md) · **🇻🇳 Tiếng Việt**

[Lộ trình](docs/ROADMAP.vi.md) · [Cài đặt](docs/SETUP.vi.md) · [Thuật ngữ](docs/GLOSSARY.vi.md) · [Đóng góp](CONTRIBUTING.vi.md)

</div>

---

## Đây là gì

Sáu giai đoạn, 63 bài tập và 5 dự án portfolio đưa bạn từ chỗ "biết chút C++" đến
chỗ viết được kernel GPU mà bạn có thể bảo vệ bằng profiler.

Mỗi bài tập gồm **bốn loại file**:

| File | Nội dung |
|---|---|
| `README.md` / `README.vi.md` | đề bài, lý thuyết, và vì sao nó quan trọng — bằng cả hai thứ tiếng |
| `main.cu` | bản **khởi đầu** với các `TODO` để bạn tự điền |
| `solution.cu` | **lời giải tham chiếu**, chú thích giải thích *vì sao*, không phải *cái gì* |
| `reference.h` | kết quả chuẩn tính trên CPU, dùng chung để cả hai được kiểm tra như nhau |

Mọi chương trình đều **tự kiểm chứng kết quả với bản tham chiếu CPU và thoát với
mã khác 0 khi sai lệch**, nên cả kho mã đồng thời là một bộ kiểm thử hồi quy:

```bash
ctest --test-dir build --output-on-failure
```

Và mọi chương trình đều **tự báo cáo hiệu năng** bằng những đơn vị có ý nghĩa —
GB/s, GFLOP/s, và mức tăng tốc so với một mốc cơ sở. Bạn không bao giờ phải tin
suông một tuyên bố nào.

## Bắt đầu nhanh

```bash
git clone <repo-này> && cd cuda-learning

# 1. Môi trường phát triển (CUDA toolkit, Nsight, profiler)
docker compose up -d --build
docker compose exec cuda-dev bash

# 2. Kiểm tra GPU thật sự dùng được
verify-cuda

# 3. Build toàn bộ và chạy các lời giải tham chiếu
./scripts/build.sh
./scripts/run-all.sh

# 4. Bắt đầu học
cat phase-1-foundation/exercises/01-matmul-naive-cpu/README.vi.md
```

Không dùng Docker? Xem [docs/SETUP.vi.md](docs/SETUP.vi.md) để cài trực tiếp.

## Sáu giai đoạn

| Giai đoạn | Chủ đề | Số bài | Thời gian | Mức độ |
|---|---|---|---|---|
| [1](phase-1-foundation/README.vi.md) | **Nền tảng** — C++, cache, SIMD, roofline | 8 | 3–4 tuần | ⭐ |
| [2](phase-2-cuda-fundamentals/README.vi.md) | **CUDA cơ bản** — kernel, bộ nhớ, đánh chỉ số | 12 | 4–6 tuần | ⭐⭐ |
| [3](phase-3-intermediate/README.vi.md) | **Trung cấp** — coalescing, shared memory, mẫu song song | 16 | 6–8 tuần | ⭐⭐⭐ |
| [4](phase-4-advanced/README.vi.md) | **Nâng cao** — stream, graph, multi-GPU, atomic, PTX | 13 | 8–10 tuần | ⭐⭐⭐⭐ |
| [5](phase-5-expert/README.vi.md) | **Chuyên sâu** — cuBLAS, CUB, Tensor Core, mixed precision | 14 | 6–8 tuần | ⭐⭐⭐⭐⭐ |
| [6](phase-6-mastery/README.vi.md) | **Làm chủ** — năm dự án portfolio | 5 dự án | liên tục | 🏆 |

Chi tiết từng bài kèm mục tiêu học tập: **[docs/ROADMAP.vi.md](docs/ROADMAP.vi.md)**.

**Nên bắt đầu từ đâu.** Đã vững C++, cache và mô hình roofline? Hãy nhảy thẳng vào
Giai đoạn 2. Nếu chưa thì Giai đoạn 1 không phải phần đệm cho có — mọi khái niệm
trong đó (bố cục row-major, cache line, lane SIMD, cường độ số học) đều xuất hiện
lại trên GPU dưới một cái tên khác, và Giai đoạn 3 mặc định bạn đã nắm chắc chúng.

## Cách làm một bài tập

```bash
# 1. Đọc đề
cat phase-2-cuda-fundamentals/exercises/03-vector-add/README.vi.md

# 2. Điền các TODO trong main.cu, rồi build và chạy
cmake --build build --target p2_03_vector_add -j
./build/bin/p2/p2_03_vector_add

# 3. Bí, hoặc muốn đối chiếu? Đọc lời giải tham chiếu
cmake --build build --target p2_03_vector_add_sol -j
./build/bin/p2/p2_03_vector_add_sol
```

Chạy một bản khởi đầu chưa hoàn thành là an toàn: nó in ra phần còn thiếu thay vì
crash.

Tên target theo mẫu `p<giai_đoạn>_<số>_<slug>`, thêm hậu tố `_sol` cho lời giải
tham chiếu. File thực thi nằm ở `build/bin/p<giai_đoạn>/`.

## Yêu cầu

- **GPU:** NVIDIA, compute capability ≥ 5.0. Một số bài Giai đoạn 5 cần ≥ 7.0
  (Tensor Core) và sẽ tự bỏ qua một cách gọn gàng nếu không có.
- **CUDA Toolkit:** ≥ 11.0, khuyến nghị 12.x. CUDA 13 vẫn chạy — phần mã bị ảnh
  hưởng đã được bảo vệ theo phiên bản.
- **Trình biên dịch host:** GCC ≥ 9 hoặc Clang ≥ 10, C++17.
- **CMake:** ≥ 3.20.

Hệ thống build tự phát hiện GPU của bạn (`-DCUDA_ARCH=native`). Ghi đè bằng
`-DCUDA_ARCH=86`, hoặc `-DCUDA_ARCH=portable` để tạo fat binary.

Các bài cần nhiều hơn một GPU, hoặc cần thư viện ngoài như CUTLASS hay PyTorch, sẽ
**tự bỏ qua kèm lời giải thích** thay vì làm hỏng cả bộ kiểm thử.

## Bố cục kho mã

```
common/              header dùng chung: kiểm tra lỗi, đo thời gian, kiểm chứng, đọc/ghi PPM
cmake/               tự động phát hiện bài tập
docs/                lộ trình, cài đặt, thuật ngữ   (mỗi file đều có bản .vi.md)
scripts/             build.sh, run-all.sh, new-exercise.sh
phase-N-*/
  README.md            tổng quan giai đoạn và checklist
  exercises/NN-slug/   README.md, README.vi.md, main.cu, solution.cu, reference.h
phase-6-mastery/
  projects/slug/       đặc tả, kiến trúc, mốc công việc, mã khởi đầu
```

Không có danh sách target tập trung nào cả. Cứ thả một thư mục chứa `main.cu` vào
bất kỳ thư mục `exercises/` nào, chạy lại `cmake`, và nó sẽ được build — xem
[`scripts/new-exercise.sh`](scripts/new-exercise.sh).

## Song ngữ ngay từ thiết kế

Mọi tài liệu đều tồn tại hai bản: `X.md` (English) và `X.vi.md` (Tiếng Việt), có
thanh chuyển ngôn ngữ ngay dòng đầu tiên. **Chú thích trong mã chỉ dùng tiếng
Anh**, để một file nguồn duy nhất phục vụ được cả hai nhóm người đọc và vẫn dễ diff.

[docs/GLOSSARY.vi.md](docs/GLOSSARY.vi.md) đối chiếu thuật ngữ kỹ thuật giữa hai
ngôn ngữ — rất hữu ích khi bạn đọc tài liệu NVIDIA sau khi đã học khái niệm bằng
tiếng Việt.

## Tình trạng hiện tại

Các giai đoạn được xây dựng lần lượt theo thứ tự. Giai đoạn 1 đã hoàn tất và được
kiểm chứng đầu-cuối trên RTX 3060; các giai đoạn sau đang được bổ sung từng bài.
Mọi bài tập đã có trong cây thư mục đều build được và pass `ctest`; bài nào chưa
viết thì đơn giản là chưa có mặt. Xem [docs/ROADMAP.vi.md](docs/ROADMAP.vi.md) để
biết kế hoạch đầy đủ.

## Giấy phép

MIT. Tài liệu học tập được tham chiếu trong các bài tập thuộc về tác giả tương ứng.
