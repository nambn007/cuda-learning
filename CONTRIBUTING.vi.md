<!-- Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt** -->

# Đóng góp

Sửa lỗi, thêm bài tập mới và cải thiện bản dịch đều được hoan nghênh.

## Thêm một bài tập

```bash
./scripts/new-exercise.sh 3 17 wave-propagation "Wave propagation"
```

Lệnh đó tạo ra `phase-3-intermediate/exercises/17-wave-propagation/` với năm file
mà một bài tập cần. Không có danh sách tập trung nào phải cập nhật — chỉ cần chạy
lại `cmake -S . -B build` là thư mục mới tự động thành hai target.

### Một bài tập bắt buộc phải có gì

| File | Yêu cầu |
|---|---|
| `README.md` | Mục tiêu, kiến thức nền, nhiệm vụ, cách build/chạy, kết quả mong đợi, điểm cốt lõi, đọc thêm |
| `README.vi.md` | Cùng nội dung bằng tiếng Việt, có thanh chuyển ngôn ngữ ở dòng 1 |
| `reference.h` | Kết quả chuẩn tính trên CPU và phần thiết lập bài toán, dùng chung cho cả hai biến thể |
| `main.cu` | Bản khởi đầu. Phải **biên dịch và chạy được** khi các `TODO` còn trống, và in ra phần còn thiếu thay vì crash |
| `solution.cu` | Lời giải tham chiếu. Phải tự kiểm chứng kết quả và thoát với mã 0 |

Tuỳ chọn: `exercise.cmake` để khai báo thêm thư viện, cờ biên dịch hoặc thời gian
chờ kiểm thử dài hơn.

### Những quy tắc giữ cho kho mã nhất quán

**Mọi chương trình phải tự kiểm chứng.** So với bản tham chiếu CPU và trả về
`verifySummary()`. `ctest` là bộ kiểm thử hồi quy của kho mã; một bài tập không
thể fail thì không làm tròn nhiệm vụ của nó.

**Mọi chương trình phải báo cáo tốc độ, không chỉ thời gian.** GB/s cho phần nghẽn
bộ nhớ, GFLOP/s cho phần nghẽn tính toán, và mức tăng tốc so với một mốc cơ sở đã
nêu rõ. Hãy dùng `ResultTable`.

**Chú thích chỉ bằng tiếng Anh, trong mọi file.** Một file nguồn phục vụ cả hai
nhóm người đọc và vẫn dễ diff. Tài liệu thì song ngữ; mã nguồn thì không.

**Chú thích giải thích *vì sao*.** `// tăng i` là nhiễu. `// bốn biến tích luỹ vì
một lệnh FMA có độ trễ ~4 chu kỳ` mới chính là bài học.

**Đừng bao giờ khẳng định một kết quả bạn chưa đo.** Hãy đưa số liệu thật từ một
lần chạy thật vào mục "Kết quả mong đợi", và ghi rõ GPU nào tạo ra chúng. Nếu một
cách sửa hoá ra không giúp gì trên phần cứng của bạn, hãy nói thẳng và giải thích
trong điều kiện nào thì nó sẽ có tác dụng — một kết quả âm trung thực dạy được
nhiều hơn một chiến thắng bịa ra.

**Bỏ qua, đừng để fail.** Nếu một bài tập cần hai GPU, cần Tensor Core hoặc thư
viện ngoài, hãy phát hiện tình huống đó và trả về `skipExercise("...")`. `[SKIP]`
không phải lỗi.

**Suy giảm êm ái giữa các phiên bản CUDA.** Bảo vệ các API phụ thuộc phiên bản
bằng `#if CUDART_VERSION < 13000` hoặc tương tự. Kho mã này nhắm CUDA 11–13.

## Quy ước viết mã

- Thụt lề 4 khoảng trắng, dòng dài tối đa 100 cột; `.clang-format` theo style
  Google với hai thay đổi trên.
- Tiền tố `d_` cho con trỏ thiết bị, `h_` cho con trỏ host.
- Ưu tiên bọc mọi lời gọi runtime bằng `CUDA_CHECK(...)`. Nó không tốn gì và tiết
  kiệm hàng giờ.
- Mặc định dùng grid-stride loop.

## Về bản dịch

Bản tiếng Việt là một **tài liệu song song, không phải bản dịch máy**. Hãy giữ
thuật ngữ kỹ thuật bằng tiếng Anh ở những chỗ mà người đọc sẽ tra cứu bằng từ đó
(`warp`, `coalescing`, `occupancy`) — [docs/GLOSSARY.vi.md](docs/GLOSSARY.vi.md)
ghi lại bảng đối chiếu đã thống nhất. Nếu bạn thêm một thuật ngữ, hãy thêm vào đó
luôn.

Hai bản ngôn ngữ phải luôn đồng bộ. Một pull request sửa `README.md` mà không sửa
`README.vi.md` sẽ được yêu cầu bổ sung nửa còn lại.

## Trước khi mở pull request

```bash
./scripts/build.sh --clean          # mọi thứ biên dịch được
./scripts/run-all.sh                # mọi thứ chạy đúng
```

Hãy ghi rõ bạn đã kiểm thử trên GPU nào và phiên bản CUDA nào.
