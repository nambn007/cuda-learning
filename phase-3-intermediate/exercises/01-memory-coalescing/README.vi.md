<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 01 - Memory coalescing, đo tận tay

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [GĐ 1/05](../../../phase-1-foundation/exercises/05-cache-and-bandwidth/README.vi.md), [GĐ 2/06](../../../phase-2-cuda-fundamentals/exercises/06-matrix-add-2d/README.vi.md) | **Cần GPU**

## Mục tiêu

Đo sự thật quan trọng nhất về hiệu năng GPU. Mọi thứ còn lại trong Giai đoạn 3 đều
là kỹ thuật để lách những trường hợp bạn không thể thoả mãn nó một cách trực tiếp.

## Kiến thức nền

> **Bộ nhớ không được chuyển theo từng byte, mà theo từng khối.**

Bạn đã đo điều này trên CPU ở [GĐ 1/05](../../../phase-1-foundation/exercises/05-cache-and-bandwidth/README.vi.md),
nơi khối là một cache line 64 byte. Trên GPU nó là một **sector 32 byte**. Khi một
warp phát lệnh nạp, phần cứng tính xem 32 địa chỉ rơi vào bao nhiêu sector khác
nhau rồi nạp **toàn bộ** chúng:

| Cách truy cập | Sector mỗi warp | Byte thật sự dùng |
|---|---|---|
| 32 float liên tiếp | 4 | 128 / 128 |
| 32 float, stride 8 | **32** | 128 / 1024 |

Trường hợp thứ hai chuyển gấp 8 lần dữ liệu cho cùng một kết quả. Mã nguồn trông
không khác gì và đáp án thì y hệt.

Ba hiệu ứng, được đo tách bạch:

**1. Stride.** Khi khoảng cách địa chỉ giữa các thread liền kề lớn dần, số sector
mỗi warp tăng theo — cho tới khi mỗi thread có sector riêng và đường cong nằm
ngang. Bạn không thể phí nhiều hơn trọn một sector cho mỗi lần truy cập.

**2. Căn chỉnh.** Ngay cả một truy cập liên tục hoàn hảo cũng phải trả thêm một
sector mỗi warp nếu nó không bắt đầu đúng ranh giới sector.

**3. Cách ánh xạ.** "Cho mỗi thread một đoạn liên tục" là đúng trên CPU và là thảm
hoạ trên GPU. Đây là lỗi phổ biến nhất khi chuyển mã từ CPU sang.

## Nhiệm vụ của bạn

Mở `main.cu` và viết bốn kernel: `stridedRead`, `offsetRead`, `blockPerThread` và
`interleaved`. Rồi chạy ba phép quét.

**Hãy viết dự đoán ra trước mỗi thí nghiệm.** Đặc biệt là trước thí nghiệm 3: bạn
nghĩ chunk-per-thread chậm hơn bao nhiêu? Đa số đoán 2 lần.

Với phép quét stride, hãy in cả `sectorsPerWarp(stride)` của mô hình và một băng
thông *dự đoán* bên cạnh giá trị đo được. Quan hệ giữa hai cột đó chính là toàn bộ
bài học.

## Build và chạy

```bash
cmake --build build --target p3_01_memory_coalescing -j
./build/bin/p3/p3_01_memory_coalescing
```

## Kết quả mong đợi

RTX 3060, mảng 256 MB:

```
--- Experiment 1 - stride ---
  stride     time (ms)   useful GB/s  sectors/warp   predicted   of peak
  ----------------------------------------------------------------------------
  1              1.752         306.5           4.0       306.5       85%
  2              1.240         216.5           8.0       324.7       60%
  4              1.002         133.9          16.0       334.7       37%
  8              0.908          73.9          32.0       332.5       21%
  16             0.455          73.8          32.0       332.1       20%
  32             0.229          73.4          32.0       330.3       20%
  128            0.072          58.5          32.0       263.3       16%

--- Experiment 2 - alignment ---
  offset     time (ms)   useful GB/s vs offset 0
  ---------------------------------------------------
  0              1.750         306.7       1.00x
  1              1.929         278.2       0.91x
  4              1.924         279.0       0.91x
  8              1.804         297.6       0.97x
  16             1.800         298.2       0.97x
  31             1.911         280.9       0.92x
  32             1.745         307.7       1.00x

--- Experiment 3 ---
32-element chunk per thread          19.765     27.16 GB/s     1.00x
interleaved (grid-stride)             1.743    308.04 GB/s    11.34x
```

### Cách đọc bảng

**Cột `predicted` đứng quanh 300 GB/s ở mọi dòng.** Đó chính là điểm mấu chốt: hệ
thống bộ nhớ làm việc *vất vả y như nhau* ở stride 128 và ở stride 1. Nó đang
chuyển dữ liệu mà bạn vứt đi. Băng thông hữu ích sụp 4 lần trong khi bus vẫn bão hoà.

**Căn chỉnh có tính tuần hoàn, không đơn điệu.** Offset 8, 16 và 32 quay lại tốc độ
tối đa; 1, 2, 4 và 31 thì không. 8 float = 32 byte = đúng một sector, nên thứ quan
trọng là `offset % 8`, chứ không phải offset lớn cỡ nào. Mức phạt là ~9% chứ không
phải 25% như số sector gợi ý, vì sector dôi ra của một warp thường chính là sector
đầu tiên của warp kế tiếp, và L2 phục vụ nó.

**11.34 lần.** Chunk-per-thread chính là cấu trúc vòng lặp *đúng* trên CPU. Hãy đề
phòng nó khi chuyển mã.

## Điểm cốt lõi

- **Các thread liền kề phải chạm vào các địa chỉ liền kề.** Đó là toàn bộ quy tắc.
- **Mẫu truy cập tệ làm bão hoà bus mà không mang lại băng thông.** Đừng bao giờ
  đánh giá kernel chỉ bằng GB/s — hãy đánh giá bằng GB/s *hữu ích*.
- **Mức phạt do stride bão hoà** ở một sector mỗi thread. Vượt qua đó, stride lớn
  hơn cũng không tốn thêm.
- **Lệch căn chỉnh tốn một sector mỗi warp**, và có tính tuần hoàn theo kích thước
  sector.
- **Cách ánh xạ đúng trên CPU là thảm hoạ trên GPU.**

## Đi xa hơn

- Chạy `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` cho từng
  stride — số sector đo được có khớp `sectorsPerWarp()` không?
- Lặp lại với `double` (8 byte). Bây giờ sự sụp đổ bắt đầu từ stride nào?
- Thu mảng xuống 1 MB để nó vừa L2 rồi chạy lại. Mức phạt biến đi đâu?
- **Tiếp theo:** [02 shared-memory-basics](../02-shared-memory-basics/) giới thiệu
  vùng nhớ trung gian cho phép bạn thoả mãn quy tắc này ngay cả khi thuật toán
  chống lại bạn; [04](../04-tiled-matmul/) và [05](../05-transpose-optimized/) áp
  dụng nó.
