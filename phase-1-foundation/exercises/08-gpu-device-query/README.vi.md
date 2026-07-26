<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 08 - Hiểu rõ GPU của bạn

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐ | Thời gian: ~1 giờ | Yêu cầu: [07](../07-roofline-model/) | **Cần có GPU**

## Mục tiêu

File `.cu` đầu tiên của bạn — và nó không chạy kernel nào cả. Thay vào đó bạn
"thẩm vấn" phần cứng và tính ra trên giấy xem nó *có thể* làm được gì, để mọi con
số hiệu năng từ Giai đoạn 2 trở đi đều có cái để đối chiếu.

Làm xong bài này bạn sẽ trả lời được, cho chính card của mình: bao nhiêu thread
chạy được cùng lúc, đọc được bao nhiêu byte mỗi giây, một kernel phải làm bao
nhiêu FLOP trên mỗi byte để thoát khỏi cảnh nghẽn bộ nhớ, và một thread được dùng
bao nhiêu thanh ghi trước khi occupancy sụp.

## Kiến thức nền

`cudaGetDeviceProperties()` điền vào một struct `cudaDeviceProp` với khoảng 80
trường. Bốn nhóm có ý nghĩa với phần còn lại của lộ trình:

**Compute capability** (`major.minor`, viết là `sm_86`) — phiên bản tập lệnh. Nó
quyết định lệnh nào tồn tại: Tensor Core cần `sm_70+`, sao chép bất đồng bộ cần
`sm_80+`, điều khiển L2 residency cần `sm_80+`. Nó *không* phải con số hiệu năng;
một card `sm_75` to vẫn thắng một card `sm_86` nhỏ.

**Đỉnh lý thuyết** — phải tự tính, không có sẵn:

```
GB/s đỉnh    = 2 × memoryClockRate(kHz) × 1e3 × (memoryBusWidth / 8) / 1e9
GFLOP/s đỉnh = 2 × coresPerSM × multiProcessorCount × clockRate(kHz) × 1e3 / 1e9
```

Số `2` thứ nhất là vì GDDR chạy *double data rate*; số `2` thứ hai là vì một phép
FMA được tính thành hai phép toán. `coresPerSM` không được runtime cung cấp — đó
là bảng tra phần cứng, đã có sẵn trong `common/cuda_helper.h`.

**Điểm gãy (ridge point)** — `GFLOP/s đỉnh / GB/s đỉnh`, đơn vị FLOP trên byte.
Trên RTX 3060 con số này khoảng **37**: một kernel phải thực hiện 37 phép tính dấu
phẩy động cho mỗi byte nó chạm tới thì GPU mới chuyển sang nghẽn tính toán. Phép
cộng vector chỉ đạt 0.08. Riêng con số này giải thích vì sao Giai đoạn 3 chủ yếu
nói về bộ nhớ.

**Ngân sách occupancy** — mỗi SM có số thanh ghi và lượng shared memory cố định,
chia đều cho toàn bộ thread đang thường trú. Trên RTX 3060: 65536 thanh ghi ÷ 1536
thread = **42 thanh ghi mỗi thread** ở occupancy tối đa. Một kernel giữ tile 8×8
float trong thanh ghi đã tiêu hết 64. Vượt ngân sách thì chương trình không lỗi —
occupancy của bạn âm thầm giảm một nửa. Đó chính là nội dung Giai đoạn 3 bài 15.

> **Lưu ý cho người dùng CUDA 13:** `clockRate` và `memoryClockRate` đã bị gỡ khỏi
> `cudaDeviceProp`. Mã nguồn có bảo vệ bằng `CUDART_VERSION`; hãy đọc xung nhịp từ
> `nvidia-smi -q -d CLOCK` thay thế.

## Nhiệm vụ của bạn

Mở `main.cu`:

1. **Băng thông đỉnh** từ `memoryClockRate` và `memoryBusWidth`.
2. **FP32 đỉnh** từ `coresPerSM()`, `multiProcessorCount` và `clockRate`.
3. **Điểm gãy**, rồi so với điểm gãy CPU bạn đo được ở bài 07.
4. **Ngân sách occupancy**: số warp tối đa/SM, số thread tối đa toàn GPU, số thanh
   ghi và lượng shared memory cho mỗi thread ở occupancy 100%.

## Build và chạy

```bash
cmake --build build --target p1_08_gpu_device_query -j
./build/bin/p1/p1_08_gpu_device_query
```

## Kết quả mong đợi

Đo trên RTX 3060:

```
  Device 0: NVIDIA GeForce RTX 3060
  Compute capability : 8.6 (Ampere, sm_86)
  SMs                : 28
  FP32 lanes         : 3584 (128 per SM)
  Peak bandwidth     : 360.0 GB/s (theoretical)
  Peak FP32          : 13167.6 GFLOP/s (theoretical)

--- Roofline ---
  ridge point                  36.58 FLOP/byte

--- Occupancy budget (per SM) ---
  max resident threads / SM          1536
  max resident warps / SM            48
  registers / SM                     65536
  At 100% occupancy each thread may use at most:
  registers / thread                 42
  shared memory / thread             66.7 bytes
  whole GPU, fully occupied          43008 threads
```

## Điểm cốt lõi

- **Compute capability là mức tính năng, không phải tốc độ.**
- **Đỉnh lý thuyết được tính từ xung nhịp và độ rộng bus**, và bạn sẽ không bao
  giờ chạm tới — đạt 80% băng thông đỉnh đã là kết quả xuất sắc.
- **Điểm gãy của GPU cao hơn hẳn của CPU**, nên tỉ lệ kernel GPU bị nghẽn bộ nhớ
  còn lớn hơn nữa.
- **Occupancy là một ngân sách tài nguyên**, và thanh ghi là tài nguyên khan hiếm
  nhất. Khoảng 42 cái mỗi thread là không nhiều.

## Đi xa hơn

- Chạy `nvidia-smi -q` và tìm những con số tương ứng. Con số nào không có ở đó?
- Tính xem đọc hết toàn bộ bộ nhớ card một lượt ở băng thông đỉnh mất bao lâu. Đó
  là giới hạn dưới cho mọi kernel chạm tới toàn bộ dữ liệu.
- Ghi lại bốn câu trả lời ở cuối chương trình — Giai đoạn 3 mặc định bạn đã biết
  chúng cho card của mình.
- Đọc phụ lục [CUDA C++ Programming Guide, "Compute Capabilities"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)
  và tìm dòng ứng với kiến trúc của bạn.
