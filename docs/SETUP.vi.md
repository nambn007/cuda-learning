<!-- Ngôn ngữ: [English](SETUP.md) | **Tiếng Việt** -->

# Cài đặt

[← quay lại kho mã](../README.vi.md)

Hai lựa chọn: **Docker** (khuyến nghị — bộ công cụ, profiler và thư viện được ghim
phiên bản, giống hệt nhau với mọi người) hoặc **cài trực tiếp lên máy**.

---

## Lựa chọn 1 · Docker

### Điều kiện cần

- Một GPU NVIDIA có driver đã cài trên **máy chủ** (`nvidia-smi` phải chạy được).
- Docker Engine 19.03+ và [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Kiểm tra toolkit đã nối đúng chưa:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Nếu lệnh này in ra GPU của bạn thì mọi thứ còn lại sẽ chạy được.

### Khởi động

```bash
docker compose up -d --build     # lần build đầu mất 10-20 phút
docker compose exec cuda-dev bash

# bên trong container
verify-cuda
./scripts/build.sh
./scripts/run-all.sh
```

Kho mã được gắn (bind-mount) vào `/workspace`, nên bạn cứ sửa file trên máy chủ
bằng editor quen thuộc và build bên trong container.

Muốn bỏ qua phần tải ~500 MB của Nsight Systems:

```bash
docker compose build --build-arg INSTALL_NSIGHT_SYSTEMS=0
```

### VS Code Dev Containers

Cài extension **Dev Containers**, mở thư mục dự án, rồi `Ctrl+Shift+P` →
*Reopen in Container*. File `.devcontainer/devcontainer.json` đã cấu hình sẵn các
extension CUDA, C++, CMake và trỏ IntelliSense vào `common/`.

---

## Lựa chọn 2 · Cài trực tiếp

### Linux (Ubuntu 22.04 / 24.04)

```bash
# Driver, nếu bạn chưa có
sudo ubuntu-drivers autoinstall && sudo reboot

# CUDA Toolkit - làm theo bộ cài chính thức cho bản phân phối của bạn
# https://developer.nvidia.com/cuda-downloads
sudo apt install -y build-essential cmake ninja-build git

# Thêm CUDA vào PATH (chỉnh lại số phiên bản)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version
```

Sau đó:

```bash
./scripts/build.sh
./scripts/run-all.sh
```

### Windows

Hãy dùng **WSL2** với một bản Ubuntu rồi làm theo hướng dẫn Linux ở trên. Driver
NVIDIA trên Windows tự động cấp GPU cho WSL2; **đừng** cài driver bên trong WSL.
Windows thuần với MSVC không được kiểm thử ở đây.

### Công cụ tuỳ chọn

| Công cụ | Dùng cho | Cách cài |
|---|---|---|
| Nsight Systems (`nsys`) | GĐ 3/16, GĐ 4 | `apt install nsight-systems-cli` |
| Nsight Compute (`ncu`) | GĐ 3/16 | đi kèm CUDA toolkit |
| CUTLASS | GĐ 5/10 | `git clone https://github.com/NVIDIA/cutlass` |
| PyTorch | GĐ 5/11 | `pip install torch` |
| Header OpenGL | GĐ 5/13 | `apt install libglfw3-dev libglew-dev` |

Build các bài tuỳ chọn bằng `./scripts/build.sh --optional`.

---

## Tuỳ chọn build

```bash
cmake -S . -B build -DCUDA_ARCH=native      # mặc định: tự phát hiện GPU của máy
cmake -S . -B build -DCUDA_ARCH=86          # một kiến trúc duy nhất (build nhanh hơn)
cmake -S . -B build -DCUDA_ARCH=portable    # fat binary, sm_70..sm_89
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug   # -G, cho cuda-gdb
cmake -S . -B build -DCL_PTXAS_VERBOSE=ON      # in thanh ghi/shared từng kernel
cmake -S . -B build -DCL_ENABLE_OPTIONAL=ON    # các bài có phụ thuộc ngoài
```

Các target tiện dụng:

```bash
cmake --build build --target solutions -j   # toàn bộ lời giải tham chiếu
cmake --build build --target starters -j    # toàn bộ bản khởi đầu
ctest --test-dir build -L p3                # chạy một giai đoạn
ctest --test-dir build -R reduction         # chạy theo tên
```

> **Đừng bao giờ benchmark bản Debug.** Cờ `-G` tắt hầu hết tối ưu trên thiết bị và
> kernel chạy chậm hơn nhiều lần.

---

## Xử lý sự cố

**`nvcc: command not found`**
CUDA chưa nằm trong `PATH`. Xem lại các dòng export ở trên, hoặc dùng Docker.

**`no kernel image is available for execution on the device`**
File nhị phân được build cho kiến trúc khác. Build lại với `-DCUDA_ARCH=native`,
hoặc truyền thẳng compute capability của bạn
(`nvidia-smi --query-gpu=compute_cap --format=csv`).

**`CUDA driver version is insufficient for CUDA runtime version`**
Toolkit mới hơn driver. Hãy cập nhật driver hoặc dùng toolkit cũ hơn — phiên bản
driver phải ≥ phiên bản toolkit.

**`Failed to detect a default CUDA architecture`** khi chạy `cmake`
Không thấy GPU nào lúc cấu hình. Dùng `-DCUDA_ARCH=portable`, hoặc truyền thẳng
con số.

**`nvidia-smi` chạy trên máy chủ nhưng không chạy trong container**
Thiếu NVIDIA Container Toolkit, hoặc chưa khởi động lại Docker sau khi cài. Hãy
kiểm tra bằng lệnh `docker run --gpus all ... nvidia-smi` ở trên.

**`ncu` báo `ERR_NVGPUCTRPERM`**
Bộ đếm profiling cần quyền cao hơn. File compose đã thêm sẵn
`cap_add: [SYS_ADMIN]`; nếu cài trực tiếp, hãy làm theo
[hướng dẫn của NVIDIA](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm)
để cho phép profiling không cần root.

**Một bài tập in ra `[SKIP]`**
Nó cần phần cứng hoặc thư viện mà bạn không có — GPU thứ hai, Tensor Core,
CUTLASS. Đó không phải lỗi; bài tập sẽ nói rõ yêu cầu nào chưa được đáp ứng.

**Kernel thỉnh thoảng mới cho kết quả sai**
Đó là race condition. Chạy `compute-sanitizer --tool racecheck ./binary_của_bạn`.
Với truy cập ngoài phạm vi, dùng `compute-sanitizer ./binary_của_bạn`.

---

## Kiểm tra một bản cài mới

```bash
verify-cuda                                  # bộ công cụ + một lần chạy kernel thật
./scripts/build.sh                           # mọi thứ biên dịch được
./scripts/run-all.sh                         # mọi thứ cho kết quả đúng
```

Cả ba đều sạch nghĩa là bạn đã sẵn sàng cho
[Giai đoạn 1](../phase-1-foundation/README.vi.md).
