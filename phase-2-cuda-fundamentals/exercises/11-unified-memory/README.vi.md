<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 11 - Bộ nhớ hợp nhất: một con trỏ, và cái giá của nó

> **Giai đoạn 2 · CUDA cơ bản** | Độ khó: ⭐⭐ | Thời gian: ~1.5 giờ | Yêu cầu: [03](../03-vector-add/README.vi.md) | **Cần GPU**

## Mục tiêu

Đánh đổi `cudaMemcpy` tường minh lấy một con trỏ duy nhất dùng được ở mọi nơi, đo
chính xác sự tiện lợi đó tốn bao nhiêu, và học điều duy nhất mà bộ nhớ quản lý làm
được còn cấp phát tường minh thì hoàn toàn không.

## Kiến thức nền

`cudaMallocManaged` trả về **một con trỏ hợp lệ trên cả host lẫn device**. Không
`cudaMemcpy`, không cặp biến `h_`/`d_` song song, không lẫn lộn đang cầm cái nào.

Cái giá: driver phải di chuyển các trang nhớ hộ bạn, mà nó chỉ biết chúng cần ở đâu
bằng cách bẫy **page fault**. Từ Pascal trở đi:

```
host ghi      →  các trang di trú về bộ nhớ host
kernel đọc    →  mỗi lần chạm đầu tiên đều fault; driver di trú 4 KB;
                 warp phải chờ
host đọc      →  tất cả lại di trú ngược về
```

**Di trú theo fault ở mức 4 KB chậm hơn nhiều so với một lần DMA khối lớn.** Cách
sửa là nói cho driver biết thứ bạn vốn đã biết:

| Lời gọi | Tác dụng |
|---|---|
| `cudaMemPrefetchAsync(p, bytes, device)` | chuyển ngay, theo khối lớn |
| `cudaMemAdvise(..., SetReadMostly)` | nhân bản dữ liệu chỉ đọc ở cả hai phía |
| `cudaMemAdvise(..., SetPreferredLocation)` | ghim nơi cư trú chính |

**Oversubscription** (cấp phát vượt dung lượng) là tính năng không có bản tương
đương trong API tường minh. Bạn có thể cấp phát *nhiều bộ nhớ quản lý hơn dung
lượng vật lý của GPU*, và driver sẽ phân trang ra vào. Đó là điều khiến việc chạy
một mô hình lớn hơn VRAM trở nên khả thi. Với `cudaMalloc` bạn sẽ phải tự tay chia
nhỏ bài toán.

## Nhiệm vụ của bạn

Mở `main.cu` và làm năm thí nghiệm.

> **Quy tắc khiến bài này có ý nghĩa:** mọi biến thể phải làm đúng cùng một khối
> lượng công việc — *host ghi cả hai mảng → GPU tính → host đọc kết quả về*. Bỏ
> phần khởi tạo trên host ra khỏi một biến thể mà không bỏ ở các biến thể khác là
> cách dễ nhất để có một con số vô nghĩa, bởi vì toàn bộ chi phí của bộ nhớ quản lý
> chính là phần di chuyển do việc khởi tạo đó kích hoạt.
>
> Ngoài ra, hãy kiểm tra tính đúng đắn bằng **một lời gọi riêng lẻ**, không đặt
> trong vòng lặp đo thời gian — vòng lặp chạy thân hàm nhiều lần và kernel này cộng
> dồn vào `y`.

1. **Tường minh** `cudaMalloc` + `cudaMemcpy` — mốc cơ sở.
2. **Quản lý, không gợi ý gì.**
3. **Quản lý + `cudaMemPrefetchAsync`** trước mỗi lần một bên chạm vào dữ liệu.
4. **Chỉ riêng kernel**, khi dữ liệu đã nằm sẵn trên thiết bị.
5. **Oversubscription** — cấp phát 1.5× `totalGlobalMem` theo cả hai cách rồi đọc
   hai mã lỗi.

## Build và chạy

```bash
cmake --build build --target p2_11_unified_memory -j
cd build/testrun && ../bin/p2/p2_11_unified_memory
```

## Kết quả mong đợi

RTX 3060, 256 MB mỗi mảng:

```
--- Comparison ---
Variant                           Time (ms)   Speedup
------------------------------------------------------
explicit malloc + memcpy            153.851     1.00x
managed, no hints                   260.437     0.59x
managed + prefetch                  173.759     0.89x

  Managed memory without hints is 1.69x the explicit version.
  With prefetching it is 1.13x.

--- Kernel time with data already on the device ---
  kernel alone                 3.109 ms
  259.0 GB/s  (72% of peak)

--- Oversubscription ---
  This GPU has 11.6 GB. Trying to allocate 17.4 GB...
  cudaMalloc                   cudaErrorMemoryAllocation
  cudaMallocManaged            cudaSuccess
  [PASS] managed memory can exceed device memory
```

Ba kết luận, đều từ số đo:

1. **Bộ nhớ quản lý dùng ngây thơ tốn 1.69 lần** — có thật, nhưng chưa thảm hoạ.
2. **Prefetch lấy lại gần hết** (còn 1.13 lần). Lượng dữ liệu di chuyển là *y hệt*;
   chỉ có độ hạt thay đổi.
3. **Một khi đã cư trú đúng chỗ, bộ nhớ quản lý nhanh y như** bộ nhớ tường minh —
   nó vốn là cùng một vùng nhớ vật lý.

## Điểm cốt lõi

- **Bộ nhớ quản lý không phải bộ nhớ chậm hơn.** Nó là cùng DRAM. Thứ tốn chi phí
  là *di trú theo fault*, và một lệnh prefetch xoá bỏ điều đó.
- **Luôn đi kèm `cudaMallocManaged` với `cudaMemPrefetchAsync`** một khi bạn đã
  biết mẫu truy cập. Sự tiện lợi thì miễn phí; sự thiếu hiểu biết thì không.
- **Oversubscription không có lựa chọn thay thế.** Nếu tập dữ liệu vượt VRAM, bộ
  nhớ quản lý là thứ duy nhất chạy được.
- **Benchmark công bằng đòi hỏi khối lượng công việc giống hệt nhau.** Làm sai điều
  này rất dễ, và con số thu được trông vẫn hợp lý.

### Khi nào dùng cái nào

**Quản lý:** làm nguyên mẫu; cấu trúc bất quy tắc hoặc phải lần theo con trỏ (cây,
đồ thị) khi bạn không biết trước trang nào sẽ cần; tập dữ liệu lớn hơn VRAM.

**Tường minh:** kernel sản xuất với mẫu truy cập đều đặn đã biết; bất cứ thứ gì cần
chồng lấn truyền dữ liệu với tính toán bằng stream (GĐ 4/03).

## Đi xa hơn

- Thêm `cudaMemAdvise(x, bytes, cudaMemAdviseSetReadMostly, device)` cho mảng chỉ
  đọc. Nó có giúp không?
- Profile bằng `nsys profile --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true`
  và đếm số fault ở bản ngây thơ.
- Chạm vào *toàn bộ* vùng cấp phát vượt dung lượng. Thông lượng thay đổi thế nào
  khi tập dữ liệu vượt VRAM?
- Đọc [Programming Guide phần Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming)
  và bài blog NVIDIA *Maximizing Unified Memory Performance*.
