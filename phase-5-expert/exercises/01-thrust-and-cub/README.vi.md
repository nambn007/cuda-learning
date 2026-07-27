<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 01 - Thrust và CUB

> **Giai đoạn 5 · Chuyên sâu** | Độ khó: ⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [GĐ 3/06](../../../phase-3-intermediate/exercises/06-reduction-variants/README.vi.md) | **Cần GPU** | ✅ đã kiểm chứng trên RTX 3060

## Mục tiêu

Hiệu chuẩn. Tìm xem các kernel tự viết ở Giai đoạn 3 thật sự cách thư viện bao xa,
để sau này bạn biết khi nào tự viết là chính đáng.

## Kiến thức nền

**Thrust** là lớp kiểu STL — `sort`, `reduce`, `transform`, `inclusive_scan` trên
device vector, kèm iterator và functor. Chỉ có header, đi kèm toolkit.

**CUB** là lớp bên dưới, cung cấp cùng các thuật toán ở ba mức **warp, block và
device**, nên bạn có thể nhét `cub::BlockReduce` vào giữa kernel của mình thay vì
gọi một lệnh launch riêng. Thrust được xây trên nó.

**Quy ước gọi hai lần của CUB** — nó không bao giờ cấp phát hộ bạn:

```cuda
cub::DeviceReduce::Sum(nullptr, tempBytes, ...);   // hỏi kích thước
cudaMalloc(&d_temp, tempBytes);                    // cấp phát
cub::DeviceReduce::Sum(d_temp, tempBytes, ...);    // chạy
```

Điều này là có chủ ý: một buffer nháp được dùng lại cho nhiều lời gọi thay vì cấp
phát trong vòng lặp nóng.

## Nhiệm vụ của bạn

Mang bản reduction **v7** từ Giai đoạn 3/06 sang, rồi so với
`cub::DeviceReduce::Sum` và `thrust::reduce`. Báo cáo **phần trăm băng thông đỉnh**,
không phải mili-giây. Sau đó chạy `cub::DeviceScan` và `thrust::sort`.

**Hãy dự đoán kernel của bạn đứng ở đâu trước khi nhìn kết quả.**

## Build và chạy

```bash
cmake --build build --target p5_01_thrust_and_cub -j
./build/bin/p5/p5_01_thrust_and_cub
```

## Kết quả mong đợi

RTX 3060, 16 triệu float:

```
hand-written (Phase 3/06 v7)          0.202    332.67 GB/s     1.00x
cub::DeviceReduce::Sum                0.203    331.25 GB/s     1.00x
thrust::reduce                        0.220    304.77 GB/s     0.92x

  As a fraction of peak bandwidth:
    hand-written : 92%     CUB : 92%     Thrust : 85%

cub::DeviceScan                       0.413    325.24 GB/s
thrust::inclusive_scan                0.423    317.29 GB/s

thrust::sort : 3132 million keys per second
```

**Kernel Giai đoạn 3 của bạn ngang bằng CUB.** Đó là phần thưởng cho việc đã viết
nó bảy lần — giờ bạn biết "ngang thư viện" trông như thế nào, và nhận ra được điều
đó trong mã của chính mình.

Con số 85% của Thrust là cái giá cho tính tổng quát của nó; CUB là thứ bạn dùng khi
7% đó có ý nghĩa.

## Điểm cốt lõi

- **Một kernel tự viết tốt có thể ngang CUB** với một phép toán nghẽn bộ nhớ đơn
  giản — và giờ bạn biết điều đó từ số đo, không phải từ niềm tin.
- **Scan không hề tuần tự.** Thuật toán decoupled look-back một lượt của CUB đạt gần
  băng thông copy; tự viết nó là dự án nhiều tuần.
- **Sắp xếp là chỗ tự viết rõ ràng là sai.** Một dòng so với một dự án nghiêm túc.
- **Thrust để rõ ràng, CUB để kiểm soát, tự viết chỉ sau khi đã đo cả hai.**

## Đi xa hơn

- Dùng `cub::BlockReduce` *bên trong* một kernel Giai đoạn 3 của bạn.
- Sắp xếp cặp khoá-giá trị bằng `thrust::sort_by_key`.
- Thử `thrust::transform_reduce` với functor tự viết — một lượt thay vì hai.
- So `cub::DeviceRadixSort` trực tiếp với `thrust::sort`.
