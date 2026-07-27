<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 03 - Atomic, tranh chấp và trật tự bộ nhớ

> **Giai đoạn 4 · Nâng cao** | Độ khó: ⭐⭐⭐⭐⭐ | Thời gian: ~3 giờ | Yêu cầu: [GĐ 3/07](../../../phase-3-intermediate/exercises/07-warp-shuffle/README.vi.md) | **Cần GPU**
>
> ⚠️ **Chưa kiểm chứng trên máy tham chiếu** — không có số đo bên dưới. Hãy chạy và
> tự điền số của bạn.

## Mục tiêu

Ba chủ đề, khó dần: **tranh chấp**, **tự dựng atomic chưa có sẵn**, và **trật tự bộ
nhớ** — nơi một kernel có thể đúng trên GPU của bạn và sai trên GPU của người khác.

## Kiến thức nền

**1. Tranh chấp.** `atomicAdd` lên một địa chỉ toàn cục từ mọi thread sẽ bị tuần tự
hoá. Cách sửa là **riêng tư hoá**: mỗi block giữ bộ đếm riêng trong shared memory,
rồi gộp lại một lần. Số atomic toàn cục giảm từ một trên mỗi *phần tử* xuống vài
cái trên mỗi *block*. Đây là kỹ thuật atomic hữu ích nhất.

**2. `atomicCAS`.** Chỉ có ít phép atomic tồn tại trong phần cứng. Cực đại kiểu
float, phép gộp tuỳ biến — mọi thứ khác đều là vòng lặp compare-and-swap:

```cuda
do { assumed = old;
     old = atomicCAS(addr, assumed, f(assumed)); }
while (assumed != old);
```

Đúng, nhưng nó phải thử lại khi có tranh chấp. Hãy rút gọn trong block trước.

**3. Trật tự bộ nhớ.** **Tính nguyên tử không phải tính hiển thị.** `atomicAdd` đảm
bảo bộ đếm đúng; nó *không* nói gì về việc dữ liệu bạn vừa ghi ngay trước đó có
hiển thị với thread đọc bộ đếm sau đó hay không. `__threadfence()` mới là thứ sắp
xếp trật tự giữa chúng.

| | Phạm vi |
|---|---|
| `__threadfence_block()` | trong phạm vi block |
| `__threadfence()` | toàn thiết bị |
| `__threadfence_system()` | cả host và các GPU ngang hàng |

## Nhiệm vụ của bạn

Mở `main.cu`: `histogramGlobal`, `histogramPrivatised`, `atomicMaxFloat` +
`maxKernel`, và `singlePassReduce`.

Với cái cuối: hãy viết **có** fence trước, rồi xoá fence đi và chạy vài trăm lần.
Gần như chắc chắn nó vẫn cho kết quả đúng — **và đó chính xác là lý do lỗi này đắt
đỏ.** Nó sẽ lộ ra trên phần cứng của người khác, vài tháng sau, khi tải nặng.

## Build và chạy

```bash
cmake --build build --target p4_03_atomics_and_ordering -j
./build/bin/p4/p4_03_atomics_and_ordering
compute-sanitizer --tool racecheck ./build/bin/p4/p4_03_atomics_and_ordering
```

## Điểm cốt lõi

- **Hãy riêng tư hoá trước khi tối ưu bất cứ điều gì khác về một atomic.**
- **`atomicCAS` dựng được mọi atomic bạn cần**, với cái giá là một vòng thử lại.
- **Nguyên tử ≠ hiển thị.** Fence không phải tuỳ chọn, và bỏ nó đi thường vẫn cho
  kết quả đúng — đó chính là cái bẫy.
- Lỗi trật tự bộ nhớ không thể bắt được bằng cách test trên một máy.
