<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 02 - Shared memory và `__syncthreads`

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [01](../01-memory-coalescing/README.vi.md) | **Cần GPU**

## Mục tiêu

Học công cụ mà toàn bộ phần còn lại của Giai đoạn 3 dựa vào — một vùng nhớ nháp
nhỏ, nhanh, riêng cho từng block, do bạn tự quản lý — kể cả rào chắn đồng bộ mà nó
đòi hỏi và cái giá về occupancy mà nó lấy đi.

## Kiến thức nền

Shared memory là vùng nhớ trên chip, riêng cho mỗi thread block. Độ trễ **thấp hơn
bộ nhớ toàn cục khoảng 100 lần**, và nó rất ít: mặc định 48 KB mỗi block, 100 KB
mỗi SM trên RTX 3060, chia cho mọi block đang thường trú ở đó.

Nó giải quyết đúng hai vấn đề, và mọi bài sau trong Giai đoạn 3 đều là một trong
hai:

| Vấn đề | Cách giải | Ở bài nào |
|---|---|---|
| **Tái sử dụng** — nhiều thread cần cùng một giá trị | đọc từ global một lần, rồi phục vụ từ shared | tiled matmul (04), tích chập (14), reduction (06) |
| **Sắp xếp lại** — một truy cập không thể coalesce | dàn vào shared, nơi quy tắc coalescing không áp dụng, rồi sắp xếp lại ở đó | chuyển vị (05), histogram (09) |

**`__syncthreads()`** là một rào chắn: không thread nào đi qua cho tới khi mọi
thread đã tới, và mọi phép ghi shared memory phát ra trước đó đều hiện rõ với mọi
thread sau đó.

### Hai quy tắc hay làm người ta vấp

1. **Mọi thread trong block phải chạm tới mọi `__syncthreads()`.** Đặt một cái bên
   trong `if (tid < 100)` với block 256 thread là hành vi không xác định — thường
   là treo.
2. **Từ Volta trở đi, các thread trong một warp *không* được đồng bộ ngầm.** Mã bỏ
   qua rào chắn mà vẫn "chạy được" trên Kepler nhờ warp đi đồng bộ nay đã sai trên
   mọi GPU hiện đại. Nếu bạn thật sự muốn đồng bộ mức warp, hãy viết `__syncwarp()`.

**Tĩnh so với động.** `__shared__ float tile[256]` cố định kích thước lúc biên
dịch. `extern __shared__ float tile[]` cộng với `kernel<<<grid, block, bytes>>>`
biến nó thành tham số lúc launch — dùng cách này khi kích thước tile là một núm
tinh chỉnh lúc chạy.

## Nhiệm vụ của bạn

Mở `main.cu` và viết sáu kernel:

1. **`reverseGlobal`** — không dùng shared memory. Mốc cơ sở này tồn tại để nói một
   điều: shared memory không phải lúc nào cũng là câu trả lời.
2. **`reverseShared`** — dàn qua một tile.
3. **`reverseSharedBroken`** — vẫn thế nhưng xoá rào chắn. **Hãy dự đoán bao nhiêu
   phần trăm phần tử sẽ sai trước khi chạy.**
4. **`reverseSharedDynamic`** — shared memory động.
5. **`broadcastGlobal` / `broadcastShared`** — mẫu tái sử dụng. Dự đoán mức tăng
   tốc từ số lần đọc, rồi mới đo.

Sau đó lập bảng số block mỗi SM theo lượng shared memory mỗi block để thấy cái giá
về occupancy.

## Build và chạy

```bash
cmake --build build --target p3_02_shared_memory_basics -j
./build/bin/p3/p3_02_shared_memory_basics
compute-sanitizer --tool racecheck ./build/bin/p3/p3_02_shared_memory_basics
```

## Kết quả mong đợi

RTX 3060, 4 triệu float, block 256 thread:

```
--- Correctness ---
  [PASS] reverse via global memory
  [PASS] reverse via shared memory
  [PASS] reverse via dynamic shared memory

--- What happens without __syncthreads() ---
  2001632 of 4194304 elements wrong (47.7%)

--- Reuse ---
Variant                           Time (ms)   Speedup
------------------------------------------------------
global memory                         1.336     1.00x
shared memory                         0.583     2.29x

  Global reads per output element
    naive  : 257      shared : 1      ratio : 257x fewer reads
    measured : 2.29x faster
```

### Hai điều đáng ngẫm

**47.7% sai.** Không phải hỏng hóc tinh vi — gần nửa mảng. Thread `tid` đọc
`tile[count-1-tid]`, do một thread *khác* ghi, rất có thể thuộc một warp còn chưa
chạy. Không có rào chắn thì giữa chúng chẳng có thứ tự nào cả.

**Ít hơn 257 lần đọc, nhanh hơn 2.29 lần.** Khoảng chênh là do L1: block vừa cache,
nên phần lặp lại của bản naive phần lớn được hấp thụ. Shared memory vẫn thắng vì nó
**tường minh** — chắc chắn nằm trên chip, chắc chắn không bị lưu lượng của block
khác đẩy ra. Cache là một niềm hy vọng; shared memory là một lời hứa.

## Điểm cốt lõi

- **Shared memory dùng để tái sử dụng và để sắp xếp lại.** Nếu kernel của bạn không
  làm cả hai, nó sẽ không giúp gì.
- **Rào chắn không phải tuỳ chọn**, và thiếu một cái có thể cho kết quả *đúng* trên
  máy bạn nhưng sai ở nơi khác. Hãy dùng `compute-sanitizer --tool racecheck`.
- **Warp không được đồng bộ ngầm trên phần cứng hiện đại.**
- **Shared memory tốn occupancy.** Xin 48 KB mỗi block thì chỉ 2 block vừa một SM,
  còn lại rất ít warp để che độ trễ. Đánh đổi đó là nội dung
  [bài 15](../15-occupancy-tuning/).

## Đi xa hơn

- Chạy kernel hỏng với block 32 thread. Nó có còn sai không? Vì sao "chạy được" lại
  là kết cục nguy hiểm nhất ở đây?
- Thay `__syncthreads()` bằng `__syncwarp()` trong block 32 thread. Khi nào thì
  việc đó chính đáng?
- Quét kích thước block từ 32 tới 1024 cho kernel broadcast. Điểm tối ưu ở đâu, và
  ràng buộc nào quyết định nó?
- **Tiếp theo:** [03 bank-conflicts](../03-bank-conflicts/) — shared memory có bộ
  quy tắc truy cập riêng, và vi phạm chúng tốn tới 32 lần.
