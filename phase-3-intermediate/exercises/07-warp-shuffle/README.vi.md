<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 07 - Các nguyên thuỷ mức warp

> **Giai đoạn 3 · Trung cấp** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [06](../06-reduction-variants/README.vi.md) | **Cần GPU**

## Mục tiêu

Trao đổi giá trị trực tiếp giữa thanh ghi của các thread — không shared memory,
không rào chắn, không lưu lượng bộ nhớ. Rồi phát hiện rằng ứng dụng nổi tiếng nhất
của những nguyên thuỷ này **không mang lại gì trên trình biên dịch hiện đại**, và
đó lại là bài học hữu ích hơn bản thân kỹ thuật.

## Kiến thức nền

| Intrinsic | Tác dụng |
|---|---|
| `__shfl_sync(mask, v, lane)` | đọc `v` của lane `lane` |
| `__shfl_up_sync(mask, v, d)` | đọc của lane `tôi − d` |
| `__shfl_down_sync(mask, v, d)` | đọc của lane `tôi + d` |
| `__shfl_xor_sync(mask, v, m)` | đọc của lane `tôi XOR m` |
| `__ballot_sync(mask, p)` | một word 32 bit, mỗi lane một bit |
| `__all_sync` / `__any_sync` | `p` có đúng với mọi lane / lane nào không |
| `__activemask()` | những lane nào đang hoạt động |

**Hậu tố `_sync` và tham số mask không phải để trang trí.** Trước Volta, warp luôn
chạy đồng bộ và các intrinsic cũ không cần mask. Từ Volta, thread phân kỳ độc lập,
nên bạn phải khai báo những lane nào được kỳ vọng tham gia. Dùng `0xffffffff` khi
chắc chắn cả warp đang hoạt động, và `__activemask()` khi có thể không — ví dụ
trong grid-stride loop mà vòng cuối bỏ lại vài lane. **Một mask không khớp thực tế
là hành vi không xác định, và triệu chứng là kết quả sai chỉ xuất hiện khi tải nặng.**

**`__shfl_xor_sync` so với `__shfl_down_sync`.** Cả hai rút gọn trong 5 bước. `down`
để kết quả lại chỉ ở lane 0; `xor` (kiểu cánh bướm) để kết quả ở *mọi* lane. Cùng
chi phí, mà khỏi cần `if (lane == 0)` phía sau.

## Nhiệm vụ của bạn

Mở `main.cu` và viết: `warpAllReduceSum`, `warpInclusiveScan`, `blockReduceKernel`,
`voteDemo`, `filterNaive`, `filterWarpAggregated`.

Với cái cuối, **hãy dự đoán mức tăng tốc, đo nó, rồi — dù con số là bao nhiêu —
hãy dịch ngược kernel *naive***:

```bash
cuobjdump -sass build/bin/p3/p3_07_warp_shuffle | grep -B3 RED
```

Câu trả lời cho "vì sao tôi lại được con số đó" nằm trong đấy.

## Build và chạy

```bash
cmake --build build --target p3_07_warp_shuffle -j
./build/bin/p3/p3_07_warp_shuffle
```

## Kết quả mong đợi

RTX 3060, 16 triệu float, tỉ lệ lọt 25%:

```
  [PASS] inclusive scan within each warp
  [PASS] block reduction is correct
  [PASS] __all_sync / __any_sync / __ballot_sync + __popc

--- Warp-aggregated atomics ---
Counting elements that pass a filter
one atomic per passing thread         0.377    178.09 GB/s     1.00x
one atomic per warp (ballot)          0.390    172.01 GB/s     0.97x
```

### 0.97 lần. Không lời gì cả. Đây là lý do

Dịch ngược `filterNaive` và bạn sẽ thấy:

```
REDUX.SUM UR10, R4                ← cộng toàn warp, một lệnh duy nhất
ISETP.EQ.U32.AND P1, PT, ...      ← chọn ra một lane
@P1 RED.E.ADD.STRONG.GPU [...]    ← MỘT atomic cho cả warp
```

**nvcc đã tự thực hiện việc gộp đó rồi.** Nó làm vậy từ CUDA 9, và trên Ampere nó
dùng lệnh phần cứng `REDUX.SUM` — vốn *nhanh hơn* chuỗi `__ballot_sync` + `__popc`
mà bạn tự viết. Bản viết tay không sai; nó thừa.

> **Bài học không phải là "hãy gộp các atomic lại". Mà là: hãy đọc SASS trước khi
> tối ưu thủ công.** Một kỹ thuật nổi tiếng từ bài blog năm 2017 có thể đã thành mã
> chết vào năm 2020 vì trình biên dịch hấp thụ mất rồi. Kiểm tra chỉ tốn một câu lệnh.

**Gộp thủ công vẫn còn giá trị khi:**

- trình biên dịch không nhìn ra mẫu — số gia phụ thuộc dữ liệu, hoặc atomic nấp sau
  một lời gọi hàm
- bạn cần **vị trí** của từng lane chứ không chỉ tổng. Một phép scan trên ballot cho
  mỗi thread lọt qua một ô đầu ra riêng chỉ từ một atomic — đúng cách
  [stream compaction](../12-stream-compaction/) hoạt động
- bạn đang gộp thứ gì đó không phải một bộ đếm

## Điểm cốt lõi

- **Shuffle chuyển dữ liệu giữa các thanh ghi.** Không shared memory, không rào
  chắn, không lưu lượng bộ nhớ — 5 lệnh cho một phép reduction hoặc scan trong warp.
- **`xor` cho mọi lane kết quả; `down` chỉ cho lane 0.** Chọn theo việc bạn cần làm
  gì tiếp theo.
- **Luôn truyền mask đúng.** Dùng `__activemask()` khi warp có thể không đầy.
- **Hãy xác minh rằng một phép tối ưu vẫn còn cần thiết trước khi áp dụng nó.**
  Trình biên dịch là mục tiêu di động, và `cuobjdump -sass` là cách bạn kiểm tra.

## Đi xa hơn

- Cài một phép scan mức warp trên mask ballot (`__popc(mask & lanemask_lt())`) để
  mỗi thread lọt qua có một chỉ số đầu ra riêng. Đó là lõi của bài 12.
- Chạy lại bộ lọc với tỉ lệ lọt 100% và 1%. Thứ hạng có đổi không?
- Dùng thẳng `__reduce_add_sync` (sm_80+) rồi so với bản butterfly tự viết.
- Đọc [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
  — và để ý ngày đăng của nó khi đọc.
