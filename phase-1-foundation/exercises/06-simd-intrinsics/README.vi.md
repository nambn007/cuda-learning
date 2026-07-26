<!-- Ngôn ngữ: [English](README.md) | **Tiếng Việt** -->

# 06 - SIMD intrinsics: warp của CPU

> **Giai đoạn 1 · Nền tảng** | Độ khó: ⭐⭐⭐ | Thời gian: ~2 giờ | Yêu cầu: [05](../05-cache-and-bandwidth/)

## Mục tiêu

Tự tay viết mã AVX2 và khám phá, trước cả khi bạn chạy kernel đầu tiên, hai quy
luật chi phối hiệu năng GPU: **công việc được thực hiện theo nhóm có bề rộng cố
định**, và **thực thi rộng hơn chỉ giúp ích khi bạn không bị nghẽn bộ nhớ**.

## Kiến thức nền

Một thanh ghi AVX2 rộng 256 bit — **8 float** — và một lệnh tác động lên cả 8 lane.
Một **warp** CUDA gồm **32 thread** cùng phát ra một lệnh.

| | SIMD (AVX2) | SIMT (warp CUDA) |
|---|---|---|
| Số lane | 8 | 32 |
| Ai viết mã vector | bạn, bằng intrinsics | không ai cả — bạn viết mã vô hướng |
| Phần dư | vòng lặp đuôi `n % 8` | warp không đầy / kiểm tra biên |
| Rẽ nhánh phân kỳ | mask thủ công (`_mm256_blendv_ps`) | predication phần cứng; cả hai nhánh vẫn chạy |
| Rút gọn (reduction) | cây `hadd` giữa các lane | cây `__shfl_down_sync` giữa các lane |

Ba kernel làm rõ điều đó:

- **SAXPY** (`y = a*x + y`) — 2 FLOP cho 12 byte. Nghẽn bộ nhớ nặng. AVX2 sẽ cho
  bạn **thấp hơn 8 lần rất nhiều**, vì thanh ghi rộng 8 lane không làm DRAM nhanh lên.
- **Đa thức bậc 15** (quy tắc Horner) — 30 FLOP cho 8 byte, cường độ ~3.75. Nghẽn
  tính toán, nên AVX2 đạt gần trọn 8 lần.
- **Tích vô hướng** — vừa nghẽn bộ nhớ *vừa* cần rút gọn theo chiều ngang, đúng
  cái cây log₂(số lane) mà bạn sẽ viết lại bằng `__shfl_down_sync` ở Giai đoạn 3.

Một chi tiết không hiển nhiên trong tích vô hướng: hãy dùng **bốn biến tích luỹ
độc lập**, không phải một. Lệnh FMA có độ trễ ~4 chu kỳ nhưng phát được mỗi chu
kỳ. Một biến tích luỹ duy nhất sẽ bị tuần tự hoá bởi chính chuỗi phụ thuộc của nó
và chỉ đạt một phần tư đỉnh. Đúng mẹo song song mức lệnh này xuất hiện lại trong
reduction trên GPU.

## Nhiệm vụ của bạn

Mở `main.cpp`. Các intrinsic bạn cần:

| Intrinsic | Ý nghĩa |
|---|---|
| `__m256` | 8 float |
| `_mm256_set1_ps(a)` | phát một float ra toàn bộ lane |
| `_mm256_loadu_ps(p)` | nạp 8 float (`u` = unaligned, luôn an toàn) |
| `_mm256_fmadd_ps(a,b,c)` | `a*b + c`, một lệnh, một lần làm tròn |
| `_mm256_storeu_ps(p,v)` | ghi 8 float |

1. **`saxpyAVX2()`** — vòng chính xử lý 8 phần tử, rồi vòng vô hướng cho `n % 8`.
2. **`dotAVX2()`** — biến tích luỹ dạng vector cộng với một bước rút gọn ngang
   (`_mm256_extractf128_ps` → `_mm_add_ps` → `_mm_hadd_ps` ×2 → `_mm_cvtss_f32`).
3. **`polyAVX2()`** — vòng Horner, mỗi lần 8 phần tử.
4. **Đo hiệu năng** cả ba, so với bản vô hướng.

Không cần cờ biên dịch đặc biệt: các hàm đã mang
`__attribute__((target("avx2,fma")))`, và `main()` kiểm tra
`__builtin_cpu_supports("avx2")` trước khi gọi.

## Build và chạy

```bash
cmake --build build --target p1_06_simd_intrinsics -j
./build/bin/p1/p1_06_simd_intrinsics
```

## Kết quả mong đợi

```
--- Memory-bound kernel: SAXPY ---
Variant                           Time (ms)      GB/s    GFLOP/s   Speedup
scalar                                 ~6.0      ~8.4       ~1.4      1.00x
AVX2 (8 lanes)                         ~5.6      ~9.0       ~1.5     ~1.07x     <- gần như không đổi

--- Compute-bound kernel: degree-15 polynomial ---
scalar                                ~14.0       ...      ~9.0      1.00x
AVX2 (8 lanes)                         ~2.0       ...     ~63.0      ~7x        <- gần như hoàn hảo
```

Nếu bản vô hướng của bạn vốn đã nhanh, `-O3` đã tự vector hoá nó rồi. Kiểm tra
bằng `g++ -O3 -fopt-info-vec`.

## Điểm cốt lõi

- **Bề rộng SIMD chỉ giúp mã nghẽn tính toán.** Với kernel nghẽn bộ nhớ thì bus là
  giới hạn và 8 lane chẳng thay đổi gì. Đây là lý do phổ biến nhất khiến một kernel
  "đã song song hoá" không nhanh lên — trên CPU *lẫn* GPU.
- **Vòng lặp đuôi là phần cứng bị bỏ phí.** Một warp có 3 thread hoạt động trên 32
  tốn đúng bằng một warp đầy.
- **Reduction cần một cây.** Chuỗi `hadd` ở đây giống hệt về cấu trúc với
  `__shfl_down_sync`.
- **Độ trễ đòi hỏi nhiều biến tích luỹ.** Thông lượng ≠ độ trễ; hãy giữ nhiều
  chuỗi độc lập cùng chạy.

## Đi xa hơn

- Thêm nhánh AVX-512 (`__m512`, 16 float) với một thuộc tính `target` thứ hai và
  điều phối lúc chạy. 16 lane có giúp gì cho SAXPY không?
- Cài phần đuôi bằng mask với `_mm256_maskload_ps` thay cho vòng lặp vô hướng.
- So sánh với `#pragma omp simd` — trình biên dịch tiến gần tới đâu?
- Đọc [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).
