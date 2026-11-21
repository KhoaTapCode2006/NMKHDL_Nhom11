# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)

## 1. Đo đạc hiệu năng (Performance Benchmark)
- **Tổng thời gian thực thi (Total Wall Time)**: `19.52s`
- **Thời gian trung bình mỗi publication**: `0.976s`
- **Thời gian trung bình mỗi version**: `0.651s`
- **RAM tối đa (Peak RSS Memory)**: `101.22 MB`
- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: `35.52 MB`

## 2. Tỉ lệ parse thành công & Version Coverage
| Chỉ số | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Tổng số publications | 20 | 100% |
| Tổng số versions được xử lý | 30 | 100% |
| Versions parse thành công | 24 | 80.0% |
| Versions rỗng (không có TeX source) | 2 | 6.67% |
| Versions lỗi (failed) | 4 | 13.33% |
| **Version Coverage (TeX Source)** | **24 / 30** | **80.0%** |

## 3. Thống kê Khử trùng lặp Elements (Deduplication)
- **Tổng elements trước dedup**: `7611`
- **Tổng elements sau dedup**: `6377`
- **Số lượng phần tử trùng lặp đã loại bỏ**: `1234`
- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`16.21%`**

### Phân bố Elements theo Kind (Trước vs Sau Dedup)
| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |
|---|---|---|---|---|
| `block_formula` | 447 | 347 | 100 | 22.37% |
| `document` | 24 | 24 | 0 | 0.0% |
| `figure` | 192 | 173 | 19 | 9.9% |
| `item` | 202 | 178 | 24 | 11.88% |
| `itemize` | 59 | 53 | 6 | 10.17% |
| `paragraph` | 41 | 26 | 15 | 36.59% |
| `section` | 189 | 138 | 51 | 26.98% |
| `sentence` | 6113 | 5198 | 915 | 14.97% |
| `subsection` | 260 | 159 | 101 | 38.85% |
| `subsubsection` | 84 | 81 | 3 | 3.57% |