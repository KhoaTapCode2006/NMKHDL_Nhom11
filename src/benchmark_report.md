# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)

## 1. Đo đạc hiệu năng (Performance Benchmark)
- **Tổng thời gian thực thi (Total Wall Time)**: `13.48s`
- **Thời gian trung bình mỗi publication**: `0.674s`
- **Thời gian trung bình mỗi version**: `0.449s`
- **RAM tối đa (Peak RSS Memory)**: `102.52 MB`
- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: `36.13 MB`

## 2. Tỉ lệ parse thành công & Version Coverage
| Chỉ số | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Tổng số publications | 20 | 100% |
| Tổng số versions được xử lý | 30 | 100% |
| Versions parse thành công | 23 | 76.67% |
| Versions rỗng (không có TeX source) | 3 | 10.0% |
| Versions lỗi (failed) | 4 | 13.33% |
| **Version Coverage (TeX Source)** | **23 / 30** | **76.67%** |

## 3. Thống kê Khử trùng lặp Elements (Deduplication)
- **Tổng elements trước dedup**: `7168`
- **Tổng elements sau dedup**: `5934`
- **Số lượng phần tử trùng lặp đã loại bỏ**: `1234`
- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`17.22%`**

### Phân bố Elements theo Kind (Trước vs Sau Dedup)
| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |
|---|---|---|---|---|
| `block_formula` | 437 | 337 | 100 | 22.88% |
| `document` | 23 | 23 | 0 | 0.0% |
| `figure` | 187 | 168 | 19 | 10.16% |
| `item` | 202 | 178 | 24 | 11.88% |
| `itemize` | 59 | 53 | 6 | 10.17% |
| `paragraph` | 41 | 26 | 15 | 36.59% |
| `section` | 182 | 131 | 51 | 28.02% |
| `sentence` | 5715 | 4800 | 915 | 16.01% |
| `subsection` | 248 | 147 | 101 | 40.73% |
| `subsubsection` | 74 | 71 | 3 | 4.05% |
