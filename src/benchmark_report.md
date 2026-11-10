# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)

## 1. Đo đạc hiệu năng (Performance Benchmark)
- **Tổng thời gian thực thi (Total Wall Time)**: `13.26s`
- **Thời gian trung bình mỗi publication**: `1.326s`
- **Thời gian trung bình mỗi version**: `0.663s`
- **RAM tối đa (Peak RSS Memory)**: `95.67 MB`
- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: `32.26 MB`

## 2. Tỉ lệ parse thành công & Version Coverage
| Chỉ số | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Tổng số publications | 10 | 100% |
| Tổng số versions được xử lý | 20 | 100% |
| Versions parse thành công | 18 | 90.0% |
| Versions rỗng (không có TeX source) | 2 | 10.0% |
| Versions lỗi (failed) | 0 | 0.0% |
| **Version Coverage (TeX Source)** | **18 / 20** | **90.0%** |

## 3. Thống kê Khử trùng lặp Elements (Deduplication)
- **Tổng elements trước dedup**: `6491`
- **Tổng elements sau dedup**: `3364`
- **Số lượng phần tử trùng lặp đã loại bỏ**: `3127`
- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`48.17%`**

### Phân bố Elements theo Kind (Trước vs Sau Dedup)
| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |
|---|---|---|---|---|
| `block_formula` | 659 | 287 | 372 | 56.45% |
| `document` | 18 | 18 | 0 | 0.0% |
| `figure` | 171 | 104 | 67 | 39.18% |
| `item` | 447 | 247 | 200 | 44.74% |
| `itemize` | 123 | 8 | 115 | 93.5% |
| `paragraph` | 39 | 31 | 8 | 20.51% |
| `section` | 168 | 85 | 83 | 49.4% |
| `sentence` | 4714 | 2495 | 2219 | 47.07% |
| `subsection` | 141 | 78 | 63 | 44.68% |
| `subsubsection` | 11 | 11 | 0 | 0.0% |