# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)

## 1. Đo đạc hiệu năng (Performance Benchmark)
- **Tổng thời gian thực thi (Total Wall Time)**: `1.47s`
- **Thời gian trung bình mỗi publication**: `1.471s`
- **Thời gian trung bình mỗi version**: `1.471s`
- **RAM tối đa (Peak RSS Memory)**: `90.73 MB`
- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: `30.41 MB`

## 2. Tỉ lệ parse thành công & Version Coverage
| Chỉ số | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Tổng số publications | 1 | 100% |
| Tổng số versions được xử lý | 1 | 100% |
| Versions parse thành công | 1 | 100.0% |
| Versions rỗng (không có TeX source) | 0 | 0.0% |
| Versions lỗi (failed) | 0 | 0.0% |
| **Version Coverage (TeX Source)** | **1 / 1** | **100.0%** |

## 3. Thống kê Khử trùng lặp Elements (Deduplication)
- **Tổng elements trước dedup**: `174`
- **Tổng elements sau dedup**: `174`
- **Số lượng phần tử trùng lặp đã loại bỏ**: `0`
- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`0.0%`**

### Phân bố Elements theo Kind (Trước vs Sau Dedup)
| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |
|---|---|---|---|---|
| `document` | 1 | 1 | 0 | 0.0% |
| `figure` | 5 | 5 | 0 | 0.0% |
| `item` | 5 | 5 | 0 | 0.0% |
| `itemize` | 1 | 1 | 0 | 0.0% |
| `section` | 8 | 8 | 0 | 0.0% |
| `sentence` | 144 | 144 | 0 | 0.0% |
| `subsection` | 5 | 5 | 0 | 0.0% |
| `subsubsection` | 5 | 5 | 0 | 0.0% |