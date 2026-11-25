# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)

## 1. Đo đạc hiệu năng (Performance Benchmark)
- **Tổng thời gian thực thi (Total Wall Time)**: `25.12s`
- **Thời gian trung bình mỗi publication**: `1.256s`
- **Thời gian trung bình mỗi version**: `0.739s`
- **RAM tối đa / trung bình (RSS, 119 mẫu)**: `95.39 MB` / `88.2 MB`
- **Dung lượng output (elements/ + bibtex/)**: `24.30 MB`

## 2. Tỉ lệ parse thành công & Version Coverage
| Chỉ số | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Tổng số publications | 20 | 100% |
| Tổng số versions được xử lý | 34 | 100% |
| Versions parse đầy đủ (success) | 34 | 100.0% |
| Versions parse một phần (partial) | 0 | 0.0% |
| Versions rỗng (không có TeX source) | 0 | 0.0% |
| Versions lỗi (failed) | 0 | 0.0% |
| **Tỉ lệ parse thành công trên versions có TeX** | **34** | **100.0%** |
| **Version Coverage (TeX Source)** | **34 / 34** | **100.0%** |

## 3. Thống kê Khử trùng lặp Elements (Deduplication)
- **Tổng elements trước dedup**: `12304`
- **Tổng elements sau dedup**: `9098`
- **Số lượng phần tử trùng lặp đã loại bỏ**: `3206`
- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`26.06%`**

### Phân bố Elements theo Kind (Trước vs Sau Dedup)
| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |
|---|---|---|---|---|
| `block_formula` | 737 | 539 | 198 | 26.87% |
| `document` | 34 | 34 | 0 | 0.0% |
| `figure` | 423 | 343 | 80 | 18.91% |
| `item` | 616 | 493 | 123 | 19.97% |
| `itemize` | 149 | 116 | 33 | 22.15% |
| `paragraph` | 188 | 122 | 66 | 35.11% |
| `section` | 318 | 206 | 112 | 35.22% |
| `sentence` | 9397 | 6905 | 2492 | 26.52% |
| `subsection` | 358 | 268 | 90 | 25.14% |
| `subsubsection` | 84 | 72 | 12 | 14.29% |

## 4. Thống kê BibTeX
- **Entries trước / sau dedup**: `1788` / `1015` (giảm `43.23%`)
- **Citation key được đổi tên**: `4`
- **Entries được bổ sung trường khi gộp**: `41`
- **Entries trong .bib không được cite (đã loại)**: `1480`