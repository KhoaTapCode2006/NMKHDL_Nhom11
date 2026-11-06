# BÁO CÁO NHIỆM VỤ MILESTONE 2: MULTI-FILE GATHERING & STATISTIC
**Người thực hiện:** Hoàng  
**Nhánh Git:** `feature/hoang-latex-parser`  
**Dự án:** Nhập môn Khoa học Dữ liệu — Nhóm 11  

---

## 1. TỔNG QUAN CÔNG VIỆC ĐÃ HOÀN THÀNH

Theo phân công trong Milestone 2, phần việc của Hoàng phụ trách:
1. **Multi-File Gathering (Gom đa file)**: Tự động phát hiện file gốc (root) chứa `\documentclass`, đi theo các lệnh `\input{...}` và `\include{...}` đệ quy để gom đúng luồng biên dịch chính (*compilation path*), bỏ qua các file nháp/file rác không thuộc bài báo.
2. **Hierarchy Parsing & Content Extraction**: Bóc tách nội dung toàn văn sạch từ LaTeX thành các phần tử nguyên tử (**Elements**: câu văn xuôi `sentence`, công thức toán `block_formula`, hình ảnh/bảng `figure`, danh sách `item`, tiêu đề `section`) và xây dựng cây cấu trúc (**Hierarchy**).
3. **Cross-Version Deduplication (Khử trùng lặp)**: So khớp và gộp các element giống hệt nhau giữa các version (v1, v2, v3,...) về chung một `element_id` duy nhất, tiết kiệm dung lượng lưu trữ.
4. **Hỗ trợ đa định dạng xuất (3 định dạng song song)**: Tự động lưu bảng dữ liệu dưới cả **JSON**, **CSV**, và **Parquet** trong thư mục `elements/` của từng bài báo.
5. **Đo đạc hiệu năng & Thống kê (Statistic & Benchmarking)**: Đo thời gian chạy, đỉnh RAM (RSS Memory), tỉ lệ parse thành công, version coverage và tỉ lệ giảm trùng lặp.

> **Cam kết mã nguồn:** Toàn bộ code cũ của Milestone 1 (`arxiv_crawler.py`, `oai_crawler.py`) được **giữ nguyên vẹn 100%**, không chỉnh sửa bất kỳ dòng nào. Tất cả tính năng mới được module hóa độc lập trong thư mục `src/parser/`.

---

## 2. KẾT QUẢ THỰC NGHIỆM TRÊN TẬP DỮ LIỆU (50 PUBLICATIONS, 83 VERSIONS)

Pipeline đã chạy thực nghiệm thành công trên toàn bộ 50 bài báo được phân công (`2411.13799` đến `2411.13848`):

### ⏱️ Hiệu năng thực thi (Benchmarking)
- **Tổng thời gian xử lý toàn bộ tập dữ liệu**: **`17.19 giây`**
- **Thời gian trung bình mỗi bài báo**: **`0.344 giây` / publication**
- **Thời gian trung bình mỗi version**: **`0.207 giây` / version**
- **Đỉnh bộ nhớ RAM (Peak RSS Memory)**: **`29.59 MB`** *(cực kỳ nhẹ, hoàn toàn tương thích và tối ưu trên Google Colab CPU)*
- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: **`4.60 MB`**

### 🎯 Tỉ lệ thành công & Version Coverage
- **Tổng số ấn phẩm (Publications)**: `50`
- **Tổng số versions được xử lý**: `83`
- **Versions có TeX parse thành công**: **`61 / 61` (đạt tỉ lệ 100%, 0 lỗi)**
- **Versions rỗng (không có TeX trên arXiv)**: **`22 / 83` (`26.51%`)** *(Được ghi nhận đúng theo quy định của đề để tính coverage)*
- **Version Coverage (Độ phủ nguồn TeX)**: **`73.49%`**

### 🔄 Hiệu quả khử trùng lặp (Deduplication)
- **Tổng số elements trước khi khử trùng lặp**: `23,635` elements
- **Tổng số elements sau khi khử trùng lặp**: `17,085` elements
- **Số phần tử trùng lặp xuyên version đã loại bỏ**: **`6,550` elements**
- **Tỉ lệ giảm trùng lặp toàn bài (Reduction Rate)**: **`27.71%`** *(nhiều bài 4 version như 2411.13805 giảm đến **63.32%**)*

#### Bảng phân bố Elements trước và sau khi Dedup:
| Loại phần tử (`kind`) | Trước Dedup | Sau Dedup | Đã loại bỏ trùng | Tỉ lệ giảm |
|---|---|---|---|---|
| `sentence` (câu văn) | 18,312 | 13,424 | 4,888 | 26.69% |
| `block_formula` (toán khối) | 2,232 | 1,560 | 672 | 30.11% |
| `figure` (hình/bảng) | 563 | 423 | 140 | 24.87% |
| `item` (dòng liệt kê) | 950 | 697 | 253 | 26.63% |
| `section` (tiêu đề mục) | 489 | 334 | 155 | 31.70% |
| `subsection` | 552 | 392 | 160 | 28.99% |
| `document` (gốc bài báo) | 61 | 61 | 0 | 0.0% *(mỗi version giữ 1 root)* |

---

## 3. CẤU TRÚC THƯ MỤC VÀ DỮ LIỆU ĐẦU RA

### Cấu trúc mã nguồn mới trong `NMKHDL_Nhom11/src/`:
```text
src/
├── parser/
│   ├── __init__.py
│   ├── gatherer.py      # Tìm root .tex, phân giải đệ quy \input/\include, dọn dẹp file
│   ├── extractor.py     # Chuẩn hóa toán, clean macro rác, tách sentence/formula/figure
│   ├── dedup.py         # Khử trùng lặp nội dung xuyên version, ánh xạ cây hierarchy
│   ├── benchmark.py     # Đo thời gian, RAM, coverage, sinh báo cáo Markdown & JSON
│   └── pipeline.py      # Điều phối toàn bộ quy trình, xuất JSON/CSV/Parquet
├── run_parser.py        # CLI chính để chạy gom file, parse và sinh thống kê
├── verify_output.py     # Script kiểm chứng kết quả nhanh cho các thành viên
├── benchmark_report.md  # Báo cáo Markdown sinh tự động phục vụ dựng Jupyter Book
└── benchmark_stats.json # Bảng số liệu thống kê chi tiết dạng JSON
```

### Cấu trúc dữ liệu trong thư mục của từng Publication (`.../<arxiv_id>/elements/`):
Mỗi publication có thêm một thư mục `elements/` chứa:
- `elements.json` & `elements.csv` & `elements.parquet`: Bảng các phần tử duy nhất gồm `element_id`, `arxiv_id`, `kind`, `content`.
- `hierarchy.json` & `hierarchy.csv` & `hierarchy.parquet`: Cây phả hệ gồm `arxiv_id`, `version`, `element_id`, `parent_id`.
- `compilation_manifest.json`: Ghi nhận danh sách file gốc, thứ tự các file `.tex` đã gom, và file mồ côi bị loại bỏ.
- `full_inlined_vX.tex`: Mã nguồn LaTeX đã được gom hợp nhất phẳng cho từng version (đã gỡ hết `\input`/`\include` dư thừa).

---

## 4. HƯỚNG DẪN DÀNH CHO NHÓM TRƯỞNG VÀ CÁC THÀNH VIÊN KHÁC ĐỂ CHẠY THỬ

Mọi thành viên trong nhóm đều có thể chạy thử trực tiếp trên máy cục bộ hoặc Google Colab theo các bước sau:

### Bước 1: Mở terminal tại thư mục dự án
```powershell
cd C:\arxiv_project\NMKHDL_Nhom11
```

### Bước 2: Kiểm chứng nhanh kết quả đã parse (Không cần chạy lại toàn bộ)
Chạy script kiểm tra mẫu dữ liệu, xem cấu trúc cây và bằng chứng khử trùng lặp:
```powershell
python src/verify_output.py
```

### Bước 3: Chạy lại toàn bộ Parser Pipeline (Nếu muốn)
- **Lựa chọn 1: Chạy và giữ nguyên file `.tex` gốc (không xóa)**:
  ```powershell
  python src/run_parser.py --no-clean-tex
  ```
- **Lựa chọn 2: Chạy và xóa các file `.tex` thừa/nháp** (theo đúng yêu cầu dọn dẹp của đề Milestone 2):
  ```powershell
  python src/run_parser.py --clean-tex
  ```
- **Lựa chọn 3: Chạy thử trên 5 bài đầu tiên để kiểm tra nhanh**:
  ```powershell
  python src/run_parser.py --limit 5 --no-clean-tex
  ```

---
*Báo cáo được hoàn thành vào ngày 06/11/2026 bởi Hoàng (NMKHDL - Nhóm 11).*
