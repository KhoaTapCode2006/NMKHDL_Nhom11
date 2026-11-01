"""Module thống kê dữ liệu cào bài báo arXiv phục vụ báo cáo Jupyter Book.

Tập trung vào 3 nhóm chỉ số cốt lõi theo yêu cầu của đồ án:
1. Kích thước trung bình (bytes) của bài báo trước và sau khi loại bỏ hình ảnh.
2. Số lượng references trung bình mỗi bài, tỉ lệ crawl metadata references thành công.
3. Đánh giá hiệu năng thực nghiệm (Performance Benchmarking): Thời gian chạy, RAM và Dung lượng lưu trữ.
Hỗ trợ so sánh trực tiếp giữa 2 phương pháp: arxiv_crawler vs oai_crawler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiết lập phong cách hiển thị biểu đồ đẹp, chuẩn khoa học
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})


def format_table(df: pd.DataFrame):
    """Định dạng bảng căn lề trái toàn bộ nội dung (trừ cột index)."""
    if not isinstance(df, pd.DataFrame):
        return df
    if df.empty:
        return df
    try:
        return df.style.set_properties(**{
            "text-align": "left",
            "padding-left": "8px",
            "padding-right": "8px",
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "left"), ("padding-left", "8px")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding-left", "8px")]},
        ])
    except Exception:
        return df


def collect_crawl_metrics(sources_dir: str | Path, crawler_name: str = "") -> pd.DataFrame:
    """Quét thư mục kết quả cào (sourcesArxiv hoặc sourcesOAI) và thu thập metrics.

    Args:
        sources_dir: Đường dẫn tới thư mục chứa kết quả cào.
        crawler_name: Tên phương pháp cào (vd: 'arxiv_crawler', 'oai_crawler').

    Returns:
        pd.DataFrame chứa thông tin chi tiết từng phiên bản bài báo.
    """
    sources_path = Path(sources_dir)
    if not sources_path.exists():
        print(f"[Cảnh báo] Thư mục '{sources_dir}' chưa tồn tại hoặc chưa có dữ liệu.")
        return pd.DataFrame()

    records = []
    # Duyệt qua các thư mục paper/<version>/metadata.json
    for metadata_file in sorted(sources_path.glob("*/v*/metadata.json")):
        version_dir = metadata_file.parent
        paper_dir = version_dir.parent
        paper_id = paper_dir.name
        version_str = version_dir.name

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"[Lỗi] Không đọc được {metadata_file}: {exc}")
            continue

        # Đọc references.json nếu có
        ref_file = version_dir / "references.json"
        references = None
        if ref_file.exists():
            try:
                with open(ref_file, "r", encoding="utf-8") as f:
                    references = json.load(f)
            except Exception:
                references = None

        # Kích thước
        raw_archive_bytes = meta.get("raw_archive_bytes", 0)
        total_uncompressed = meta.get("total_uncompressed_bytes", 0)
        filtered_uncompressed = meta.get("filtered_uncompressed_bytes", 0)

        # Nếu chưa ghi nhận trong metadata nhưng đã có file tex/bib trên đĩa
        if filtered_uncompressed == 0:
            for file_path in version_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {".tex", ".bib"}:
                    filtered_uncompressed += file_path.stat().st_size

        bytes_saved = max(0, total_uncompressed - filtered_uncompressed)
        reduction_pct = (bytes_saved / total_uncompressed * 100.0) if total_uncompressed > 0 else 0.0

        # References
        ref_status = meta.get("references_status")
        if ref_status is None:
            if references is not None and isinstance(references, list):
                ref_status = "success"
            else:
                ref_status = "failed" if ref_file.exists() else "unknown"

        ref_count = len(references) if isinstance(references, list) else meta.get("references_count", 0)
        arxiv_ref_count = 0
        if isinstance(references, list):
            for r in references:
                if isinstance(r, dict):
                    ext = r.get("externalIds")
                    if isinstance(ext, dict) and ext.get("ArXiv"):
                        arxiv_ref_count += 1

        records.append({
            "crawler": crawler_name or sources_path.name,
            "paper_id": paper_id,
            "version": version_str,
            "arxiv_id": meta.get("arxiv_id", f"{paper_id}{version_str}"),
            "title": meta.get("title", ""),
            "source_status": meta.get("source_status", "unknown"),
            "raw_archive_bytes": raw_archive_bytes,
            "total_uncompressed_bytes": total_uncompressed,
            "filtered_uncompressed_bytes": filtered_uncompressed,
            "bytes_saved": bytes_saved,
            "reduction_percent": reduction_pct,
            "references_status": ref_status,
            "references_count": ref_count,
            "arxiv_references_count": arxiv_ref_count,
        })

    df = pd.DataFrame(records)
    return df


# --------------------------------------------------------------------------- #
# Yêu cầu 1: Kích thước bài báo trước và sau khi loại bỏ hình ảnh
# --------------------------------------------------------------------------- #
def compute_paper_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán bảng thống kê chi tiết kích thước trước và sau khi lọc ảnh."""
    if df.empty or "total_uncompressed_bytes" not in df.columns:
        return pd.DataFrame()

    total_bytes = df["total_uncompressed_bytes"]
    filtered_bytes = df["filtered_uncompressed_bytes"]
    saved_bytes = df["bytes_saved"]
    pct_reduction = df["reduction_percent"]

    stats_dict = {
        "Chỉ số": [
            "Số lượng phiên bản bài báo (N)",
            "Kích thước trung bình trước khi lọc (Bytes)",
            "Kích thước trung bình trước khi lọc (KB)",
            "Kích thước trung vị trước khi lọc (KB)",
            "Kích thước trung bình sau khi lọc (Bytes)",
            "Kích thước trung bình sau khi lọc (KB)",
            "Kích thước trung vị sau khi lọc (KB)",
            "Dung lượng trung bình tiết kiệm được (KB)",
            "Tỷ lệ giảm trung bình (%)",
            "Tỷ lệ giảm trung vị (%)",
        ],
        "Giá trị": [
            f"{len(df):,}",
            f"{total_bytes.mean():,.2f}",
            f"{total_bytes.mean() / 1024:,.2f} KB",
            f"{total_bytes.median() / 1024:,.2f} KB",
            f"{filtered_bytes.mean():,.2f}",
            f"{filtered_bytes.mean() / 1024:,.2f} KB",
            f"{filtered_bytes.median() / 1024:,.2f} KB",
            f"{saved_bytes.mean() / 1024:,.2f} KB",
            f"{pct_reduction.mean():.2f}%",
            f"{pct_reduction.median():.2f}%",
        ],
    }
    return pd.DataFrame(stats_dict)


def plot_size_comparison(df: pd.DataFrame, title_suffix: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    """Vẽ biểu đồ cột nhóm thể hiện kích thước trung bình và trung vị trước vs sau khi lọc ảnh."""
    if df.empty:
        print("[Cảnh báo] DataFrame rỗng, không thể vẽ biểu đồ.")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    before_kb = df["total_uncompressed_bytes"] / 1024
    after_kb = df["filtered_uncompressed_bytes"] / 1024

    metrics = ["Trung bình (Mean)", "Trung vị (Median)"]
    before_vals = [before_kb.mean(), before_kb.median()]
    after_vals = [after_kb.mean(), after_kb.median()]

    x = np.arange(len(metrics))
    width = 0.35

    # Biểu đồ 2 cột ghép: Đỏ đậm (trước lọc) và Xanh dương đậm (sau lọc)
    rects1 = ax.bar(x - width / 2, before_vals, width, label="Trước khi lọc", color="#ff0000", alpha=0.90)
    rects2 = ax.bar(x + width / 2, after_vals, width, label="Sau khi lọc", color="#0d00ff", alpha=0.90)

    ax.set_ylabel("Kích thước (KB)", fontweight="bold")
    title = "Kích thước bài báo trước và sau khi loại bỏ hình ảnh"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True)

    # Gán nhãn giá trị trên từng cột
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:,.1f} KB",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:,.1f} KB",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    # Hiển thị % giảm
    avg_reduction = df["reduction_percent"].mean()
    ax.text(0.5, 0.90, f"Tiết kiệm trung bình: {avg_reduction:.1f}% dung lượng",
            transform=ax.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffd42a", ec="none", alpha=0.3),
            fontweight="bold")

    return ax


# --------------------------------------------------------------------------- #
# Yêu cầu 2: References & Tỷ lệ crawl thành công
# --------------------------------------------------------------------------- #
def compute_reference_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán bảng thống kê chi tiết về References và Tỷ lệ thành công."""
    if df.empty or "references_count" not in df.columns:
        return pd.DataFrame()

    # Tính theo từng bài báo duy nhất (paper_id) vì references thu thập theo base_id
    paper_grouped = df.groupby("paper_id").first().reset_index()
    total_papers = len(paper_grouped)

    success_papers = (paper_grouped["references_status"] == "success").sum()
    failed_papers = total_papers - success_papers
    success_rate = (success_papers / total_papers * 100.0) if total_papers > 0 else 0.0

    ref_counts = paper_grouped["references_count"]
    arxiv_ref_counts = paper_grouped["arxiv_references_count"]

    stats_dict = {
        "Chỉ số": [
            "Tổng số bài báo đánh giá (Unique papers)",
            "Số bài crawl References thành công",
            "Số bài crawl References thất bại / chưa index",
            "Tỷ lệ crawl metadata references thành công (%)",
            "Số lượng references trung bình mỗi bài (Mean)",
            "Số lượng references trung vị mỗi bài (Median)",
            "Số lượng references lớn nhất (Max)",
            "Số lượng references nhỏ nhất (Min)",
            "Số references có ArXiv ID trung bình mỗi bài",
        ],
        "Giá trị": [
            f"{total_papers:,}",
            f"{success_papers:,}",
            f"{failed_papers:,}",
            f"{success_rate:.2f}%",
            f"{ref_counts.mean():.2f}",
            f"{ref_counts.median():.1f}",
            f"{ref_counts.max():,}",
            f"{ref_counts.min():,}",
            f"{arxiv_ref_counts.mean():.2f}",
        ],
    }
    return pd.DataFrame(stats_dict)


def plot_reference_stats(df: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """Vẽ 2 biểu đồ: (1) Phân phối số lượng references, (2) Tỷ lệ crawl thành công."""
    if df.empty:
        return None

    paper_grouped = df.groupby("paper_id").first().reset_index()
    total_papers = len(paper_grouped)
    success_papers = (paper_grouped["references_status"] == "success").sum()
    failed_papers = total_papers - success_papers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Biểu đồ 1: Phân phối số lượng references (1 cột: màu tím nhạt; mean: đỏ đậm; median: xanh dương đậm)
    ax1.hist(paper_grouped["references_count"], bins=10, color="#b39ddb", edgecolor="#7e57c2", alpha=0.85)
    mean_refs = paper_grouped["references_count"].mean()
    median_refs = paper_grouped["references_count"].median()
    ax1.axvline(mean_refs, color="#c40000", linestyle="--", linewidth=2, label=f"Mean: {mean_refs:.1f}")
    ax1.axvline(median_refs, color="#0018cd", linestyle=":", linewidth=2.2, label=f"Median: {median_refs:.1f}")
    ax1.set_xlabel("Số lượng References", fontweight="bold")
    ax1.set_ylabel("Số lượng bài báo", fontweight="bold")
    title1 = "Phân phối số lượng References mỗi bài báo"
    if title_suffix:
        title1 += f" ({title_suffix})"
    ax1.set_title(title1, fontweight="bold", pad=10)
    ax1.legend()

    # Biểu đồ 2: Tỷ lệ crawl thành công
    labels = ["Thành công", "Thất bại / Không tìm thấy"]
    counts = [success_papers, failed_papers]
    colors = ["#2bf22b", "#ff0000"]  # Xanh dương đậm và Đỏ đậm
    # Tránh lỗi pie khi một trong các giá trị bằng 0
    non_zero = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    p_labels, p_counts, p_colors = zip(*non_zero) if non_zero else (labels, [1, 0], colors)

    wedges, texts, autotexts = ax2.pie(
        p_counts,
        labels=p_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=p_colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
    title2 = "Tỉ lệ crawl metadata references"
    if title_suffix:
        title2 += f" ({title_suffix})"
    ax2.set_title(title2, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# So sánh đối chiếu 2 phương pháp: arxiv_crawler vs oai_crawler
# --------------------------------------------------------------------------- #
def compare_crawlers_stats(df_arxiv: pd.DataFrame, df_oai: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng so sánh trực tiếp cả 2 tiêu chí giữa arxiv_crawler và oai_crawler."""
    # Nhóm theo paper
    p_arxiv = df_arxiv.groupby("paper_id").first().reset_index() if not df_arxiv.empty else pd.DataFrame()
    p_oai = df_oai.groupby("paper_id").first().reset_index() if not df_oai.empty else pd.DataFrame()

    def get_val(df_full, p_df, col_size_before, col_size_after, col_ref):
        if df_full.empty:
            return ["N/A"] * 6
        n_ver = len(df_full)
        size_before_kb = df_full[col_size_before].mean() / 1024 if col_size_before in df_full else 0
        size_after_kb = df_full[col_size_after].mean() / 1024 if col_size_after in df_full else 0
        red_pct = df_full["reduction_percent"].mean() if "reduction_percent" in df_full else 0
        mean_ref = p_df[col_ref].mean() if not p_df.empty and col_ref in p_df else 0
        succ_rate = (
            (p_df["references_status"] == "success").sum() / len(p_df) * 100.0
            if not p_df.empty and "references_status" in p_df
            else 0
        )
        return [
            f"{n_ver:,} phiên bản",
            f"{size_before_kb:,.2f} KB",
            f"{size_after_kb:,.2f} KB",
            f"{red_pct:.2f}%",
            f"{mean_ref:.2f} refs/bài",
            f"{succ_rate:.2f}%",
        ]

    cols = [
        "Tổng số phiên bản tải được",
        "Kích thước trung bình trước khi lọc ảnh (KB)",
        "Kích thước trung bình sau khi lọc ảnh (KB)",
        "Tỷ lệ dung lượng tiết kiệm trung bình (%)",
        "Số lượng references trung bình / bài",
        "Tỷ lệ crawl references thành công (%)",
    ]

    val_arxiv = get_val(df_arxiv, p_arxiv, "total_uncompressed_bytes", "filtered_uncompressed_bytes", "references_count")
    val_oai = get_val(df_oai, p_oai, "total_uncompressed_bytes", "filtered_uncompressed_bytes", "references_count")

    comp_df = pd.DataFrame({
        "Tiêu chí thống kê": cols,
        "arXiv API (arxiv_crawler)": val_arxiv,
        "OAI-PMH (oai_crawler)": val_oai,
    })
    return comp_df


def plot_side_by_side_size(df_arxiv: pd.DataFrame, df_oai: pd.DataFrame) -> plt.Figure:
    """Vẽ biểu đồ so sánh trực diện kích thước trước vs sau lọc ảnh giữa 2 phương pháp."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if not df_arxiv.empty:
        plot_size_comparison(df_arxiv, title_suffix="arXiv API", ax=ax1)
    else:
        ax1.set_title("arxiv_crawler (Chưa có dữ liệu)")

    if not df_oai.empty:
        plot_size_comparison(df_oai, title_suffix="OAI-PMH", ax=ax2)
    else:
        ax2.set_title("oai_crawler (Chưa có dữ liệu)")

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Yêu cầu 3: Đánh giá Hiệu năng (Scraper's Performance Benchmarking)
# --------------------------------------------------------------------------- #
def load_benchmark_data(
    results_csv: str | Path = "benchmark_results.csv",
    timeseries_csv: str | Path = "benchmark_timeseries.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc dữ liệu hiệu năng từ file benchmark_results.csv và benchmark_timeseries.csv."""
    res_path = Path(results_csv)
    if not res_path.exists():
        for candidate in [Path("src") / res_path.name, Path("..") / res_path.name, Path(__file__).parent / res_path.name]:
            if candidate.exists():
                res_path = candidate
                break

    ts_path = Path(timeseries_csv)
    if not ts_path.exists():
        for candidate in [Path("src") / ts_path.name, Path("..") / ts_path.name, Path(__file__).parent / ts_path.name]:
            if candidate.exists():
                ts_path = candidate
                break

    bench_df = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    ts_df = pd.read_csv(ts_path) if ts_path.exists() else pd.DataFrame()
    return bench_df, ts_df


def _find_crawler_row(benchmark_df: pd.DataFrame, crawler_name: str) -> pd.Series | None:
    if benchmark_df.empty or "crawler" not in benchmark_df.columns:
        return None
    norm = crawler_name.strip().lower()
    matched = benchmark_df[benchmark_df["crawler"].str.lower() == norm]
    if matched.empty:
        matched = benchmark_df[benchmark_df["crawler"].str.lower().str.contains(norm)]
    if matched.empty:
        return None
    return matched.iloc[-1]


def compute_performance_stats(benchmark_df: pd.DataFrame, crawler_name: str) -> pd.DataFrame:
    """Tạo bảng thống kê chi tiết Thời gian chạy và Bộ nhớ & Lưu trữ cho 1 crawler."""
    row = _find_crawler_row(benchmark_df, crawler_name)
    if row is None:
        return pd.DataFrame()

    papers = int(row.get("papers_crawled", 50))
    versions = int(row.get("versions_crawled", 0))
    wall_sec = float(row.get("total_wall_time_sec", 0.0))
    disc_sec = float(row.get("discovery_time_sec", 0.0))
    disc_pct = (disc_sec / wall_sec * 100.0) if wall_sec > 0 else 0.0
    avg_sec = float(row.get("avg_paper_time_sec", 0.0))
    med_sec = float(row.get("median_paper_time_sec", 0.0))
    min_sec = float(row.get("min_paper_time_sec", 0.0))
    max_sec = float(row.get("max_paper_time_sec", 0.0))

    base_ram = float(row.get("baseline_ram_mb", 0.0))
    avg_ram = float(row.get("avg_ram_mb", 0.0))
    peak_ram = float(row.get("peak_ram_mb", 0.0))
    overhead_ram = float(row.get("ram_overhead_mb", max(0.0, peak_ram - base_ram)))

    raw_bytes = int(row.get("raw_archive_bytes", 0))
    uncomp_bytes = int(row.get("total_uncompressed_bytes", 0))
    max_disk_bytes = int(row.get("max_disk_storage_bytes", 0))
    final_disk_bytes = int(row.get("final_output_storage_bytes", 0))
    saving_pct = float(
        row.get(
            "storage_saving_pct",
            (1.0 - final_disk_bytes / uncomp_bytes) * 100.0 if uncomp_bytes > 0 else 0.0,
        )
    )

    stats_dict = {
        "Nhóm tiêu chí": [
            "Quy mô mẫu",
            "Thời gian chạy (Running Time)",
            "Thời gian chạy (Running Time)",
            "Thời gian chạy (Running Time)",
            "Thời gian chạy (Running Time)",
            "Thời gian chạy (Running Time)",
            "Thời gian chạy (Running Time)",
            "Bộ nhớ (Memory Footprint)",
            "Bộ nhớ (Memory Footprint)",
            "Bộ nhớ (Memory Footprint)",
            "Lưu trữ (Disk & Storage)",
            "Lưu trữ (Disk & Storage)",
            "Lưu trữ (Disk & Storage)",
            "Lưu trữ (Disk & Storage)",
            "Lưu trữ (Disk & Storage)",
        ],
        "Chỉ số hiệu năng": [
            "Số bài báo / Tổng số phiên bản cào được",
            "Tổng thời gian chạy thực tế (Total Wall Time)",
            "Thời gian khám phá mục bài báo (Discovery Time)",
            "Thời gian xử lý trung bình mỗi bài (Mean Time/Paper)",
            "Thời gian xử lý trung vị mỗi bài (Median Time/Paper)",
            "Thời gian xử lý nhanh nhất 1 bài (Min Time/Paper)",
            "Thời gian xử lý chậm nhất 1 bài (Max Time/Paper)",
            "Mức tiêu thụ RAM cơ sở (Baseline RAM)",
            "Mức tiêu thụ RAM trung bình (Average RAM)",
            "Mức tiêu thụ RAM đỉnh tối đa (Peak RAM)",
            "Tổng dung lượng gói nén .tar.gz tải về (raw_archive_bytes)",
            "Dung lượng giải nén lý thuyết trước lọc (total_uncompressed_bytes)",
            "Dung lượng đĩa tối đa cần thiết lúc cào (max_disk_storage_bytes)",
            "Dung lượng đầu ra cuối cùng trên ổ đĩa (final_output_storage_bytes)",
            "Tỷ lệ tiết kiệm không gian đĩa thực tế so với lý thuyết (%)",
        ],
        "Giá trị": [
            f"{papers:,} bài ({versions:,} phiên bản)",
            f"{wall_sec:,.2f} giây ({wall_sec / 60.0:,.2f} phút)",
            f"{disc_sec:,.2f} giây (chiếm {disc_pct:.1f}% tổng thời gian)",
            f"{avg_sec:,.2f} giây / bài",
            f"{med_sec:,.2f} giây / bài",
            f"{min_sec:,.2f} giây",
            f"{max_sec:,.2f} giây",
            f"{base_ram:,.2f} MB",
            f"{avg_ram:,.2f} MB",
            f"{peak_ram:,.2f} MB (Tăng đỉnh +{overhead_ram:,.2f} MB)",
            f"{raw_bytes / (1024 * 1024):,.2f} MB ({raw_bytes:,} Bytes)",
            f"{uncomp_bytes / (1024 * 1024):,.2f} MB ({uncomp_bytes:,} Bytes)",
            f"{max_disk_bytes / (1024 * 1024):,.2f} MB ({max_disk_bytes:,} Bytes)",
            f"{final_disk_bytes / (1024 * 1024):,.2f} MB ({final_disk_bytes:,} Bytes)",
            f"{saving_pct:.2f}%",
        ],
    }
    return pd.DataFrame(stats_dict)


def compare_performance_stats(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng so sánh đối chiếu trực diện các chỉ số hiệu năng giữa arxiv_crawler và oai_crawler."""
    if benchmark_df.empty:
        return pd.DataFrame()

    row_arxiv = _find_crawler_row(benchmark_df, "arxiv_crawler")
    row_oai = _find_crawler_row(benchmark_df, "oai_crawler")

    def extract_metrics(row: pd.Series | None) -> list[str]:
        if row is None:
            return ["N/A"] * 12
        papers = int(row.get("papers_crawled", 50))
        versions = int(row.get("versions_crawled", 0))
        wall_sec = float(row.get("total_wall_time_sec", 0.0))
        disc_sec = float(row.get("discovery_time_sec", 0.0))
        avg_sec = float(row.get("avg_paper_time_sec", 0.0))
        med_sec = float(row.get("median_paper_time_sec", 0.0))
        min_sec = float(row.get("min_paper_time_sec", 0.0))
        max_sec = float(row.get("max_paper_time_sec", 0.0))
        avg_ram = float(row.get("avg_ram_mb", 0.0))
        peak_ram = float(row.get("peak_ram_mb", 0.0))
        raw_mb = float(row.get("raw_archive_bytes", 0)) / (1024 * 1024)
        uncomp_mb = float(row.get("total_uncompressed_bytes", 0)) / (1024 * 1024)
        max_disk_mb = float(row.get("max_disk_storage_bytes", 0)) / (1024 * 1024)
        final_disk_mb = float(row.get("final_output_storage_bytes", 0)) / (1024 * 1024)
        saving_pct = float(row.get("storage_saving_pct", 0.0))

        return [
            f"{papers:,} bài ({versions:,} phiên bản)",
            f"{wall_sec:,.2f} s ({wall_sec / 60.0:,.2f} phút)",
            f"{disc_sec:,.2f} s",
            f"{avg_sec:,.2f} s / bài",
            f"{med_sec:,.2f} s / bài",
            f"{min_sec:,.2f} s / {max_sec:,.2f} s",
            f"{avg_ram:,.2f} MB",
            f"{peak_ram:,.2f} MB",
            f"{raw_mb:,.2f} MB",
            f"{uncomp_mb:,.2f} MB",
            f"{max_disk_mb:,.2f} MB",
            f"{final_disk_mb:,.2f} MB (Tiết kiệm {saving_pct:.2f}%)",
        ]

    criteria = [
        "Số bài báo / Tổng số phiên bản",
        "Tổng thời gian chạy (Total Wall Time)",
        "Thời gian khám phá mục bài (Discovery Time)",
        "Thời gian xử lý trung bình / bài (Mean)",
        "Thời gian xử lý trung vị / bài (Median)",
        "Thời gian xử lý Min / Max mỗi bài",
        "Mức tiêu thụ RAM trung bình (Average RAM)",
        "Mức tiêu thụ RAM đỉnh tối đa (Peak RAM)",
        "Tổng dung lượng gói nén gốc (raw_archive_bytes)",
        "Dung lượng giải nén lý thuyết chưa lọc (total_uncompressed_bytes)",
        "Dung lượng đĩa tối đa lúc cào (max_disk_storage_bytes)",
        "Dung lượng đầu ra cuối cùng trên đĩa (final_output_storage_bytes)",
    ]

    comp_df = pd.DataFrame({
        "Tiêu chí hiệu năng": criteria,
        "arXiv API (arxiv_crawler)": extract_metrics(row_arxiv),
        "OAI-PMH (oai_crawler)": extract_metrics(row_oai),
    })
    return comp_df


def plot_performance_charts(
    benchmark_df: pd.DataFrame,
    timeseries_df: pd.DataFrame | None = None,
) -> tuple[plt.Figure, plt.Figure, plt.Figure] | None:
    """Vẽ 3 bộ biểu đồ đánh giá hiệu năng: Thời gian chạy, RAM Footprint và Dung lượng Lưu trữ."""
    if benchmark_df.empty:
        print("[Cảnh báo] benchmark_df rỗng, không thể vẽ biểu đồ hiệu năng.")
        return None

    row_arxiv = _find_crawler_row(benchmark_df, "arxiv_crawler")
    row_oai = _find_crawler_row(benchmark_df, "oai_crawler")
    if row_arxiv is None or row_oai is None:
        print("[Cảnh báo] Cần đủ dữ liệu của cả arxiv_crawler và oai_crawler để vẽ biểu đồ đối sánh.")
        return None

    # ----------------------------------------------------------------------- #
    # Biểu đồ 1: So sánh Thời gian chạy (Total Wall Time, Discovery Time, Time/Paper)
    # ----------------------------------------------------------------------- #
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.2))

    macro_labels = ["Tổng thời gian chạy\n(Total Wall Time)", "Thời gian Khám phá\n(Discovery Time)"]
    arxiv_macro = [float(row_arxiv["total_wall_time_sec"]), float(row_arxiv["discovery_time_sec"])]
    oai_macro = [float(row_oai["total_wall_time_sec"]), float(row_oai["discovery_time_sec"])]

    x1 = np.arange(len(macro_labels))
    w1 = 0.34
    r1_arxiv = ax1a.bar(x1 - w1 / 2, arxiv_macro, w1, label="arXiv API (arxiv_crawler)", color="#0d00ff", alpha=0.90)
    r1_oai = ax1a.bar(x1 + w1 / 2, oai_macro, w1, label="OAI-PMH (oai_crawler)", color="#ff0000", alpha=0.90)

    ax1a.set_ylabel("Thời gian (Giây)", fontweight="bold")
    ax1a.set_title("Tổng thời gian chạy & Thời gian Discovery", fontweight="bold", pad=12)
    ax1a.set_xticks(x1)
    ax1a.set_xticklabels(macro_labels)
    ax1a.set_ylim(0, max(arxiv_macro + oai_macro) * 1.18)
    ax1a.legend(frameon=True)

    for rect in list(r1_arxiv) + list(r1_oai):
        h = rect.get_height()
        ax1a.annotate(
            f"{h:,.1f}s",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    speedup_pct = (1.0 - arxiv_macro[0] / oai_macro[0]) * 100.0 if oai_macro[0] > 0 else 0.0
    ax1a.text(
        0.5,
        0.90,
        f"arXiv API nhanh hơn {speedup_pct:.1f}% tổng thời gian nhờ Batch Discovery",
        transform=ax1a.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffd42a", ec="none", alpha=0.35),
        fontweight="bold",
        fontsize=9.5,
    )

    paper_labels = ["Trung bình\n(Mean)", "Trung vị\n(Median)", "Nhanh nhất\n(Min)", "Chậm nhất\n(Max)"]
    arxiv_paper_t = [
        float(row_arxiv["avg_paper_time_sec"]),
        float(row_arxiv["median_paper_time_sec"]),
        float(row_arxiv["min_paper_time_sec"]),
        float(row_arxiv["max_paper_time_sec"]),
    ]
    oai_paper_t = [
        float(row_oai["avg_paper_time_sec"]),
        float(row_oai["median_paper_time_sec"]),
        float(row_oai["min_paper_time_sec"]),
        float(row_oai["max_paper_time_sec"]),
    ]

    x2 = np.arange(len(paper_labels))
    r2_arxiv = ax1b.bar(x2 - w1 / 2, arxiv_paper_t, w1, label="arXiv API (arxiv_crawler)", color="#0d00ff", alpha=0.90)
    r2_oai = ax1b.bar(x2 + w1 / 2, oai_paper_t, w1, label="OAI-PMH (oai_crawler)", color="#ff0000", alpha=0.90)

    ax1b.set_ylabel("Thời gian xử lý / bài (Giây)", fontweight="bold")
    ax1b.set_title("Thời gian xử lý trên từng bài báo (S2 + Tải & Giải nén)", fontweight="bold", pad=12)
    ax1b.set_xticks(x2)
    ax1b.set_xticklabels(paper_labels)
    ax1b.set_ylim(0, max(arxiv_paper_t + oai_paper_t) * 1.18)
    ax1b.legend(frameon=True)

    for rect in list(r2_arxiv) + list(r2_oai):
        h = rect.get_height()
        ax1b.annotate(
            f"{h:,.1f}s",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    fig1.tight_layout()

    # ----------------------------------------------------------------------- #
    # Biểu đồ 2: Mức tiêu thụ RAM (Peak vs Average RAM & Biến thiên theo thời gian)
    # ----------------------------------------------------------------------- #
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.2))

    ram_labels = ["RAM cơ sở\n(Baseline)", "RAM trung bình\n(Average)", "RAM đỉnh\n(Peak)"]
    arxiv_ram = [
        float(row_arxiv.get("baseline_ram_mb", 0.0)),
        float(row_arxiv["avg_ram_mb"]),
        float(row_arxiv["peak_ram_mb"]),
    ]
    oai_ram = [
        float(row_oai.get("baseline_ram_mb", 0.0)),
        float(row_oai["avg_ram_mb"]),
        float(row_oai["peak_ram_mb"]),
    ]

    x_ram = np.arange(len(ram_labels))
    r_ram_arxiv = ax2a.bar(x_ram - w1 / 2, arxiv_ram, w1, label="arXiv API (arxiv_crawler)", color="#0d00ff", alpha=0.90)
    r_ram_oai = ax2a.bar(x_ram + w1 / 2, oai_ram, w1, label="OAI-PMH (oai_crawler)", color="#ff0000", alpha=0.90)

    ax2a.set_ylabel("Mức tiêu thụ RAM (MB)", fontweight="bold")
    ax2a.set_title("So sánh mức tiêu thụ Bộ nhớ (Peak RAM vs Average RAM)", fontweight="bold", pad=12)
    ax2a.set_xticks(x_ram)
    ax2a.set_xticklabels(ram_labels)
    ax2a.set_ylim(0, max(arxiv_ram + oai_ram) * 1.20)
    ax2a.legend(frameon=True)

    for rect in list(r_ram_arxiv) + list(r_ram_oai):
        h = rect.get_height()
        ax2a.annotate(
            f"{h:,.1f} MB",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    if timeseries_df is not None and not timeseries_df.empty and "ram_mb" in timeseries_df.columns:
        ram_ts = timeseries_df
        if "record_type" in ram_ts.columns:
            ram_only = ram_ts[ram_ts["record_type"] == "ram_sample"]
            if not ram_only.empty:
                ram_ts = ram_only

        ts_arxiv = ram_ts[ram_ts["crawler"] == "arxiv_crawler"].sort_values("elapsed_sec")
        ts_oai = ram_ts[ram_ts["crawler"] == "oai_crawler"].sort_values("elapsed_sec")

        if not ts_arxiv.empty:
            ax2b.plot(
                ts_arxiv["elapsed_sec"],
                ts_arxiv["ram_mb"],
                color="#0d00ff",
                linewidth=1.8,
                alpha=0.90,
                label=f"arXiv API (Peak: {float(row_arxiv['peak_ram_mb']):.1f} MB)",
            )
        if not ts_oai.empty:
            ax2b.plot(
                ts_oai["elapsed_sec"],
                ts_oai["ram_mb"],
                color="#ff0000",
                linewidth=1.8,
                alpha=0.85,
                label=f"OAI-PMH (Peak: {float(row_oai['peak_ram_mb']):.1f} MB)",
            )
        ax2b.set_xlabel("Thời gian chạy (Giây)", fontweight="bold")
        ax2b.set_ylabel("RAM RSS (MB)", fontweight="bold")
        ax2b.set_title("Biến thiên mức tiêu thụ RAM theo thời gian (Chu kỳ 0.5s)", fontweight="bold", pad=12)
        ax2b.legend(frameon=True)
    else:
        ax2b.set_title("Chưa có dữ liệu chuỗi thời gian RAM (benchmark_timeseries.csv)", fontweight="bold")

    fig2.tight_layout()

    # ----------------------------------------------------------------------- #
    # Biểu đồ 3: So sánh Dung lượng Lưu trữ (Lý thuyết, Gói nén gốc, Đĩa tối đa, Đầu ra cuối)
    # ----------------------------------------------------------------------- #
    fig3, ax3 = plt.subplots(figsize=(12, 5.5))

    storage_labels = [
        "Dung lượng giải nén lý thuyết\n(Trước khi lọc ảnh)",
        "Tổng dung lượng gói nén gốc\n(raw_archive_bytes)",
        "Dung lượng đĩa tối đa lúc cào\n(max_disk_storage_bytes)",
        "Dung lượng đầu ra cuối cùng\n(final_output_storage_bytes)",
    ]
    arxiv_storage_mb = [
        float(row_arxiv["total_uncompressed_bytes"]) / (1024 * 1024),
        float(row_arxiv["raw_archive_bytes"]) / (1024 * 1024),
        float(row_arxiv["max_disk_storage_bytes"]) / (1024 * 1024),
        float(row_arxiv["final_output_storage_bytes"]) / (1024 * 1024),
    ]
    oai_storage_mb = [
        float(row_oai["total_uncompressed_bytes"]) / (1024 * 1024),
        float(row_oai["raw_archive_bytes"]) / (1024 * 1024),
        float(row_oai["max_disk_storage_bytes"]) / (1024 * 1024),
        float(row_oai["final_output_storage_bytes"]) / (1024 * 1024),
    ]

    x_st = np.arange(len(storage_labels))
    w_st = 0.35
    r_st_arxiv = ax3.bar(x_st - w_st / 2, arxiv_storage_mb, w_st, label="arXiv API (arxiv_crawler)", color="#0d00ff", alpha=0.90)
    r_st_oai = ax3.bar(x_st + w_st / 2, oai_storage_mb, w_st, label="OAI-PMH (oai_crawler)", color="#ff0000", alpha=0.90)

    ax3.set_ylabel("Dung lượng (MB)", fontweight="bold")
    ax3.set_title(
        "So sánh Dung lượng Lưu trữ Lý thuyết, Gói nén gốc (.tar.gz) và Đầu ra Thực tế trên Đĩa",
        fontweight="bold",
        pad=12,
    )
    ax3.set_xticks(x_st)
    ax3.set_xticklabels(storage_labels)
    ax3.set_ylim(0, max(arxiv_storage_mb + oai_storage_mb) * 1.20)
    ax3.legend(frameon=True)

    for rect in list(r_st_arxiv) + list(r_st_oai):
        h = rect.get_height()
        ax3.annotate(
            f"{h:,.2f} MB",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    save_arxiv_pct = float(row_arxiv.get("storage_saving_pct", 0.0))
    save_oai_pct = float(row_oai.get("storage_saving_pct", 0.0))
    ax3.text(
        0.5,
        0.90,
        f"Tối ưu lưu trữ thực tế trên đĩa: arXiv API giảm {save_arxiv_pct:.1f}% | OAI-PMH giảm {save_oai_pct:.1f}% so với giải nén lý thuyết",
        transform=ax3.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffd42a", ec="none", alpha=0.35),
        fontweight="bold",
    )

    fig3.tight_layout()
    return fig1, fig2, fig3
