"""Module thống kê dữ liệu cào bài báo arXiv phục vụ báo cáo Jupyter Book.

Tập trung vào 2 chỉ số cốt lõi theo yêu cầu của đồ án:
1. Kích thước trung bình (bytes) của bài báo trước và sau khi loại bỏ hình ảnh.
2. Số lượng references trung bình mỗi bài, tỉ lệ crawl metadata references thành công.
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
