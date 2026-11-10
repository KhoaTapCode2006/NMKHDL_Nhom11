"""
Parser Benchmark and Statistics Tracker.
Measures execution time, memory usage (tracemalloc + psutil RSS),
parse success rate, version coverage, and element deduplication metrics.
Exports summary reports formatted for Jupyter Book.
"""

from __future__ import annotations

import os
import time
import json
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional
from pathlib import Path


class ParserBenchmark:
    """Collects and reports performance, coverage, and deduplication statistics."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_wall_time: float = 0.0
        self.end_wall_time: float = 0.0
        self.peak_tracemalloc_bytes: int = 0
        self.peak_rss_bytes: int = 0

        # Detailed tracking records
        self.pub_stats: List[Dict[str, Any]] = []
        self.version_logs: List[Dict[str, Any]] = []

    def start(self):
        """Starts benchmarking timers and memory tracing."""
        tracemalloc.start()
        self.start_wall_time = time.perf_counter()
        self.peak_rss_bytes = self.process.memory_info().rss

    def update_peak_memory(self):
        """Updates peak memory measurements."""
        rss = self.process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss

    def stop(self):
        """Stops benchmarking and records final peak memory."""
        self.end_wall_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.peak_tracemalloc_bytes = peak
        tracemalloc.stop()
        self.update_peak_memory()

    def record_version(
        self,
        arxiv_id: str,
        version: str,
        status: str,  # "success", "empty_version", "failed"
        duration_sec: float,
        error_msg: Optional[str] = None,
        compilation_files_count: int = 0,
        elements_count: int = 0,
    ):
        """Records parsing result for a single version."""
        self.update_peak_memory()
        self.version_logs.append({
            "arxiv_id": arxiv_id,
            "version": version,
            "status": status,
            "duration_sec": round(duration_sec, 4),
            "error_msg": error_msg,
            "compilation_files_count": compilation_files_count,
            "elements_count": elements_count,
        })

    def record_publication(
        self,
        arxiv_id: str,
        expected_versions: int,
        crawled_versions: int,
        tex_versions: int,
        empty_versions: int,
        dedup_stats: Dict[str, Any],
        duration_sec: float,
    ):
        """Records aggregated statistics for a publication."""
        self.update_peak_memory()
        self.pub_stats.append({
            "arxiv_id": arxiv_id,
            "expected_versions": expected_versions,
            "crawled_versions": crawled_versions,
            "tex_versions": tex_versions,
            "empty_versions": empty_versions,
            "total_before": dedup_stats.get("total_before", 0),
            "total_after": dedup_stats.get("total_after", 0),
            "reduction_count": dedup_stats.get("reduction_count", 0),
            "reduction_percentage": dedup_stats.get("reduction_percentage", 0.0),
            "by_kind_before": dedup_stats.get("by_kind_before", {}),
            "by_kind_after": dedup_stats.get("by_kind_after", {}),
            "duration_sec": round(duration_sec, 4),
        })

    def generate_summary(self) -> Dict[str, Any]:
        """Calculates global summary metrics."""
        total_time = self.end_wall_time - self.start_wall_time
        num_pubs = len(self.pub_stats)
        num_versions = len(self.version_logs)

        # Success rate
        success_versions = sum(1 for v in self.version_logs if v["status"] == "success")
        empty_versions = sum(1 for v in self.version_logs if v["status"] == "empty_version")
        failed_versions = sum(1 for v in self.version_logs if v["status"] == "failed")

        success_rate = (success_versions / num_versions * 100.0) if num_versions > 0 else 0.0

        # Version coverage
        total_expected_versions = sum(p["expected_versions"] for p in self.pub_stats)
        total_tex_versions = sum(p["tex_versions"] for p in self.pub_stats)
        total_empty_versions = sum(p["empty_versions"] for p in self.pub_stats)
        overall_coverage = (
            (total_tex_versions / total_expected_versions * 100.0)
            if total_expected_versions > 0
            else 0.0
        )

        # Deduplication aggregations
        total_elements_before = sum(p["total_before"] for p in self.pub_stats)
        total_elements_after = sum(p["total_after"] for p in self.pub_stats)
        total_reduced = total_elements_before - total_elements_after
        overall_reduction_pct = (
            (total_reduced / total_elements_before * 100.0)
            if total_elements_before > 0
            else 0.0
        )

        kind_before_totals: Dict[str, int] = {}
        kind_after_totals: Dict[str, int] = {}
        for p in self.pub_stats:
            for k, count in p.get("by_kind_before", {}).items():
                kind_before_totals[k] = kind_before_totals.get(k, 0) + count
            for k, count in p.get("by_kind_after", {}).items():
                kind_after_totals[k] = kind_after_totals.get(k, 0) + count

        return {
            "performance": {
                "total_wall_time_sec": round(total_time, 2),
                "avg_time_per_pub_sec": round(total_time / num_pubs, 3) if num_pubs > 0 else 0.0,
                "avg_time_per_version_sec": round(total_time / num_versions, 3) if num_versions > 0 else 0.0,
                "peak_rss_mb": round(self.peak_rss_bytes / (1024 * 1024), 2),
                "peak_tracemalloc_mb": round(self.peak_tracemalloc_bytes / (1024 * 1024), 2),
            },
            "success_rate": {
                "total_publications": num_pubs,
                "total_versions_processed": num_versions,
                "successful_versions": success_versions,
                "empty_versions": empty_versions,
                "failed_versions": failed_versions,
                "success_rate_percentage": round(success_rate, 2),
            },
            "version_coverage": {
                "total_expected_versions": total_expected_versions,
                "versions_with_tex": total_tex_versions,
                "empty_versions": total_empty_versions,
                "coverage_percentage": round(overall_coverage, 2),
            },
            "deduplication": {
                "total_elements_before": total_elements_before,
                "total_elements_after": total_elements_after,
                "reduction_count": total_reduced,
                "reduction_percentage": round(overall_reduction_pct, 2),
                "by_kind_before": kind_before_totals,
                "by_kind_after": kind_after_totals,
            },
        }

    def export_markdown_report(self, output_path: Path) -> str:
        """Generates a comprehensive Markdown report ready for Jupyter Book."""
        summary = self.generate_summary()
        perf = summary["performance"]
        succ = summary["success_rate"]
        cov = summary["version_coverage"]
        dedup = summary["deduplication"]

        md = []
        md.append("# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)\n")

        md.append("## 1. Đo đạc hiệu năng (Performance Benchmark)")
        md.append(f"- **Tổng thời gian thực thi (Total Wall Time)**: `{perf['total_wall_time_sec']}s`")
        md.append(f"- **Thời gian trung bình mỗi publication**: `{perf['avg_time_per_pub_sec']}s`")
        md.append(f"- **Thời gian trung bình mỗi version**: `{perf['avg_time_per_version_sec']}s`")
        md.append(f"- **RAM tối đa (Peak RSS Memory)**: `{perf['peak_rss_mb']} MB`")
        md.append(f"- **Bộ nhớ cấp phát động tối đa (Peak Tracemalloc)**: `{perf['peak_tracemalloc_mb']} MB`\n")

        md.append("## 2. Tỉ lệ parse thành công & Version Coverage")
        md.append("| Chỉ số | Số lượng | Tỉ lệ (%) |")
        md.append("|---|---|---|")
        md.append(f"| Tổng số publications | {succ['total_publications']} | 100% |")
        md.append(f"| Tổng số versions được xử lý | {succ['total_versions_processed']} | 100% |")
        md.append(f"| Versions parse thành công | {succ['successful_versions']} | {succ['success_rate_percentage']}% |")
        md.append(f"| Versions rỗng (không có TeX source) | {succ['empty_versions']} | {round(succ['empty_versions']/succ['total_versions_processed']*100, 2) if succ['total_versions_processed']>0 else 0}% |")
        md.append(f"| Versions lỗi (failed) | {succ['failed_versions']} | {round(succ['failed_versions']/succ['total_versions_processed']*100, 2) if succ['total_versions_processed']>0 else 0}% |")
        md.append(f"| **Version Coverage (TeX Source)** | **{cov['versions_with_tex']} / {cov['total_expected_versions']}** | **{cov['coverage_percentage']}%** |\n")

        md.append("## 3. Thống kê Khử trùng lặp Elements (Deduplication)")
        md.append(f"- **Tổng elements trước dedup**: `{dedup['total_elements_before']}`")
        md.append(f"- **Tổng elements sau dedup**: `{dedup['total_elements_after']}`")
        md.append(f"- **Số lượng phần tử trùng lặp đã loại bỏ**: `{dedup['reduction_count']}`")
        md.append(f"- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`{dedup['reduction_percentage']}%`**\n")

        md.append("### Phân bố Elements theo Kind (Trước vs Sau Dedup)")
        md.append("| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |")
        md.append("|---|---|---|---|---|")

        all_kinds = sorted(set(list(dedup["by_kind_before"].keys()) + list(dedup["by_kind_after"].keys())))
        for k in all_kinds:
            bef = dedup["by_kind_before"].get(k, 0)
            aft = dedup["by_kind_after"].get(k, 0)
            diff = bef - aft
            pct = round(diff / bef * 100.0, 2) if bef > 0 else 0.0
            md.append(f"| `{k}` | {bef} | {aft} | {diff} | {pct}% |")

        md_content = "\n".join(md)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        return md_content
