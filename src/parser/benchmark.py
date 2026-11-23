"""
Parser benchmark and statistics tracker.

Measures wall time, memory (psutil RSS sampled by a background thread, and
optionally tracemalloc), parse success per version, version coverage, element
and BibTeX deduplication, and output size. Summaries are exported as JSON and
Markdown for the report.

``tracemalloc`` hooks every allocation and slows Python down noticeably, so it
is off by default; enable it only to inspect heap usage, never for timing runs.
"""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psutil

PARSED_STATUSES = {"success", "partial"}


def directory_size_bytes(paths: Iterable[Path]) -> int:
    """Total size of every file below the given directories."""
    total = 0
    for path in paths:
        path = Path(path)
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total


class ParserBenchmark:
    """Collects and reports performance, coverage, and deduplication statistics."""

    def __init__(self, trace_memory: bool = False, sample_interval_sec: float = 0.2):
        self.process = psutil.Process(os.getpid())
        self.trace_memory = trace_memory
        self.sample_interval_sec = sample_interval_sec
        self.start_wall_time: float = 0.0
        self.end_wall_time: float = 0.0
        self.peak_tracemalloc_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self.baseline_rss_bytes: int = 0
        self._rss_sum: float = 0.0
        self._rss_count: int = 0
        self._stop_event = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self.output_size_bytes: Optional[int] = None

        self.pub_stats: List[Dict[str, Any]] = []
        self.version_logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Memory sampling
    # ------------------------------------------------------------------ #
    def update_peak_memory(self) -> None:
        rss = self.process.memory_info().rss
        self._rss_sum += rss
        self._rss_count += 1
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_sec):
            self.update_peak_memory()

    def start(self) -> None:
        """Start the wall clock and the background RSS sampler."""
        if self.trace_memory:
            tracemalloc.start()
        self.baseline_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.baseline_rss_bytes
        self._rss_sum, self._rss_count = 0.0, 0
        self._stop_event.clear()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()
        self.start_wall_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the clock and the sampler and record the final peaks."""
        self.end_wall_time = time.perf_counter()
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)
        self.update_peak_memory()
        if self.trace_memory and tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            self.peak_tracemalloc_bytes = peak
            tracemalloc.stop()

    def measure_output(self, paths: Iterable[Path]) -> int:
        """Record the size of the final output (e.g. every elements/ and bibtex/ dir)."""
        self.output_size_bytes = directory_size_bytes(paths)
        return self.output_size_bytes

    # ------------------------------------------------------------------ #
    # Records
    # ------------------------------------------------------------------ #
    def record_version(
        self,
        arxiv_id: str,
        version: str,
        status: str,  # "success", "partial", "empty_version", "failed"
        duration_sec: float,
        error_msg: Optional[str] = None,
        compilation_files_count: int = 0,
        elements_count: int = 0,
    ) -> None:
        """Records parsing result for a single version."""
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
        bibtex_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records aggregated statistics for a publication."""
        bibtex_stats = bibtex_stats or {}
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
            "bibtex_before": bibtex_stats.get("total_raw_entries", 0),
            "bibtex_after": bibtex_stats.get("dedup_entries_count", 0),
            "bibtex_renamed_keys": bibtex_stats.get("renamed_keys_count", 0),
            "bibtex_enriched": bibtex_stats.get("fields_enriched_count", 0),
            "bibtex_uncited_dropped": bibtex_stats.get("uncited_bib_entries_dropped", 0),
            "duration_sec": round(duration_sec, 4),
        })

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def generate_summary(self) -> Dict[str, Any]:
        """Calculates global summary metrics."""
        total_time = self.end_wall_time - self.start_wall_time
        num_pubs = len(self.pub_stats)
        num_versions = len(self.version_logs)

        def count(status: str) -> int:
            return sum(1 for v in self.version_logs if v["status"] == status)

        success_versions = count("success")
        partial_versions = count("partial")
        empty_versions = count("empty_version")
        failed_versions = count("failed")
        parsed_versions = success_versions + partial_versions
        with_tex = num_versions - empty_versions

        def pct(part: float, whole: float) -> float:
            return round(part / whole * 100.0, 2) if whole else 0.0

        total_expected_versions = sum(p["expected_versions"] for p in self.pub_stats)
        total_tex_versions = sum(p["tex_versions"] for p in self.pub_stats)
        total_empty_versions = sum(p["empty_versions"] for p in self.pub_stats)

        total_elements_before = sum(p["total_before"] for p in self.pub_stats)
        total_elements_after = sum(p["total_after"] for p in self.pub_stats)
        total_reduced = total_elements_before - total_elements_after

        kind_before_totals: Dict[str, int] = {}
        kind_after_totals: Dict[str, int] = {}
        for p in self.pub_stats:
            for k, value in p.get("by_kind_before", {}).items():
                kind_before_totals[k] = kind_before_totals.get(k, 0) + value
            for k, value in p.get("by_kind_after", {}).items():
                kind_after_totals[k] = kind_after_totals.get(k, 0) + value

        bib_before = sum(p.get("bibtex_before", 0) for p in self.pub_stats)
        bib_after = sum(p.get("bibtex_after", 0) for p in self.pub_stats)
        mib = 1024 * 1024
        avg_rss = self._rss_sum / self._rss_count if self._rss_count else self.peak_rss_bytes

        return {
            "performance": {
                "total_wall_time_sec": round(total_time, 2),
                "avg_time_per_pub_sec": round(total_time / num_pubs, 3) if num_pubs else 0.0,
                "avg_time_per_version_sec": (
                    round(total_time / num_versions, 3) if num_versions else 0.0
                ),
                "peak_rss_mb": round(self.peak_rss_bytes / mib, 2),
                "avg_rss_mb": round(avg_rss / mib, 2),
                "baseline_rss_mb": round(self.baseline_rss_bytes / mib, 2),
                "rss_samples": self._rss_count,
                "tracemalloc_enabled": self.trace_memory,
                "peak_tracemalloc_mb": round(self.peak_tracemalloc_bytes / mib, 2),
                "output_size_bytes": self.output_size_bytes,
            },
            "success_rate": {
                "total_publications": num_pubs,
                "total_versions_processed": num_versions,
                "successful_versions": success_versions,
                "partial_versions": partial_versions,
                "parsed_versions": parsed_versions,
                "empty_versions": empty_versions,
                "failed_versions": failed_versions,
                # parsed (success + partial) over every version
                "success_rate_percentage": pct(parsed_versions, num_versions),
                # parsed over the versions that actually have TeX sources
                "parse_success_rate_percentage": pct(parsed_versions, with_tex),
            },
            "version_coverage": {
                "total_expected_versions": total_expected_versions,
                "versions_with_tex": total_tex_versions,
                "empty_versions": total_empty_versions,
                "coverage_percentage": pct(total_tex_versions, total_expected_versions),
            },
            "deduplication": {
                "total_elements_before": total_elements_before,
                "total_elements_after": total_elements_after,
                "reduction_count": total_reduced,
                "reduction_percentage": pct(total_reduced, total_elements_before),
                "by_kind_before": kind_before_totals,
                "by_kind_after": kind_after_totals,
            },
            "bibtex": {
                "entries_before": bib_before,
                "entries_after": bib_after,
                "reduction_percentage": pct(bib_before - bib_after, bib_before),
                "renamed_keys": sum(p.get("bibtex_renamed_keys", 0) for p in self.pub_stats),
                "entries_enriched": sum(p.get("bibtex_enriched", 0) for p in self.pub_stats),
                "uncited_bib_entries_dropped": sum(
                    p.get("bibtex_uncited_dropped", 0) for p in self.pub_stats
                ),
            },
            "failures": [
                {"arxiv_id": v["arxiv_id"], "version": v["version"],
                 "status": v["status"], "error": v["error_msg"]}
                for v in self.version_logs
                if v["status"] in {"failed", "partial"}
            ],
        }

    def export_markdown_report(self, output_path: Path) -> str:
        """Generates a Markdown summary of the run."""
        summary = self.generate_summary()
        perf = summary["performance"]
        succ = summary["success_rate"]
        cov = summary["version_coverage"]
        dedup = summary["deduplication"]
        bib = summary["bibtex"]
        total = succ["total_versions_processed"]

        def share(value: int) -> str:
            return f"{round(value / total * 100, 2) if total else 0}%"

        md = ["# Báo cáo Thống kê & Đo lường Hiệu năng Parser (Milestone 2)\n"]

        md.append("## 1. Đo đạc hiệu năng (Performance Benchmark)")
        md.append(f"- **Tổng thời gian thực thi (Total Wall Time)**: `{perf['total_wall_time_sec']}s`")
        md.append(f"- **Thời gian trung bình mỗi publication**: `{perf['avg_time_per_pub_sec']}s`")
        md.append(f"- **Thời gian trung bình mỗi version**: `{perf['avg_time_per_version_sec']}s`")
        md.append(
            f"- **RAM tối đa / trung bình (RSS, {perf['rss_samples']} mẫu)**: "
            f"`{perf['peak_rss_mb']} MB` / `{perf['avg_rss_mb']} MB`"
        )
        if perf["tracemalloc_enabled"]:
            md.append(f"- **Peak tracemalloc**: `{perf['peak_tracemalloc_mb']} MB`")
        if perf["output_size_bytes"] is not None:
            md.append(
                f"- **Dung lượng output (elements/ + bibtex/)**: "
                f"`{perf['output_size_bytes'] / (1024 * 1024):.2f} MB`"
            )
        md.append("")

        md.append("## 2. Tỉ lệ parse thành công & Version Coverage")
        md.append("| Chỉ số | Số lượng | Tỉ lệ (%) |")
        md.append("|---|---|---|")
        md.append(f"| Tổng số publications | {succ['total_publications']} | 100% |")
        md.append(f"| Tổng số versions được xử lý | {total} | 100% |")
        md.append(f"| Versions parse đầy đủ (success) | {succ['successful_versions']} | {share(succ['successful_versions'])} |")
        md.append(f"| Versions parse một phần (partial) | {succ['partial_versions']} | {share(succ['partial_versions'])} |")
        md.append(f"| Versions rỗng (không có TeX source) | {succ['empty_versions']} | {share(succ['empty_versions'])} |")
        md.append(f"| Versions lỗi (failed) | {succ['failed_versions']} | {share(succ['failed_versions'])} |")
        md.append(f"| **Tỉ lệ parse thành công trên versions có TeX** | **{succ['parsed_versions']}** | **{succ['parse_success_rate_percentage']}%** |")
        md.append(f"| **Version Coverage (TeX Source)** | **{cov['versions_with_tex']} / {cov['total_expected_versions']}** | **{cov['coverage_percentage']}%** |\n")

        md.append("## 3. Thống kê Khử trùng lặp Elements (Deduplication)")
        md.append(f"- **Tổng elements trước dedup**: `{dedup['total_elements_before']}`")
        md.append(f"- **Tổng elements sau dedup**: `{dedup['total_elements_after']}`")
        md.append(f"- **Số lượng phần tử trùng lặp đã loại bỏ**: `{dedup['reduction_count']}`")
        md.append(f"- **Tỉ lệ giảm trùng lặp (Reduction rate)**: **`{dedup['reduction_percentage']}%`**\n")

        md.append("### Phân bố Elements theo Kind (Trước vs Sau Dedup)")
        md.append("| Loại phần tử (Kind) | Trước Dedup | Sau Dedup | Giảm (Trùng lặp) | Tỉ lệ giảm (%) |")
        md.append("|---|---|---|---|---|")
        all_kinds = sorted(set(dedup["by_kind_before"]) | set(dedup["by_kind_after"]))
        for k in all_kinds:
            bef = dedup["by_kind_before"].get(k, 0)
            aft = dedup["by_kind_after"].get(k, 0)
            diff = bef - aft
            rate = round(diff / bef * 100.0, 2) if bef > 0 else 0.0
            md.append(f"| `{k}` | {bef} | {aft} | {diff} | {rate}% |")

        md.append("\n## 4. Thống kê BibTeX")
        md.append(f"- **Entries trước / sau dedup**: `{bib['entries_before']}` / `{bib['entries_after']}` (giảm `{bib['reduction_percentage']}%`)")
        md.append(f"- **Citation key được đổi tên**: `{bib['renamed_keys']}`")
        md.append(f"- **Entries được bổ sung trường khi gộp**: `{bib['entries_enriched']}`")
        md.append(f"- **Entries trong .bib không được cite (đã loại)**: `{bib['uncited_bib_entries_dropped']}`")

        if summary["failures"]:
            md.append("\n## 5. Versions failed / partial")
            md.append("| arXiv ID | Version | Trạng thái | Lý do |")
            md.append("|---|---|---|---|")
            for item in summary["failures"]:
                reason = (item["error"] or "").replace("|", "\\|")
                md.append(f"| {item['arxiv_id']} | {item['version']} | {item['status']} | {reason} |")

        md_content = "\n".join(md)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        return md_content
