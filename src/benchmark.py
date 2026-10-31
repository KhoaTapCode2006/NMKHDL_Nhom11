"""Script thực nghiệm và đo lường hiệu năng (Performance Benchmarking) từ đầu đến cuối
cho 2 phương pháp cào dữ liệu arXiv:
1. arxiv_crawler (arXiv API + Source Download + Semantic Scholar API)
2. oai_crawler   (OAI-PMH arXivRaw + Source Download + Semantic Scholar API)

Tập mẫu mặc định: 50 bài báo arXiv từ ID 2510.10788 đến 2510.10837.
Thiết kế tối ưu, tương thích hoàn toàn với môi trường Google Colab (CPU-only) và máy cục bộ.

Cách sử dụng qua CLI:
    python benchmark.py --crawler both --clean
    python benchmark.py --crawler arxiv --clean
    python benchmark.py --crawler oai --clean
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import threading
import time
from pathlib import Path

import arxiv
import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv

import arxiv_crawler
import oai_crawler

load_dotenv()

DEFAULT_YEAR_MONTH = "2510"
DEFAULT_START_SEQ = 10788
DEFAULT_END_SEQ = 10837
RAM_SAMPLE_INTERVAL_SEC = 0.5


class MemoryMonitor:
    """Luồng chạy ngầm lấy mẫu mức tiêu thụ RAM (RSS MB) định kỳ mỗi 0.5 giây."""

    def __init__(self, interval_sec: float = RAM_SAMPLE_INTERVAL_SEC):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.start_time: float = 0.0
        self.baseline_ram_mb: float = 0.0

    def _current_rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def _record_sample(self, crawler_name: str) -> None:
        elapsed = max(0.0, time.perf_counter() - self.start_time)
        rss_mb = self._current_rss_mb()
        self.samples.append({
            "crawler": crawler_name,
            "record_type": "ram_sample",
            "step_index": len(self.samples),
            "elapsed_sec": round(elapsed, 3),
            "ram_mb": round(rss_mb, 3),
            "delta_ram_mb": round(max(0.0, rss_mb - self.baseline_ram_mb), 3),
            "paper_id": "",
            "versions_count": 0,
            "paper_time_sec": np.nan,
            "discovery_time_sec": np.nan,
        })

    def _sampling_loop(self, crawler_name: str) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._record_sample(crawler_name)

    def start(self, crawler_name: str) -> None:
        gc.collect()
        self.samples.clear()
        self._stop_event.clear()
        self.start_time = time.perf_counter()
        self.baseline_ram_mb = self._current_rss_mb()
        self._record_sample(crawler_name)
        self._thread = threading.Thread(
            target=self._sampling_loop,
            args=(crawler_name,),
            daemon=True,
        )
        self._thread.start()

    def stop(self, crawler_name: str) -> tuple[float, float, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._record_sample(crawler_name)
        ram_vals = [s["ram_mb"] for s in self.samples]
        peak_ram = max(ram_vals) if ram_vals else self.baseline_ram_mb
        avg_ram = float(np.mean(ram_vals)) if ram_vals else self.baseline_ram_mb
        return peak_ram, avg_ram, self.baseline_ram_mb


def compute_storage_footprint(save_dir: Path) -> dict[str, int | float]:
    """Đo lường chi tiết dung lượng lưu trữ lý thuyết, đỉnh tạm thời và thực tế trên ổ đĩa."""
    if not save_dir.exists():
        return {
            "papers_crawled": 0,
            "versions_crawled": 0,
            "raw_archive_bytes": 0,
            "total_uncompressed_bytes": 0,
            "filtered_uncompressed_bytes": 0,
            "max_single_archive_bytes": 0,
            "max_disk_storage_bytes": 0,
            "final_output_storage_bytes": 0,
            "storage_saving_pct": 0.0,
        }

    raw_archive_bytes = 0
    total_uncompressed_bytes = 0
    filtered_uncompressed_bytes = 0
    max_single_archive_bytes = 0
    versions_crawled = 0
    paper_ids = set()

    for meta_file in sorted(save_dir.glob("*/v*/metadata.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        versions_crawled += 1
        paper_ids.add(meta_file.parent.parent.name)

        raw_b = int(meta.get("raw_archive_bytes", 0) or 0)
        uncomp_b = int(meta.get("total_uncompressed_bytes", 0) or 0)
        filt_b = int(meta.get("filtered_uncompressed_bytes", 0) or 0)

        if filt_b == 0:
            for fpath in meta_file.parent.rglob("*"):
                if fpath.is_file() and fpath.suffix.lower() in {".tex", ".bib"}:
                    filt_b += fpath.stat().st_size

        raw_archive_bytes += raw_b
        total_uncompressed_bytes += uncomp_b
        filtered_uncompressed_bytes += filt_b
        if raw_b > max_single_archive_bytes:
            max_single_archive_bytes = raw_b

    # Đo thực tế toàn bộ thư mục đầu ra trên ổ đĩa (.tex, .bib, metadata.json, references.json)
    final_output_storage_bytes = sum(
        f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
    )

    # Dung lượng đĩa tối đa cần thiết trong lúc cào = dung lượng đầu ra + gói nén lớn nhất
    max_disk_storage_bytes = final_output_storage_bytes + max_single_archive_bytes

    storage_saving_pct = (
        (1.0 - final_output_storage_bytes / total_uncompressed_bytes) * 100.0
        if total_uncompressed_bytes > 0
        else 0.0
    )

    return {
        "papers_crawled": len(paper_ids),
        "versions_crawled": versions_crawled,
        "raw_archive_bytes": raw_archive_bytes,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "filtered_uncompressed_bytes": filtered_uncompressed_bytes,
        "max_single_archive_bytes": max_single_archive_bytes,
        "max_disk_storage_bytes": max_disk_storage_bytes,
        "final_output_storage_bytes": final_output_storage_bytes,
        "storage_saving_pct": round(storage_saving_pct, 2),
    }


def _prepare_directory(save_dir: Path, clean: bool) -> None:
    """Đảm bảo thư mục lưu trữ sạch khi có cờ --clean để không bị skip bởi cơ chế Resume."""
    if clean and save_dir.exists():
        print(f"[CLEAN] Đang xóa dữ liệu cũ trong '{save_dir}' để đo hiệu năng thực tế...")
        shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)


def benchmark_arxiv_crawler(
    target_ids: list[str],
    save_dir: Path,
    clean: bool = True,
    with_refs: bool = True,
) -> tuple[dict, list[dict]]:
    """Chạy benchmark end-to-end cho phương pháp arxiv_crawler."""
    crawler_name = "arxiv_crawler"
    print("=" * 72)
    print(f"[BENCHMARK] Bắt đầu đo hiệu năng: {crawler_name} ({len(target_ids)} bài báo)")
    print("=" * 72)

    _prepare_directory(save_dir, clean=clean)

    monitor = MemoryMonitor(interval_sec=RAM_SAMPLE_INTERVAL_SEC)
    paper_events: list[dict] = []
    per_paper_times_sec: list[float] = []
    discovery_time_sec = 0.0

    monitor.start(crawler_name)
    wall_start = time.perf_counter()

    paper_idx = 0
    for batch in arxiv_crawler.chunk_list(target_ids, chunk_size=50):
        # 1) Đo thời gian khám phá mục bài báo (Discovery Time)
        t_disc0 = time.perf_counter()
        search = arxiv.Search(id_list=batch)
        papers = list(arxiv_crawler.client.results(search))
        batch_disc_sec = time.perf_counter() - t_disc0
        discovery_time_sec += batch_disc_sec
        per_item_disc_sec = batch_disc_sec / max(1, len(papers))

        # 2) Đo thời gian xử lý trọn vẹn từng bài báo (References + Tải & giải nén các version)
        for paper in papers:
            paper_idx += 1
            t_paper0 = time.perf_counter()

            latest = paper.entry_id.rsplit("/", 1)[-1]
            base_id, _, latest_version = latest.rpartition("v")
            num_versions = int(latest_version) if latest_version.isdigit() else 1
            print(f"[{crawler_name}] ({paper_idx}/{len(target_ids)}) {base_id} (v1..v{num_versions})")

            paper_dir = save_dir / base_id.replace("/", "-")
            ref_file = paper_dir / "v1" / "references.json"
            references = None
            if ref_file.exists():
                try:
                    references = json.loads(ref_file.read_text(encoding="utf-8"))
                except Exception:
                    references = None

            if references is None and with_refs:
                try:
                    references = arxiv_crawler.get_paper_references(base_id)
                except Exception as exc:
                    print(f"  [S2] Lỗi lấy references cho {base_id}: {exc}")
                    references = None

            for version_number in range(1, num_versions + 1):
                try:
                    res = arxiv_crawler.get_source(
                        f"{base_id}v{version_number}",
                        save_dir=save_dir,
                        paper=paper,
                        references=references,
                    )
                    print(f"  {base_id}v{version_number}: {res.get('source_status')}")
                except Exception as exc:
                    print(f"  {base_id}v{version_number}: ERROR {type(exc).__name__}: {exc}")

            paper_elapsed = time.perf_counter() - t_paper0
            per_paper_times_sec.append(paper_elapsed)

            current_ram_mb = monitor.process.memory_info().rss / (1024 * 1024)
            paper_events.append({
                "crawler": crawler_name,
                "record_type": "paper_time",
                "step_index": paper_idx,
                "elapsed_sec": round(time.perf_counter() - wall_start, 3),
                "ram_mb": round(current_ram_mb, 3),
                "delta_ram_mb": round(max(0.0, current_ram_mb - monitor.baseline_ram_mb), 3),
                "paper_id": base_id,
                "versions_count": num_versions,
                "paper_time_sec": round(paper_elapsed, 3),
                "discovery_time_sec": round(per_item_disc_sec, 3),
            })

    total_wall_time_sec = time.perf_counter() - wall_start
    peak_ram_mb, avg_ram_mb, baseline_ram_mb = monitor.stop(crawler_name)
    storage_stats = compute_storage_footprint(save_dir)

    times_arr = np.array(per_paper_times_sec, dtype=float) if per_paper_times_sec else np.array([0.0])
    summary = {
        "crawler": crawler_name,
        "papers_target": len(target_ids),
        "papers_crawled": storage_stats["papers_crawled"],
        "versions_crawled": storage_stats["versions_crawled"],
        "total_wall_time_sec": round(total_wall_time_sec, 3),
        "discovery_time_sec": round(discovery_time_sec, 3),
        "total_paper_process_time_sec": round(float(times_arr.sum()), 3),
        "avg_paper_time_sec": round(float(times_arr.mean()), 3),
        "median_paper_time_sec": round(float(np.median(times_arr)), 3),
        "min_paper_time_sec": round(float(times_arr.min()), 3),
        "max_paper_time_sec": round(float(times_arr.max()), 3),
        "std_paper_time_sec": round(float(times_arr.std()), 3),
        "baseline_ram_mb": round(baseline_ram_mb, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "avg_ram_mb": round(avg_ram_mb, 2),
        "ram_overhead_mb": round(max(0.0, peak_ram_mb - baseline_ram_mb), 2),
        **storage_stats,
    }

    timeseries_records = monitor.samples + paper_events
    return summary, timeseries_records


def benchmark_oai_crawler(
    target_ids: list[str],
    save_dir: Path,
    clean: bool = True,
    with_refs: bool = True,
) -> tuple[dict, list[dict]]:
    """Chạy benchmark end-to-end cho phương pháp oai_crawler."""
    crawler_name = "oai_crawler"
    print("=" * 72)
    print(f"[BENCHMARK] Bắt đầu đo hiệu năng: {crawler_name} ({len(target_ids)} bài báo)")
    print("=" * 72)

    _prepare_directory(save_dir, clean=clean)

    # Reset bộ đếm throttle của OAI trước khi chạy
    oai_crawler._last_request_ts = 0.0

    monitor = MemoryMonitor(interval_sec=RAM_SAMPLE_INTERVAL_SEC)
    paper_events: list[dict] = []
    per_paper_times_sec: list[float] = []
    discovery_time_sec = 0.0

    monitor.start(crawler_name)
    wall_start = time.perf_counter()

    for paper_idx, arxiv_id in enumerate(target_ids, start=1):
        print(f"[{crawler_name}] ({paper_idx}/{len(target_ids)}) {arxiv_id}")

        # 1) Đo thời gian khám phá OAI-PMH (fetch_oai_record + extract_versions_from_oai)
        t_disc0 = time.perf_counter()
        try:
            xml_root = oai_crawler.fetch_oai_record(arxiv_id)
            versions = oai_crawler.extract_versions_from_oai(xml_root)
        except Exception as exc:
            print(f"  skipped OAI discovery: {type(exc).__name__}: {exc}")
            discovery_time_sec += time.perf_counter() - t_disc0
            continue
        item_disc_sec = time.perf_counter() - t_disc0
        discovery_time_sec += item_disc_sec

        if not versions:
            continue

        # 2) Đo thời gian xử lý trọn vẹn bài báo (Semantic Scholar references + tải & giải nén các version)
        t_paper0 = time.perf_counter()
        oai_crawler.crawl_paper(versions, save_dir=save_dir, with_refs=with_refs)
        paper_elapsed = time.perf_counter() - t_paper0
        per_paper_times_sec.append(paper_elapsed)

        current_ram_mb = monitor.process.memory_info().rss / (1024 * 1024)
        paper_events.append({
            "crawler": crawler_name,
            "record_type": "paper_time",
            "step_index": paper_idx,
            "elapsed_sec": round(time.perf_counter() - wall_start, 3),
            "ram_mb": round(current_ram_mb, 3),
            "delta_ram_mb": round(max(0.0, current_ram_mb - monitor.baseline_ram_mb), 3),
            "paper_id": versions[0]["base_id"],
            "versions_count": len(versions),
            "paper_time_sec": round(paper_elapsed, 3),
            "discovery_time_sec": round(item_disc_sec, 3),
        })

    total_wall_time_sec = time.perf_counter() - wall_start
    peak_ram_mb, avg_ram_mb, baseline_ram_mb = monitor.stop(crawler_name)
    storage_stats = compute_storage_footprint(save_dir)

    times_arr = np.array(per_paper_times_sec, dtype=float) if per_paper_times_sec else np.array([0.0])
    summary = {
        "crawler": crawler_name,
        "papers_target": len(target_ids),
        "papers_crawled": storage_stats["papers_crawled"],
        "versions_crawled": storage_stats["versions_crawled"],
        "total_wall_time_sec": round(total_wall_time_sec, 3),
        "discovery_time_sec": round(discovery_time_sec, 3),
        "total_paper_process_time_sec": round(float(times_arr.sum()), 3),
        "avg_paper_time_sec": round(float(times_arr.mean()), 3),
        "median_paper_time_sec": round(float(np.median(times_arr)), 3),
        "min_paper_time_sec": round(float(times_arr.min()), 3),
        "max_paper_time_sec": round(float(times_arr.max()), 3),
        "std_paper_time_sec": round(float(times_arr.std()), 3),
        "baseline_ram_mb": round(baseline_ram_mb, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "avg_ram_mb": round(avg_ram_mb, 2),
        "ram_overhead_mb": round(max(0.0, peak_ram_mb - baseline_ram_mb), 2),
        **storage_stats,
    }

    timeseries_records = monitor.samples + paper_events
    return summary, timeseries_records


def save_benchmark_outputs(
    summaries: list[dict],
    timeseries: list[dict],
    results_csv: Path,
    timeseries_csv: Path,
) -> None:
    """Lưu kết quả tổng hợp và chuỗi thời gian vào file CSV (cập nhật hoặc ghi mới)."""
    new_res_df = pd.DataFrame(summaries)
    if results_csv.exists() and not new_res_df.empty:
        try:
            old_res_df = pd.read_csv(results_csv)
            old_res_df = old_res_df[~old_res_df["crawler"].isin(new_res_df["crawler"])]
            new_res_df = pd.concat([old_res_df, new_res_df], ignore_index=True)
        except Exception:
            pass
    new_res_df.to_csv(results_csv, index=False, encoding="utf-8")

    new_ts_df = pd.DataFrame(timeseries)
    if timeseries_csv.exists() and not new_ts_df.empty:
        try:
            old_ts_df = pd.read_csv(timeseries_csv)
            old_ts_df = old_ts_df[~old_ts_df["crawler"].isin(new_ts_df["crawler"])]
            new_ts_df = pd.concat([old_ts_df, new_ts_df], ignore_index=True)
        except Exception:
            pass
    new_ts_df.to_csv(timeseries_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"[HOÀN TẤT] Đã lưu kết quả tổng hợp tại : {results_csv}")
    print(f"[HOÀN TẤT] Đã lưu dữ liệu timeseries tại: {timeseries_csv}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking hiệu năng end-to-end cho arxiv_crawler và oai_crawler."
    )
    parser.add_argument(
        "--crawler",
        choices=["arxiv", "oai", "both"],
        default="both",
        help="Chọn phương pháp crawler cần đo hiệu năng (mặc định: both).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Xóa dữ liệu cũ trong thư mục lưu trữ trước khi bấm giờ để tránh skip bài.",
    )
    parser.add_argument(
        "--no-refs",
        action="store_true",
        help="Bỏ qua bước gọi Semantic Scholar API nếu chỉ muốn đo tốc độ tải source.",
    )
    parser.add_argument(
        "--year-month",
        default=DEFAULT_YEAR_MONTH,
        help="Tháng/năm của tập ID mẫu (mặc định: 2510).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START_SEQ,
        help="Số thứ tự bài báo bắt đầu (mặc định: 10788).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DEFAULT_END_SEQ,
        help="Số thứ tự bài báo kết thúc (mặc định: 10837).",
    )
    parser.add_argument(
        "--arxiv-dir",
        default="./sourcesArxiv",
        help="Thư mục đầu ra cho arxiv_crawler (mặc định: ./sourcesArxiv).",
    )
    parser.add_argument(
        "--oai-dir",
        default="./sourcesOAI",
        help="Thư mục đầu ra cho oai_crawler (mặc định: ./sourcesOAI).",
    )
    parser.add_argument(
        "--results-csv",
        default="benchmark_results.csv",
        help="Đường dẫn file CSV lưu kết quả tổng hợp.",
    )
    parser.add_argument(
        "--timeseries-csv",
        default="benchmark_timeseries.csv",
        help="Đường dẫn file CSV lưu mẫu RAM và thời gian xử lý từng bài.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_ids = [
        f"{args.year_month}.{seq:05d}" for seq in range(args.start, args.end + 1)
    ]
    with_refs = not args.no_refs

    summaries: list[dict] = []
    timeseries: list[dict] = []

    if args.crawler in {"arxiv", "both"}:
        summary_arxiv, ts_arxiv = benchmark_arxiv_crawler(
            target_ids=target_ids,
            save_dir=Path(args.arxiv_dir),
            clean=args.clean,
            with_refs=with_refs,
        )
        summaries.append(summary_arxiv)
        timeseries.extend(ts_arxiv)

    if args.crawler in {"oai", "both"}:
        summary_oai, ts_oai = benchmark_oai_crawler(
            target_ids=target_ids,
            save_dir=Path(args.oai_dir),
            clean=args.clean,
            with_refs=with_refs,
        )
        summaries.append(summary_oai)
        timeseries.extend(ts_oai)

    save_benchmark_outputs(
        summaries=summaries,
        timeseries=timeseries,
        results_csv=Path(args.results_csv),
        timeseries_csv=Path(args.timeseries_csv),
    )


if __name__ == "__main__":
    main()

