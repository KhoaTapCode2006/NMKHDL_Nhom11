"""
Script thực thi nhiệm vụ Khử Trùng Lặp (Deduplication - Phần của Kiệt) cho Milestone 2:
1. Khử trùng lặp Reference Entries (BibTeX & \\bibitem) trong cùng version và xuyên version,
   hợp nhất các trường thông tin (field unionization) và lưu vào thư mục `bibtex/`.
2. Đổi tên toàn bộ các lệnh trích dẫn `\\cite{...}` trong `elements` sang `citation_key` chuẩn.
3. Khử trùng lặp Full-text Content (`elements`) xuyên version, bảo toàn cấu trúc cây phân cấp,
   cập nhật bảng `hierarchy` và cập nhật ngược lại vào cây `hierarchy_tree.json` của từng version.
4. Xuất báo cáo thống kê tổng hợp `src/dedup_summary_stats.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.dedup import PublicationDeduplicator


def resolve_default_source_dir() -> Path:
    """Tự động tìm thư mục chứa dữ liệu bài báo (sourcesArxiv hoặc source)."""
    candidates = [
        curr_dir.parent / "sourcesArxiv",
        curr_dir / "sourcesArxiv",
        curr_dir.parent / "source",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2 — Chạy khử trùng lặp Reference Entries & Full-text Content (Phần của Kiệt)"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(resolve_default_source_dir()),
        help="Đường dẫn tới thư mục chứa các bài báo (mặc định: ./sourcesArxiv)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=str(curr_dir / "dedup_summary_stats.json"),
        help="Đường dẫn lưu file thống kê tổng hợp Deduplication (mặc định: src/dedup_summary_stats.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng bài báo cần xử lý (tùy chọn)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_dir).resolve()

    if not source_path.is_dir():
        print(f"Lỗi: Thư mục nguồn không tồn tại: {source_path}")
        sys.exit(1)

    pub_dirs = sorted(
        [
            d
            for d in source_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in ("elements", "bibtex")
        ]
    )
    if args.limit:
        pub_dirs = pub_dirs[: args.limit]

    print("=" * 78)
    print("  MILESTONE 2: REFERENCE & FULL-TEXT DEDUPLICATION (NHIỆM VỤ: KIỆT)")
    print(f"  Thư mục dữ liệu: {source_path}")
    print(f"  Tổng số bài báo: {len(pub_dirs)}")
    print("=" * 78)

    start_time = time.perf_counter()

    pub_results: List[Dict[str, Any]] = []
    total_elems_before = 0
    total_elems_after = 0
    total_bib_before = 0
    total_bib_after = 0
    total_renamed_keys = 0
    total_cite_cmds_updated = 0
    total_enriched_entries = 0
    by_kind_before_global: Dict[str, int] = {}
    by_kind_after_global: Dict[str, int] = {}

    for idx, p_dir in enumerate(pub_dirs, 1):
        res = PublicationDeduplicator.deduplicate_and_save_publication(
            pub_dir=p_dir,
            clean_before_match=True,
        )
        pub_results.append(res)

        e_stat = res["elements_dedup"]
        b_stat = res["bibtex_dedup"]

        total_elems_before += e_stat["total_before"]
        total_elems_after += e_stat["total_after"]
        total_bib_before += b_stat["total_raw_entries"]
        total_bib_after += b_stat["dedup_entries_count"]
        total_renamed_keys += b_stat["renamed_keys_count"]
        total_cite_cmds_updated += e_stat.get("cite_commands_updated", 0)
        total_enriched_entries += b_stat.get("fields_enriched_count", 0)

        for k, cnt in e_stat.get("by_kind_before", {}).items():
            by_kind_before_global[k] = by_kind_before_global.get(k, 0) + cnt
        for k, cnt in e_stat.get("by_kind_after", {}).items():
            by_kind_after_global[k] = by_kind_after_global.get(k, 0) + cnt

        print(
            f"[{idx:02d}/{len(pub_dirs):02d}] {res['arxiv_id']} ({res['total_versions']}v) | "
            f"Elements: {e_stat['total_before']} -> {e_stat['total_after']} (-{e_stat['reduction_percentage']}%) | "
            f"BibTeX: {b_stat['total_raw_entries']} -> {b_stat['dedup_entries_count']} (-{b_stat['reduction_percentage']}%)"
        )

    elapsed = time.perf_counter() - start_time
    elems_removed = total_elems_before - total_elems_after
    elems_red_pct = (
        round(elems_removed / total_elems_before * 100.0, 2)
        if total_elems_before > 0
        else 0.0
    )
    bib_removed = total_bib_before - total_bib_after
    bib_red_pct = (
        round(bib_removed / total_bib_before * 100.0, 2)
        if total_bib_before > 0
        else 0.0
    )

    summary_payload = {
        "execution_time_sec": round(elapsed, 2),
        "total_publications": len(pub_dirs),
        "multi_version_publications": sum(1 for r in pub_results if r["total_versions"] > 1),
        "total_versions": sum(r["total_versions"] for r in pub_results),
        "tex_versions": sum(r["tex_versions"] for r in pub_results),
        "empty_versions": sum(r["empty_versions"] for r in pub_results),
        "fulltext_deduplication": {
            "total_elements_before": total_elems_before,
            "total_elements_after": total_elems_after,
            "removed_duplicates": elems_removed,
            "reduction_percentage": elems_red_pct,
            "by_kind_before": by_kind_before_global,
            "by_kind_after": by_kind_after_global,
        },
        "reference_deduplication": {
            "total_entries_before": total_bib_before,
            "total_entries_after": total_bib_after,
            "removed_duplicates": bib_removed,
            "reduction_percentage": bib_red_pct,
            "renamed_citation_keys": total_renamed_keys,
            "cite_commands_updated_in_text": total_cite_cmds_updated,
            "entries_enriched_by_field_union": total_enriched_entries,
        },
        "publications": pub_results,
    }

    summary_path = Path(args.summary_json).resolve()
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print(f"  Hoàn thành khử trùng lặp {len(pub_dirs)} bài báo trong {elapsed:.2f}s")
    print(
        f"  [Full-text Elements] Trước: {total_elems_before:,} -> Sau: {total_elems_after:,} "
        f"(Giảm {elems_removed:,} phần tử, đạt {elems_red_pct}%)"
    )
    print(
        f"  [Reference Entries]  Trước: {total_bib_before:,} -> Sau: {total_bib_after:,} "
        f"(Giảm {bib_removed:,} entries, đạt {bib_red_pct}%)"
    )
    print(
        f"  [Citation Updates]   Đổi tên {total_renamed_keys} keys trùng | "
        f"Cập nhật {total_cite_cmds_updated} lệnh \\cite{{}} | "
        f"Hợp nhất bổ sung trường cho {total_enriched_entries} entries"
    )
    print(f"  Đã lưu thống kê tổng hợp tại: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
