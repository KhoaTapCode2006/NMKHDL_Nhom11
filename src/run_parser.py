"""
CLI Runner for Milestone 2 Multi-File Gathering, Hierarchy Parsing, and Benchmarking.
Executes the parser pipeline, generates elements directories, cleans redundant files,
and produces comprehensive statistics reports for Jupyter Book.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add current directory to path
curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.benchmark import ParserBenchmark
from parser.pipeline import ParsingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Milestone 2 LaTeX Parser, Multi-File Gathering & Statistics Runner"
    )
    # Default to sourcesArxiv or ./source
    default_source = curr_dir / "sourcesArxiv"
    if not default_source.exists():
        default_source = curr_dir.parent / "source"

    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(default_source),
        help="Path to publications source directory (default: src/sourcesArxiv or ./source)",
    )
    parser.add_argument(
        "--clean-tex",
        action="store_true",
        default=True,
        help="Delete redundant/orphan .tex files once parsed into elements/ (default: True)",
    )
    parser.add_argument(
        "--no-clean-tex",
        dest="clean_tex",
        action="store_false",
        help="Keep original .tex files in version directories",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default=str(curr_dir / "benchmark_report.md"),
        help="Path to output Markdown benchmark report (default: src/benchmark_report.md)",
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default=str(curr_dir / "benchmark_stats.json"),
        help="Path to output JSON statistics (default: src/benchmark_stats.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of publications to process (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.source_dir).resolve()

    if not source_path.exists():
        print(f"❌ Error: Source directory does not exist: {source_path}")
        sys.exit(1)

    benchmark = ParserBenchmark()
    pipeline = ParsingPipeline(
        source_dir=source_path,
        clean_redundant_tex=args.clean_tex,
        benchmark=benchmark,
    )

    pub_dirs = sorted([
        d for d in source_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "elements"
    ])

    if args.limit:
        pub_dirs = pub_dirs[: args.limit]

    print(f"============================================================")
    print(f"  Milestone 2: LaTeX Multi-File Gathering & Statistics")
    print(f"  Target Source Directory: {source_path}")
    print(f"  Total Publications: {len(pub_dirs)}")
    print(f"  Redundant TeX Cleanup: {'ENABLED' if args.clean_tex else 'DISABLED'}")
    print(f"============================================================")

    benchmark.start()

    for idx, p_dir in enumerate(pub_dirs, 1):
        res = pipeline.process_publication(p_dir)
        d_stats = res["dedup_stats"]
        print(
            f"[{idx:02d}/{len(pub_dirs):02d}] {res['arxiv_id']} -> "
            f"{res['elements_count']} unique elements | "
            f"before: {d_stats['total_before']} -> after: {d_stats['total_after']} "
            f"({d_stats['reduction_percentage']}% reduced)"
        )

    benchmark.stop()
    summary = benchmark.generate_summary()

    # Export markdown report
    md_content = benchmark.export_markdown_report(Path(args.report_md))
    print(f"\n✅ Markdown report generated: {args.report_md}")

    # Export JSON statistics
    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Statistics JSON generated: {args.stats_json}")

    print("\n" + md_content)


if __name__ == "__main__":
    main()
