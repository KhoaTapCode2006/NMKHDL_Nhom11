"""Compatibility runner for the unified publication parsing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.pipeline import ParsingPipeline


def process_publication_hierarchy(pub_dir: Path) -> Dict[str, Any]:
    """Compatibility API that runs all stages and returns hierarchy statistics."""
    pub_dir = Path(pub_dir).resolve()
    result = ParsingPipeline(
        source_dir=pub_dir.parent,
        clean_redundant_tex=False,
    ).process_publication(pub_dir)
    return {
        **result,
        "hierarchy_edges_count": result["hierarchy_count"],
        "bibtex_entries_count": result["bibtex_entries_count"],
    }


def main(argv: List[str] | None = None) -> int:
    default_source = curr_dir / "sourcesArxiv"
    if not default_source.exists():
        default_source = curr_dir.parent / "source"

    parser = argparse.ArgumentParser(
        description="Run the unified gather, hierarchy, normalize, and dedup pipeline."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="Directory containing publication folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of publications processed.",
    )
    args = parser.parse_args(argv)
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    pub_dirs = ParsingPipeline.find_publications(source_dir)
    if args.limit is not None:
        pub_dirs = pub_dirs[: args.limit]

    for index, pub_dir in enumerate(pub_dirs, 1):
        result = process_publication_hierarchy(pub_dir)
        print(
            f"[{index:02d}/{len(pub_dirs):02d}] {result['arxiv_id']} | "
            f"versions: {result['tex_versions']} parsed, "
            f"{result['empty_versions']} empty, {result['failed_versions']} failed | "
            f"elements: {result['elements_count']} | "
            f"BibTeX: {result['bibtex_entries_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
