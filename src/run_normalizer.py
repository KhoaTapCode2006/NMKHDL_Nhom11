"""Normalize hierarchy-tree JSON files for every publication version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.pipeline import ParsingPipeline


def find_hierarchy_trees(source_dir: Path) -> List[Path]:
    """Compatibility wrapper for discovering generated per-version trees."""
    return ParsingPipeline.find_hierarchy_trees(source_dir)


def normalize_publications(source_dir: Path) -> List[Path]:
    """Normalize each discovered version tree in place."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    outputs = ParsingPipeline.normalize_source(source_dir)
    if not outputs:
        raise FileNotFoundError(
            f"No per-version hierarchy tree files found under: {source_dir}"
        )
    return outputs


def main(argv: List[str] | None = None) -> int:
    default_source = curr_dir / "sourcesArxiv"
    if not default_source.exists():
        default_source = curr_dir.parent / "source"

    parser = argparse.ArgumentParser(
        description="Normalize hierarchy trees for all publication versions."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="Directory containing publication folders (default: src/sourcesArxiv or ./source).",
    )
    args = parser.parse_args(argv)

    outputs = normalize_publications(args.source_dir)
    print(f"Normalized {len(outputs)} hierarchy tree(s).")
    for output_file in outputs:
        print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
