"""Normalize hierarchy-tree JSON files for every publication version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.hierarchy_normalizer import HierarchyNormalizer


def find_hierarchy_trees(source_dir: Path) -> List[Path]:
    """Find one hierarchy tree per version, preferring the version-local copy."""
    publication_dirs = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "elements"
    )
    tree_files: List[Path] = []

    for publication_dir in publication_dirs:
        elements_dir = publication_dir / "elements"
        version_dirs = sorted(
            (
                path
                for path in publication_dir.iterdir()
                if path.is_dir()
                and path.name.startswith("v")
                and path.name[1:].isdigit()
            ),
            key=lambda path: int(path.name[1:]),
        )

        if version_dirs:
            for version_dir in version_dirs:
                version_tree = version_dir / "hierarchy_tree.json"
                elements_tree = elements_dir / f"hierarchy_tree_{version_dir.name}.json"
                if version_tree.is_file():
                    tree_files.append(version_tree)
                elif elements_tree.is_file():
                    tree_files.append(elements_tree)
        elif elements_dir.is_dir():
            tree_files.extend(
                sorted(elements_dir.glob("hierarchy_tree_v*.json"))
            )

    return tree_files


def normalize_publications(source_dir: Path) -> List[Path]:
    """Normalize each discovered version tree in place."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    tree_files = find_hierarchy_trees(source_dir)
    if not tree_files:
        raise FileNotFoundError(
            f"No per-version hierarchy tree files found under: {source_dir}"
        )

    outputs: List[Path] = []
    for tree_file in tree_files:
        outputs.append(HierarchyNormalizer.normalize_file(tree_file, tree_file))
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
