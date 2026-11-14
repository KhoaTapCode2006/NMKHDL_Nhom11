"""End-to-end orchestration for gathering and parsing publication sources."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import ParserBenchmark
from .dedup import PublicationDeduplicator
from .gatherer import MultiFileGatherer
from .hierarchy_builder import HierarchyBuilder
from .hierarchy_normalizer import HierarchyNormalizer

logger = logging.getLogger(__name__)


class ParsingPipeline:
    """Orchestrate gathering, artifact persistence, hierarchy, normalization, and dedup."""

    def __init__(
        self,
        source_dir: str | Path,
        clean_redundant_tex: bool = True,
        benchmark: Optional[ParserBenchmark] = None,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.clean_redundant_tex = clean_redundant_tex
        self.benchmark = benchmark or ParserBenchmark()

    @staticmethod
    def _version_key(version_name: str) -> int:
        if version_name.startswith("v") and version_name[1:].isdigit():
            return int(version_name[1:])
        if version_name.isdigit():
            return int(version_name)
        return 0

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Unable to read JSON file %s: %s", path, error)
            return None

    @classmethod
    def find_publications(cls, source_dir: Path) -> List[Path]:
        """Find publication folders while excluding generated top-level stores."""
        source_dir = Path(source_dir)
        return sorted(
            (
                path
                for path in source_dir.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and path.name not in {"elements", "bibtex"}
            ),
            key=lambda path: path.name,
        )

    @classmethod
    def find_hierarchy_trees(cls, source_dir: Path) -> List[Path]:
        """Find per-version trees, preferring version-local copies."""
        tree_files: List[Path] = []
        for publication_dir in cls.find_publications(source_dir):
            elements_dir = publication_dir / "elements"
            version_dirs = sorted(
                (
                    path
                    for path in publication_dir.iterdir()
                    if path.is_dir()
                    and path.name.startswith("v")
                    and path.name[1:].isdigit()
                ),
                key=lambda path: cls._version_key(path.name),
            )
            if version_dirs:
                for version_dir in version_dirs:
                    version_tree = version_dir / "hierarchy_tree.json"
                    elements_tree = (
                        elements_dir / f"hierarchy_tree_{version_dir.name}.json"
                    )
                    if version_tree.is_file():
                        tree_files.append(version_tree)
                    elif elements_tree.is_file():
                        tree_files.append(elements_tree)
            elif elements_dir.is_dir():
                tree_files.extend(
                    sorted(elements_dir.glob("hierarchy_tree_v*.json"))
                )
        return tree_files

    def process_version(
        self,
        arxiv_id: str,
        version_name: str,
        version_dir: Path,
        metadata: Optional[dict] = None,
        inlined_path: Optional[Path] = None,
        previous_manifest: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Gather and parse one version, retaining the result for pipeline persistence."""
        start_time = time.perf_counter()
        gather_res = MultiFileGatherer.gather(version_dir, metadata=metadata)
        has_source_tex = version_dir.is_dir() and any(
            version_dir.rglob("*.tex")
        )
        reused_inlined_content = False

        # A previous successful run may already have removed the source .tex files.
        if (
            not gather_res["success"]
            and not has_source_tex
            and inlined_path
            and inlined_path.is_file()
        ):
            was_successful = (
                previous_manifest is None
                or previous_manifest.get("status") == "success"
            )
            if was_successful:
                inlined_content = inlined_path.read_text(encoding="utf-8")
                if inlined_content.strip():
                    reused_inlined_content = True
                    root_name = (
                        previous_manifest.get("root_file")
                        if previous_manifest
                        else None
                    )
                    gather_res = {
                        **gather_res,
                        "success": True,
                        "is_complete": True,
                        "root_file": Path(root_name) if root_name else None,
                        "compilation_files": [],
                        "orphan_files": [],
                        "inlined_content": inlined_content,
                        "unresolved_includes": [],
                        "cycle_errors": [],
                        "read_errors": [],
                        "error": None,
                    }

        if not gather_res["success"]:
            error = gather_res["error"] or "Gather failed without an error message."
            status = (
                "empty_version"
                if "No .tex files found" in error
                else "failed"
            )
            self.benchmark.record_version(
                arxiv_id=arxiv_id,
                version=version_name,
                status=status,
                duration_sec=time.perf_counter() - start_time,
                error_msg=error,
            )
            return {
                "success": False,
                "status": status,
                "error": error,
                "gather_res": gather_res,
                "elements": [],
                "hierarchy": [],
                "preserved_references": {},
                "reused_inlined_content": False,
            }

        try:
            elements, hierarchy, preserved_references = HierarchyBuilder.parse_version(
                arxiv_id=arxiv_id,
                version=version_name,
                inlined_tex=gather_res["inlined_content"],
            )
        except Exception as error:
            error_message = f"Hierarchy parsing failed: {error}"
            self.benchmark.record_version(
                arxiv_id=arxiv_id,
                version=version_name,
                status="failed",
                duration_sec=time.perf_counter() - start_time,
                error_msg=error_message,
                compilation_files_count=len(gather_res["compilation_files"]),
            )
            return {
                "success": False,
                "status": "failed",
                "error": error_message,
                "gather_res": gather_res,
                "elements": [],
                "hierarchy": [],
                "preserved_references": {},
                "reused_inlined_content": False,
            }

        self.benchmark.record_version(
            arxiv_id=arxiv_id,
            version=version_name,
            status="success",
            duration_sec=time.perf_counter() - start_time,
            compilation_files_count=len(gather_res["compilation_files"]),
            elements_count=len(elements),
        )
        return {
            "success": True,
            "status": "success",
            "error": None,
            "gather_res": gather_res,
            "elements": elements,
            "hierarchy": hierarchy,
            "preserved_references": preserved_references,
            "reused_inlined_content": reused_inlined_content,
        }

    def build_publication(self, pub_dir: Path) -> Dict[str, Any]:
        """Gather sources, persist inline TeX/manifests, and build pre-dedup trees.

        MultiFileGatherer only returns in-memory results. This method owns the
        publication-level files consumed by the later normalization and dedup stages.
        """
        pub_dir = Path(pub_dir).resolve()
        if not pub_dir.is_dir():
            raise FileNotFoundError(f"Publication directory does not exist: {pub_dir}")

        arxiv_id = pub_dir.name
        elements_dir = pub_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = elements_dir / "compilation_manifest.json"
        old_manifest = self._read_json(manifest_path)
        if not isinstance(old_manifest, dict):
            old_manifest = {}

        version_dirs = sorted(
            (
                path
                for path in pub_dir.iterdir()
                if path.is_dir()
                and path.name.startswith("v")
                and path.name[1:].isdigit()
            ),
            key=lambda path: self._version_key(path.name),
        )
        if version_dirs:
            versions: List[Tuple[str, Path]] = [
                (path.name, path) for path in version_dirs
            ]
        else:
            versions = [
                (
                    inline_path.stem.replace("full_inlined_", ""),
                    pub_dir / inline_path.stem.replace("full_inlined_", ""),
                )
                for inline_path in sorted(elements_dir.glob("full_inlined_v*.tex"))
            ]

        raw_references_by_version: Dict[str, Any] = {}
        compilation_manifest: Dict[str, Any] = {}
        gather_results: Dict[str, Dict[str, Any]] = {}
        successful_versions = 0
        empty_versions = 0
        failed_versions = 0

        for version_name, version_dir in versions:
            inline_path = elements_dir / f"full_inlined_{version_name}.tex"
            old_record = old_manifest.get(version_name)
            if not isinstance(old_record, dict):
                old_record = None

            metadata = None
            metadata_path = version_dir / "metadata.json"
            if metadata_path.is_file():
                metadata_value = self._read_json(metadata_path)
                if isinstance(metadata_value, dict):
                    metadata = metadata_value

            result = self.process_version(
                arxiv_id=arxiv_id,
                version_name=version_name,
                version_dir=version_dir,
                metadata=metadata,
                inlined_path=inline_path,
                previous_manifest=old_record,
            )
            gather_res = result["gather_res"]

            if result["success"]:
                successful_versions += 1
                gather_results[version_name] = gather_res
                inline_path.write_text(
                    gather_res["inlined_content"],
                    encoding="utf-8",
                )
                output = HierarchyBuilder.build_version_hierarchy_outputs(
                    arxiv_id=arxiv_id,
                    version_name=version_name,
                    raw_elements=result["elements"],
                    raw_hierarchy=result["hierarchy"],
                    is_empty=False,
                    status="success",
                )
                self._write_json(
                    elements_dir / f"pre_dedup_elements_{version_name}.json",
                    result["elements"],
                )
                self._write_json(
                    elements_dir / f"pre_dedup_hierarchy_{version_name}.json",
                    result["hierarchy"],
                )
                compilation_manifest[version_name] = {
                    "status": "success",
                    "is_empty": False,
                    "root_file": (
                        (
                            old_record.get("root_file")
                            if old_record
                            else None
                        )
                        if result["reused_inlined_content"]
                        else (
                            gather_res["root_file"].name
                            if gather_res["root_file"]
                            else None
                        )
                    ),
                    "compilation_files": (
                        old_record.get("compilation_files", []) if old_record else []
                    )
                    if result["reused_inlined_content"]
                    else [path.name for path in gather_res["compilation_files"]],
                    "orphan_files": (
                        old_record.get("orphan_files", []) if old_record else []
                    )
                    if result["reused_inlined_content"]
                    else [path.name for path in gather_res["orphan_files"]],
                }
                raw_references_by_version[version_name] = {
                    "status": "success",
                    **result["preserved_references"],
                }
            else:
                status = result["status"]
                is_empty = status == "empty_version"
                empty_versions += int(is_empty)
                failed_versions += int(not is_empty)
                output = HierarchyBuilder.build_version_hierarchy_outputs(
                    arxiv_id=arxiv_id,
                    version_name=version_name,
                    raw_elements=[],
                    raw_hierarchy=[],
                    is_empty=is_empty,
                    error_msg=result["error"],
                    status=status,
                )
                self._write_json(
                    elements_dir / f"pre_dedup_elements_{version_name}.json",
                    [],
                )
                self._write_json(
                    elements_dir / f"pre_dedup_hierarchy_{version_name}.json",
                    [],
                )
                compilation_manifest[version_name] = {
                    "status": status,
                    "is_empty": is_empty,
                    "root_file": None,
                    "compilation_files": [],
                    "orphan_files": [
                        path.name
                        for path in gather_res.get("orphan_files", [])
                    ],
                    "error": result["error"],
                }
                raw_references_by_version[version_name] = {
                    "status": status,
                    "thebibliography_blocks": [],
                    "bibitem_entries": [],
                    "bibliography_commands": [],
                    "reference_sections": [],
                }

            self._write_json(
                elements_dir / f"elements_{version_name}.json",
                output["elements"],
            )
            self._write_json(
                elements_dir / f"hierarchy_{version_name}.json",
                output["hierarchy"],
            )
            self._write_json(
                elements_dir / f"hierarchy_tree_{version_name}.json",
                output["tree"],
            )
            if version_dir.is_dir():
                self._write_json(version_dir / "hierarchy_tree.json", output["tree"])

        self._write_json(manifest_path, compilation_manifest)
        self._write_json(
            elements_dir / "raw_references.json",
            raw_references_by_version,
        )
        return {
            "arxiv_id": arxiv_id,
            "total_versions": len(versions),
            "successful_versions": successful_versions,
            "empty_versions": empty_versions,
            "failed_versions": failed_versions,
            "version_names": [name for name, _ in versions],
            "gather_results": gather_results,
            "compilation_manifest": compilation_manifest,
        }

    @classmethod
    def normalize_publication(cls, pub_dir: Path) -> List[Path]:
        """Normalize generated hierarchy trees in both publication locations."""
        pub_dir = Path(pub_dir).resolve()
        elements_dir = pub_dir / "elements"
        version_dirs = sorted(
            (
                path
                for path in pub_dir.iterdir()
                if path.is_dir()
                and path.name.startswith("v")
                and path.name[1:].isdigit()
            ),
            key=lambda path: cls._version_key(path.name),
        )

        tree_files: List[Path] = []
        if version_dirs:
            for version_dir in version_dirs:
                local_tree = version_dir / "hierarchy_tree.json"
                shared_tree = elements_dir / f"hierarchy_tree_{version_dir.name}.json"
                if local_tree.is_file():
                    tree_files.append(local_tree)
                if shared_tree.is_file():
                    tree_files.append(shared_tree)
        elif elements_dir.is_dir():
            tree_files.extend(sorted(elements_dir.glob("hierarchy_tree_v*.json")))

        outputs: List[Path] = []
        for tree_file in dict.fromkeys(tree_files):
            outputs.append(HierarchyNormalizer.normalize_file(tree_file, tree_file))
        return outputs

    @classmethod
    def normalize_source(cls, source_dir: Path) -> List[Path]:
        """Normalize generated trees across all publications under a source root."""
        source_dir = Path(source_dir).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        return [
            tree_file
            for pub_dir in cls.find_publications(source_dir)
            for tree_file in cls.normalize_publication(pub_dir)
        ]

    @staticmethod
    def deduplicate_publication(
        pub_dir: Path,
        clean_before_match: bool = True,
    ) -> Dict[str, Any]:
        """Run reference, citation, and full-text deduplication for one paper."""
        return PublicationDeduplicator.deduplicate_and_save_publication(
            pub_dir=pub_dir,
            clean_before_match=clean_before_match,
        )

    def process_publication(self, pub_dir: Path) -> Dict[str, Any]:
        """Run all four pipeline stages and persist their combined outputs."""
        publication_start = time.perf_counter()
        build_result = self.build_publication(pub_dir)
        normalized_trees = self.normalize_publication(pub_dir)
        dedup_result = self.deduplicate_publication(pub_dir)

        deleted_files: Dict[str, List[str]] = {}
        if self.clean_redundant_tex:
            for version_name, gather_res in build_result["gather_results"].items():
                version_dir = Path(pub_dir) / version_name
                deleted = MultiFileGatherer.clean_redundant_files(
                    version_dir=version_dir,
                    compilation_files=gather_res["compilation_files"],
                    orphan_files=gather_res["orphan_files"],
                    keep_root=False,
                )
                deleted_files[version_name] = [path.name for path in deleted]

        element_stats = dedup_result["elements_dedup"]
        duration = time.perf_counter() - publication_start
        self.benchmark.record_publication(
            arxiv_id=build_result["arxiv_id"],
            expected_versions=build_result["total_versions"],
            crawled_versions=build_result["total_versions"],
            tex_versions=build_result["successful_versions"],
            empty_versions=build_result["empty_versions"],
            dedup_stats=element_stats,
            duration_sec=duration,
        )
        return {
            "arxiv_id": build_result["arxiv_id"],
            "elements_count": dedup_result["elements_count"],
            "hierarchy_count": dedup_result["hierarchy_count"],
            "bibtex_entries_count": dedup_result["bibtex_count"],
            "dedup_stats": element_stats,
            "bibtex_stats": dedup_result["bibtex_dedup"],
            "total_versions": build_result["total_versions"],
            "tex_versions": build_result["successful_versions"],
            "empty_versions": build_result["empty_versions"],
            "failed_versions": build_result["failed_versions"],
            "normalized_trees": [str(path) for path in normalized_trees],
            "elements_dir": str(Path(pub_dir).resolve() / "elements"),
            "deleted_files": deleted_files,
        }

    def run_all(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run the full pipeline for each publication and return benchmark data."""
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"Source directory does not exist: {self.source_dir}"
            )
        pub_dirs = self.find_publications(self.source_dir)
        if limit is not None:
            pub_dirs = pub_dirs[:limit]

        self.benchmark.start()
        try:
            for index, pub_dir in enumerate(pub_dirs, 1):
                result = self.process_publication(pub_dir)
                logger.info(
                    "[%d/%d] %s: %d elements, %d BibTeX entries",
                    index,
                    len(pub_dirs),
                    result["arxiv_id"],
                    result["elements_count"],
                    result["bibtex_entries_count"],
                )
        finally:
            self.benchmark.stop()
        return self.benchmark.generate_summary()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    src_dir = Path(__file__).resolve().parents[1]
    default_source = src_dir / "sourcesArxiv"
    if not default_source.exists():
        default_source = src_dir.parent / "source"

    parser = argparse.ArgumentParser(
        description="Run the complete LaTeX gathering and parsing pipeline."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="Directory containing publication folders.",
    )
    parser.add_argument(
        "--clean-tex",
        dest="clean_tex",
        action="store_true",
        default=True,
        help="Delete redundant/orphan TeX files after processing (default).",
    )
    parser.add_argument(
        "--no-clean-tex",
        dest="clean_tex",
        action="store_false",
        default=True,
        help="Keep original TeX files after processing.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=src_dir / "benchmark_report.md",
        help="Path to the Markdown benchmark report.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=src_dir / "benchmark_stats.json",
        help="Path to the JSON benchmark statistics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of publications to process.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1

    benchmark = ParserBenchmark()
    pipeline = ParsingPipeline(
        source_dir=source_dir,
        clean_redundant_tex=args.clean_tex,
        benchmark=benchmark,
    )
    pub_dirs = ParsingPipeline.find_publications(source_dir)
    if args.limit is not None:
        pub_dirs = pub_dirs[: args.limit]

    print("=" * 60)
    print("LaTeX gathering and parsing pipeline")
    print(f"Source directory: {source_dir}")
    print(f"Publications: {len(pub_dirs)}")
    print(f"Redundant TeX cleanup: {'enabled' if args.clean_tex else 'disabled'}")
    print("=" * 60)

    benchmark.start()
    try:
        for index, pub_dir in enumerate(pub_dirs, 1):
            result = pipeline.process_publication(pub_dir)
            stats = result["dedup_stats"]
            print(
                f"[{index:02d}/{len(pub_dirs):02d}] {result['arxiv_id']} | "
                f"elements: {result['elements_count']} | "
                f"BibTeX: {result['bibtex_entries_count']} | "
                f"dedup: {stats['total_before']} -> {stats['total_after']}"
            )
    finally:
        benchmark.stop()

    summary = benchmark.generate_summary()
    report = benchmark.export_markdown_report(args.report_md)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Markdown report: {args.report_md}")
    print(f"Statistics JSON: {args.stats_json}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
