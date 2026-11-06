"""
Full Parsing Pipeline for LaTeX Publications.
Integrates MultiFileGatherer, HierarchyExtractor, ElementDeduplicator,
file cleanup, and ParserBenchmark.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .gatherer import MultiFileGatherer
from .extractor import HierarchyExtractor
from .dedup import ElementDeduplicator
from .benchmark import ParserBenchmark


class ParsingPipeline:
    """Orchestrates end-to-end multi-file gathering, parsing, deduplication, and cleanup."""

    def __init__(
        self,
        source_dir: str | Path,
        clean_redundant_tex: bool = True,
        benchmark: Optional[ParserBenchmark] = None,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.clean_redundant_tex = clean_redundant_tex
        self.benchmark = benchmark or ParserBenchmark()

    def process_version(
        self,
        arxiv_id: str,
        version_name: str,
        version_dir: Path,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single version directory:
        1. Gathers files through root and \\input/\\include
        2. Extracts hierarchy and elements
        3. Returns raw version parsing result
        """
        start_t = time.perf_counter()

        # Step 1: Gathering
        gather_res = MultiFileGatherer.gather(version_dir, metadata=metadata)
        gather_time = time.perf_counter() - start_t

        if not gather_res["success"]:
            status = "empty_version" if "No .tex files found" in (gather_res["error"] or "") else "failed"
            self.benchmark.record_version(
                arxiv_id=arxiv_id,
                version=version_name,
                status=status,
                duration_sec=gather_time,
                error_msg=gather_res["error"],
            )
            return {
                "success": False,
                "error": gather_res["error"],
                "gather_res": gather_res,
                "elements": [],
                "hierarchy": [],
            }

        # Step 2: Extraction
        extract_start = time.perf_counter()
        elements, hierarchy = HierarchyExtractor.parse_version(
            arxiv_id=arxiv_id,
            version=version_name,
            inlined_tex=gather_res["inlined_content"],
        )
        total_version_time = time.perf_counter() - start_t

        self.benchmark.record_version(
            arxiv_id=arxiv_id,
            version=version_name,
            status="success",
            duration_sec=total_version_time,
            compilation_files_count=len(gather_res["compilation_files"]),
            elements_count=len(elements),
        )

        return {
            "success": True,
            "error": None,
            "gather_res": gather_res,
            "elements": elements,
            "hierarchy": hierarchy,
        }

    def process_publication(
        self,
        pub_dir: Path,
    ) -> Dict[str, Any]:
        """
        Processes all versions of a publication:
        1. Multi-file gathers & parses each version
        2. Deduplicates elements across versions
        3. Stores results into <pub_dir>/elements/
        4. Cleans redundant/orphan .tex files if enabled
        """
        pub_start_t = time.perf_counter()
        pub_dir = Path(pub_dir).resolve()
        arxiv_id = pub_dir.name

        # Identify all version directories (v1, v2, ...)
        version_dirs = sorted([
            d for d in pub_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ], key=lambda x: int(x.name[1:]))

        # Check metadata
        meta = None
        for vd in version_dirs:
            meta_p = vd / "metadata.json"
            if meta_p.is_file():
                try:
                    with open(meta_p, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    break
                except Exception:
                    pass

        raw_version_data: Dict[str, Any] = {}
        successful_gather_results: Dict[str, Any] = {}
        tex_versions_count = 0
        empty_versions_count = 0

        for vd in version_dirs:
            v_name = vd.name
            v_meta_p = vd / "metadata.json"
            v_meta = meta
            if v_meta_p.is_file():
                try:
                    with open(v_meta_p, "r", encoding="utf-8") as f:
                        v_meta = json.load(f)
                except Exception:
                    pass

            v_res = self.process_version(arxiv_id, v_name, vd, metadata=v_meta)
            if v_res["success"]:
                tex_versions_count += 1
                raw_version_data[v_name] = (v_res["elements"], v_res["hierarchy"])
                successful_gather_results[v_name] = v_res["gather_res"]
            else:
                if "empty" in (v_res["error"] or "").lower():
                    empty_versions_count += 1

        # Step 3: Deduplication across versions
        dedup_res = ElementDeduplicator.deduplicate_publication(
            arxiv_id=arxiv_id,
            version_parsed_data=raw_version_data,
        )

        # Step 4: Save to <pub_dir>/elements/
        elements_dir = pub_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        # 4.1 Save JSON format
        elements_path = elements_dir / "elements.json"
        with open(elements_path, "w", encoding="utf-8") as f:
            json.dump(dedup_res["elements"], f, ensure_ascii=False, indent=2)

        hierarchy_path = elements_dir / "hierarchy.json"
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            json.dump(dedup_res["hierarchy"], f, ensure_ascii=False, indent=2)

        # 4.2 Save CSV format
        try:
            import pandas as pd
            df_elements = pd.DataFrame(dedup_res["elements"])
            df_elements.to_csv(elements_dir / "elements.csv", index=False, encoding="utf-8")

            df_hierarchy = pd.DataFrame(dedup_res["hierarchy"])
            df_hierarchy.to_csv(elements_dir / "hierarchy.csv", index=False, encoding="utf-8")

            # 4.3 Save Parquet format (if pyarrow or fastparquet is available)
            try:
                df_elements.to_parquet(elements_dir / "elements.parquet", index=False)
                df_hierarchy.to_parquet(elements_dir / "hierarchy.parquet", index=False)
            except Exception:
                pass
        except Exception:
            pass

        # Save compilation paths manifest and inlined full text
        compilation_manifest = {}
        for v_name, g_res in successful_gather_results.items():
            compilation_manifest[v_name] = {
                "root_file": str(g_res["root_file"].name) if g_res["root_file"] else None,
                "compilation_files": [f.name for f in g_res["compilation_files"]],
                "orphan_files": [f.name for f in g_res["orphan_files"]],
            }
            # Save assembled inlined full tex
            inlined_tex_path = elements_dir / f"full_inlined_{v_name}.tex"
            inlined_tex_path.write_text(g_res["inlined_content"], encoding="utf-8")

        manifest_path = elements_dir / "compilation_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(compilation_manifest, f, ensure_ascii=False, indent=2)

        # Step 5: Clean redundant .tex files if requested
        deleted_files_log = {}
        if self.clean_redundant_tex:
            for vd in version_dirs:
                v_name = vd.name
                if v_name in successful_gather_results:
                    g_res = successful_gather_results[v_name]
                    deleted = MultiFileGatherer.clean_redundant_files(
                        version_dir=vd,
                        compilation_files=g_res["compilation_files"],
                        orphan_files=g_res["orphan_files"],
                        keep_root=False,  # Already preserved in elements/full_inlined_vX.tex
                    )
                    deleted_files_log[v_name] = [f.name for f in deleted]

        pub_total_time = time.perf_counter() - pub_start_t

        expected_versions = len(version_dirs)
        self.benchmark.record_publication(
            arxiv_id=arxiv_id,
            expected_versions=expected_versions,
            crawled_versions=len(version_dirs),
            tex_versions=tex_versions_count,
            empty_versions=empty_versions_count,
            dedup_stats=dedup_res["stats"],
            duration_sec=pub_total_time,
        )

        return {
            "arxiv_id": arxiv_id,
            "elements_count": len(dedup_res["elements"]),
            "hierarchy_count": len(dedup_res["hierarchy"]),
            "dedup_stats": dedup_res["stats"],
            "elements_dir": str(elements_dir),
            "deleted_files": deleted_files_log,
        }

    def run_all(self) -> Dict[str, Any]:
        """Runs the pipeline over all publications in source_dir."""
        pub_dirs = sorted([
            d for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "elements"
        ])

        self.benchmark.start()
        results = []

        print(f"🚀 Starting Parser Pipeline on {len(pub_dirs)} publications in {self.source_dir}...")
        for idx, p_dir in enumerate(pub_dirs, 1):
            res = self.process_publication(p_dir)
            results.append(res)
            print(
                f"[{idx}/{len(pub_dirs)}] Processed {res['arxiv_id']}: "
                f"{res['elements_count']} unique elements, "
                f"dedup reduction {res['dedup_stats']['reduction_percentage']}%"
            )

        self.benchmark.stop()
        summary = self.benchmark.generate_summary()
        return summary
