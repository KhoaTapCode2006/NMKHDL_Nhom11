"""Run Milestones 1–3 and export sharded Parquet tables for Hugging Face."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import arxiv_crawler
from . import crawler_common
from .model import run_matching
from .model.data_cleaning import (
    build_references_table,
    clean_year,
    flatten_references_json,
    normalize_arxiv_id,
)
from .parser import pipeline as parsing_pipeline

logger = logging.getLogger(__name__)

ALL_STAGES: Tuple[str, ...] = ("crawl", "parse", "match", "export")
DEFAULT_SHARD_ROWS: int = 200_000

TABLE_SCHEMAS: Dict[str, pa.Schema] = {
    "papers": pa.schema(
        [
            ("arxiv_id", pa.string(), False),
            ("title", pa.string()),
            ("authors", pa.list_(pa.string()), False),
            ("submission_date", pa.string()),
            ("revised_dates", pa.list_(pa.string()), False),
            ("venue", pa.string()),
        ]
    ),
    "references": pa.schema(
        [
            ("arxiv_id", pa.string(), False),
            ("ref_arxiv_id", pa.string(), False),
            ("ref_title", pa.string()),
            ("ref_authors", pa.list_(pa.string()), False),
            ("ref_submission_date", pa.string()),
            ("ref_s2_id", pa.string()),
        ]
    ),
    "elements": pa.schema(
        [
            ("element_id", pa.string(), False),
            ("arxiv_id", pa.string(), False),
            ("kind", pa.string(), False),
            ("content", pa.string()),
        ]
    ),
    "hierarchy": pa.schema(
        [
            ("arxiv_id", pa.string(), False),
            ("version", pa.int64(), False),
            ("element_id", pa.string(), False),
            ("parent_id", pa.string(), False),
        ]
    ),
    "bibtex": pa.schema(
        [
            ("arxiv_id", pa.string(), False),
            ("citation_key", pa.string(), False),
            ("entry_type", pa.string(), False),
            ("title", pa.string()),
            ("authors", pa.list_(pa.string()), False),
            ("year", pa.int64()),
            ("doi", pa.string()),
            ("journal", pa.string()),
            ("url", pa.string()),
        ]
    ),
    "predictions": pa.schema(
        [
            ("arxiv_id", pa.string(), False),
            ("partition", pa.string(), False),
            ("citation_key", pa.string(), False),
            ("groundtruth_arxiv_id", pa.string()),
            ("cand_1", pa.string()),
            ("cand_2", pa.string()),
            ("cand_3", pa.string()),
            ("cand_4", pa.string()),
            ("cand_5", pa.string()),
        ]
    ),
}


SourceDirsInput = Union[Path, Sequence[Path]]


def _normalize_source_dirs(source_dirs: SourceDirsInput) -> List[Path]:
    if isinstance(source_dirs, (str, Path)):
        raw = [Path(source_dirs)]
    else:
        raw = [Path(item) for item in source_dirs]
    resolved: List[Path] = []
    seen: Set[Path] = set()
    for path in raw:
        real = path.resolve()
        if real not in seen:
            seen.add(real)
            resolved.append(real)
    return resolved


def _canonical_pub_id(raw_id: Any) -> str:
    text = str(raw_id or "").strip()
    return normalize_arxiv_id(text) or text


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read valid JSON from {path}: {error}") from error


def _version_number(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"v?(\d+)", str(value).strip())
    if not match:
        raise ValueError(f"Invalid publication version: {value!r}")
    return int(match.group(1))


def _string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if value is not None]


def _shard_path(dataset_dir: Path, table_name: str, shard_index: int) -> Path:
    return dataset_dir / "data" / table_name / f"part-{shard_index:05d}.parquet"


def _prepare_table_dir(dataset_dir: Path, table_name: str) -> Path:
    table_dir = dataset_dir / "data" / table_name
    table_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in table_dir.glob("*.parquet"):
        stale_file.unlink()
    return table_dir


def _write_sharded_table(
    dataset_dir: Path,
    table_name: str,
    rows: Iterable[dict],
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> List[Path]:
    """Stream rows into bounded-size Parquet shards to keep memory flat."""
    if shard_rows <= 0:
        raise ValueError(f"shard_rows must be positive, got {shard_rows}")
    schema = TABLE_SCHEMAS[table_name]
    _prepare_table_dir(dataset_dir, table_name)

    written_paths: List[Path] = []
    total_rows = 0
    batch: List[dict] = []

    def _flush(shard_idx: int, records: List[dict]) -> None:
        nonlocal total_rows
        out_path = _shard_path(dataset_dir, table_name, shard_idx)
        arrow_table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(arrow_table, out_path, compression="snappy")
        written_paths.append(out_path)
        total_rows += arrow_table.num_rows

    for row in rows:
        batch.append(row)
        if len(batch) >= shard_rows:
            _flush(len(written_paths), batch)
            batch = []

    if batch or not written_paths:
        _flush(len(written_paths), batch)

    logger.info(
        "Wrote %d row(s) across %d shard(s) for table '%s'",
        total_rows,
        len(written_paths),
        table_name,
    )
    return written_paths


def _write_table(
    dataset_dir: Path,
    table_name: str,
    rows: Iterable[dict],
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> Path:
    """Compatibility helper returning the first shard path."""
    return _write_sharded_table(dataset_dir, table_name, rows, shard_rows=shard_rows)[0]


def _discover_unique_publications(
    source_dirs: SourceDirsInput,
    allowed_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Path]]:
    """Collect unique ``(arxiv_id, pub_dir)`` sorted by canonical arXiv ID."""
    by_id: Dict[str, Path] = {}
    for source_dir in _normalize_source_dirs(source_dirs):
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.iterdir()):
            if (
                not path.is_dir()
                or path.name.startswith(".")
                or path.name in {"elements", "bibtex"}
            ):
                continue
            pub_id = _canonical_pub_id(path.name)
            if allowed_ids is not None and pub_id not in allowed_ids:
                continue
            by_id.setdefault(pub_id, path)
    return sorted(by_id.items(), key=lambda item: item[0])


def export_crawl_tables(
    source_dirs: SourceDirsInput,
    dataset_dir: Path,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    allowed_ids: Optional[Set[str]] = None,
) -> None:
    """Export the ``papers`` and arXiv-keyed ``references`` tables."""
    publications = _discover_unique_publications(source_dirs, allowed_ids=allowed_ids)
    paper_rows: List[dict] = []

    for arxiv_id, pub_dir in publications:
        versions: List[dict] = []
        for metadata_path in sorted(pub_dir.glob("v*/metadata.json")):
            metadata = _read_json(metadata_path)
            if not isinstance(metadata, dict) or not metadata.get("arxiv_id"):
                raise ValueError(f"Missing arxiv_id in {metadata_path}")
            versions.append(metadata)
        if not versions:
            continue

        versions.sort(key=lambda row: _version_number(row.get("version", "v1")))
        latest = versions[-1]
        first_version = versions[0]
        canonical_id = _canonical_pub_id(first_version.get("base_id") or arxiv_id)
        stored_revised_dates = _string_list(first_version.get("revised_dates"))
        revised_dates: List[str] = []
        for index, row in enumerate(versions[1:]):
            date = row.get("version_date") or row.get("updated_date")
            if not date and index < len(stored_revised_dates):
                date = stored_revised_dates[index]
            if date:
                revised_dates.append(str(date))
        if not versions[1:]:
            revised_dates = stored_revised_dates

        paper_rows.append(
            {
                "arxiv_id": canonical_id,
                "title": latest.get("title") or first_version.get("title"),
                "authors": _string_list(
                    latest.get("authors") or first_version.get("authors")
                ),
                "submission_date": next(
                    (
                        row.get("submission_date")
                        for row in versions
                        if row.get("submission_date")
                    ),
                    None,
                ),
                "revised_dates": revised_dates,
                "venue": latest.get("venue")
                or latest.get("journal_ref")
                or first_version.get("venue")
                or first_version.get("journal_ref"),
            }
        )
    _write_sharded_table(dataset_dir, "papers", paper_rows, shard_rows=shard_rows)

    raw_frames = [
        flatten_references_json(src_dir)
        for src_dir in _normalize_source_dirs(source_dirs)
        if src_dir.is_dir()
    ]
    non_empty_frames = [frame for frame in raw_frames if not frame.empty]
    merged_raw_refs = (
        pd.concat(non_empty_frames, ignore_index=True)
        if non_empty_frames
        else pd.DataFrame()
    )
    references_table = (
        build_references_table(merged_raw_refs)
        if not merged_raw_refs.empty
        else pd.DataFrame(columns=TABLE_SCHEMAS["references"].names)
    )
    if allowed_ids is not None and not references_table.empty:
        references_table = references_table[
            references_table["arxiv_id"].map(_canonical_pub_id).isin(allowed_ids)
        ].reset_index(drop=True)

    def _iter_reference_rows() -> Iterator[dict]:
        for row in references_table.to_dict(orient="records"):
            row["arxiv_id"] = _canonical_pub_id(row.get("arxiv_id"))
            row["ref_arxiv_id"] = _canonical_pub_id(row.get("ref_arxiv_id"))
            row["ref_authors"] = _string_list(row.get("ref_authors"))
            for column in (
                "arxiv_id",
                "ref_arxiv_id",
                "ref_title",
                "ref_submission_date",
                "ref_s2_id",
            ):
                val = row.get(column)
                row[column] = None if pd.isna(val) else str(val)
            yield row

    _write_sharded_table(
        dataset_dir, "references", _iter_reference_rows(), shard_rows=shard_rows
    )


def _read_publication_table_rows(pub_dir: Path, relative_path: str) -> List[dict]:
    table_path = pub_dir / relative_path
    if not table_path.is_file():
        manifest_path = pub_dir / "elements" / "compilation_manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if isinstance(manifest, dict) and all(
                isinstance(info, dict) and info.get("status") in {"empty_version", "failed"}
                for info in manifest.values()
            ):
                return []
        raise FileNotFoundError(
            f"Expected parser output {table_path}; did Milestone 2 parsing complete?"
        )
    table_rows = _read_json(table_path)
    if not isinstance(table_rows, list) or not all(
        isinstance(row, dict) for row in table_rows
    ):
        raise ValueError(f"Expected a JSON array of object rows in {table_path}")
    return table_rows


def export_parsing_tables(
    source_dirs: SourceDirsInput,
    dataset_dir: Path,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    allowed_ids: Optional[Set[str]] = None,
) -> None:
    """Stream and export the deduplicated Milestone 2 tables as sharded Parquet."""
    publications = _discover_unique_publications(source_dirs, allowed_ids=allowed_ids)

    def _iter_elements() -> Iterator[dict]:
        for pub_id, pub_dir in publications:
            for row in _read_publication_table_rows(pub_dir, "elements/elements.json"):
                element_id = str(row.get("element_id") or "").strip()
                if not element_id:
                    raise ValueError(f"Empty element_id in {pub_dir / 'elements/elements.json'}")
                yield {
                    "element_id": element_id,
                    "arxiv_id": _canonical_pub_id(row.get("arxiv_id") or pub_id),
                    "kind": str(row.get("kind") or "unknown").strip() or "unknown",
                    "content": None if row.get("content") is None else str(row.get("content")),
                }

    def _iter_hierarchy() -> Iterator[dict]:
        for pub_id, pub_dir in publications:
            for row in _read_publication_table_rows(pub_dir, "elements/hierarchy.json"):
                element_id = str(row.get("element_id") or "").strip()
                parent_id = str(row.get("parent_id") or "").strip()
                if not element_id or not parent_id:
                    raise ValueError(
                        f"Non-null element_id and parent_id required in {pub_dir / 'elements/hierarchy.json'}"
                    )
                yield {
                    "arxiv_id": _canonical_pub_id(row.get("arxiv_id") or pub_id),
                    "version": _version_number(row.get("version")),
                    "element_id": element_id,
                    "parent_id": parent_id,
                }

    def _iter_bibtex() -> Iterator[dict]:
        for pub_id, pub_dir in publications:
            for row in _read_publication_table_rows(pub_dir, "bibtex/bibtex.json"):
                citation_key = str(row.get("citation_key") or "").strip()
                if not citation_key:
                    continue
                entry_type = str(row.get("entry_type") or "misc").strip() or "misc"
                yield {
                    "arxiv_id": _canonical_pub_id(row.get("arxiv_id") or pub_id),
                    "citation_key": citation_key,
                    "entry_type": entry_type,
                    "title": None if row.get("title") is None else str(row.get("title")),
                    "authors": _string_list(row.get("authors")),
                    "year": clean_year(row.get("year")),
                    "doi": None if row.get("doi") is None else str(row.get("doi")),
                    "journal": None if row.get("journal") is None else str(row.get("journal")),
                    "url": None if row.get("url") is None else str(row.get("url")),
                }

    _write_sharded_table(dataset_dir, "elements", _iter_elements(), shard_rows=shard_rows)
    _write_sharded_table(dataset_dir, "hierarchy", _iter_hierarchy(), shard_rows=shard_rows)
    _write_sharded_table(dataset_dir, "bibtex", _iter_bibtex(), shard_rows=shard_rows)


def export_predictions(
    matching_output_dir: Path,
    dataset_dir: Path,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    allowed_ids: Optional[Set[str]] = None,
) -> None:
    predictions_path = matching_output_dir / "predictions.parquet"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Matching did not create {predictions_path}")
    predictions = pd.read_parquet(predictions_path)
    required = TABLE_SCHEMAS["predictions"].names
    missing_columns = [column for column in required if column not in predictions]
    if missing_columns:
        raise ValueError(
            f"Matching predictions are missing columns: {', '.join(missing_columns)}"
        )
    filtered_rows: List[dict] = []
    for row in predictions[required].to_dict(orient="records"):
        for column in required:
            val = row.get(column)
            row[column] = None if (val is None or pd.isna(val)) else str(val)
        row["arxiv_id"] = _canonical_pub_id(row["arxiv_id"])
        if allowed_ids is not None and row["arxiv_id"] not in allowed_ids:
            continue
        filtered_rows.append(row)
    _write_sharded_table(
        dataset_dir, "predictions", filtered_rows, shard_rows=shard_rows
    )


def write_dataset_card(
    dataset_dir: Path,
    table_names: Optional[Sequence[str]] = None,
) -> Path:
    if table_names is None:
        present = [
            name
            for name in TABLE_SCHEMAS
            if list((dataset_dir / "data" / name).glob("*.parquet"))
        ]
        table_names = present or list(TABLE_SCHEMAS)

    configs = "\n".join(
        f"- config_name: {name}\n"
        f"  data_files:\n"
        f"  - split: train\n"
        f"    path: data/{name}/*.parquet"
        for name in table_names
    )
    card = (
        "---\n"
        "configs:\n"
        f"{configs}\n"
        "---\n\n"
        "# NMKHDL Nhom 11 Milestone Dataset\n\n"
        "Processed arXiv papers, references, parsed hierarchical elements, "
        "unified BibTeX entries, and reference-matching predictions exported "
        "as sharded Parquet tables under `data/<table>/`.\n"
    )
    path = dataset_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    return path


def _resolve_active_stages(args: argparse.Namespace) -> Set[str]:
    if getattr(args, "export_only", False):
        return {"export"}
    if getattr(args, "stages", None):
        return set(args.stages)
    active = set(ALL_STAGES)
    if getattr(args, "skip_crawl", False):
        active.discard("crawl")
    if getattr(args, "skip_parse", False):
        active.discard("parse")
    if getattr(args, "skip_match", False):
        active.discard("match")
    return active


def _has_crawl_selection(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "ids", None)
        or getattr(args, "member", None)
        or getattr(args, "year_month", None)
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the requested stages and export sharded Hugging Face Parquet tables."""
    stages = _resolve_active_stages(args)
    if not stages:
        raise ValueError("No pipeline stages selected to execute.")

    if args.year_month and (args.start is None or args.end is None):
        raise ValueError("--year-month requires both --start and --end")
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be provided together")
    if args.member and args.members_file is None:
        args.members_file = Path("members.json")
    if args.member and not args.members_file.is_file():
        raise FileNotFoundError(f"Members file not found: {args.members_file}")

    source_dirs = _normalize_source_dirs(args.source_dir)
    primary_source_dir = source_dirs[0]
    dataset_dir = args.dataset_dir.resolve()
    work_dir = args.work_dir.resolve()
    matching_output_dir = (
        args.matching_output_dir.resolve()
        if getattr(args, "matching_output_dir", None)
        else work_dir / "matching"
    )
    shard_rows = getattr(args, "shard_rows", DEFAULT_SHARD_ROWS)

    for directory in (primary_source_dir, dataset_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected_ids: Optional[List[str]] = None
    if _has_crawl_selection(args):
        selected_ids = crawler_common.resolve_ids(args)
        if not selected_ids:
            raise ValueError("The selected crawl range did not contain any arXiv IDs")
    elif "crawl" in stages:
        raise ValueError(
            "Stage 'crawl' requires --ids, --member, or --year-month with --start/--end"
        )

    manual_labels: Optional[Path] = None
    if "match" in stages:
        if getattr(args, "manual_labels", None) is None:
            raise ValueError("Stage 'match' requires --manual-labels")
        manual_labels = args.manual_labels.resolve()
        if not manual_labels.is_file():
            raise FileNotFoundError(f"Finalized manual labels not found: {manual_labels}")

    allowed_ids = (
        {_canonical_pub_id(pid) for pid in selected_ids}
        if (selected_ids is not None and getattr(args, "filter_selected_ids", False))
        else None
    )

    exported_crawl = False
    exported_parse = False
    exported_match = False

    crawl_summary: Optional[Dict[str, Any]] = None
    if "crawl" in stages:
        assert selected_ids is not None
        logger.info("Milestone 1/3: crawling %d requested IDs", len(selected_ids))
        crawl_summary = arxiv_crawler.crawl(
            selected_ids, primary_source_dir, with_refs=True
        )
        if "export" in stages:
            export_crawl_tables(
                source_dirs,
                dataset_dir,
                shard_rows=shard_rows,
                allowed_ids=allowed_ids,
            )
            exported_crawl = True

    if "parse" in stages:
        for index, src_dir in enumerate(source_dirs, 1):
            if not src_dir.is_dir():
                raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
            suffix = f"_{index}" if len(source_dirs) > 1 else ""
            parser_report = work_dir / f"parser_benchmark_report{suffix}.md"
            parser_stats = work_dir / f"parser_benchmark_stats{suffix}.json"
            logger.info("Milestone 2/3: parsing sources in %s", src_dir)
            parser_args = [
                "--source-dir",
                str(src_dir),
                "--report-md",
                str(parser_report),
                "--stats-json",
                str(parser_stats),
            ]
            if args.clean_tex:
                parser_args.append("--clean-tex")
            parser_result = parsing_pipeline.main(parser_args)
            if parser_result != 0:
                raise RuntimeError(
                    f"Milestone 2 parsing exited with status {parser_result} for {src_dir}"
                )
        if "export" in stages:
            export_parsing_tables(
                source_dirs,
                dataset_dir,
                shard_rows=shard_rows,
                allowed_ids=allowed_ids,
            )
            exported_parse = True

    matching_summary: Optional[Dict[str, Any]] = None
    if "match" in stages:
        assert manual_labels is not None
        logger.info("Milestone 3/3: matching references")
        matching_summary = run_matching.run(
            source_dir=source_dirs if len(source_dirs) > 1 else primary_source_dir,
            manual_labels_path=manual_labels,
            output_dir=matching_output_dir,
            num_members=args.num_members,
            members_file=args.members_file,
            auto_ratio=args.auto_ratio,
            seed=args.seed,
            run_ablation=not args.no_ablation,
        )
        if "export" in stages:
            export_predictions(
                matching_output_dir,
                dataset_dir,
                shard_rows=shard_rows,
                allowed_ids=allowed_ids,
            )
            exported_match = True

    if "export" in stages:
        if not exported_crawl:
            has_crawl_meta = any(
                any(src_dir.glob("*/v*/metadata.json"))
                for src_dir in source_dirs
                if src_dir.is_dir()
            )
            if has_crawl_meta:
                export_crawl_tables(
                    source_dirs,
                    dataset_dir,
                    shard_rows=shard_rows,
                    allowed_ids=allowed_ids,
                )
        if not exported_parse:
            has_parsed_outputs = any(
                any(src_dir.glob("*/elements/elements.json"))
                or any(src_dir.glob("*/elements/compilation_manifest.json"))
                for src_dir in source_dirs
                if src_dir.is_dir()
            )
            if has_parsed_outputs:
                export_parsing_tables(
                    source_dirs,
                    dataset_dir,
                    shard_rows=shard_rows,
                    allowed_ids=allowed_ids,
                )
        if not exported_match:
            predictions_file = matching_output_dir / "predictions.parquet"
            if predictions_file.is_file():
                export_predictions(
                    matching_output_dir,
                    dataset_dir,
                    shard_rows=shard_rows,
                    allowed_ids=allowed_ids,
                )
        write_dataset_card(dataset_dir)

    return {
        "stages": sorted(stages),
        "crawl": crawl_summary,
        "matching": matching_summary,
        "dataset_dir": str(dataset_dir),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Milestone 1 crawl, Milestone 2 parsing, Milestone 3 matching, "
            "and export sharded Hugging Face Parquet tables."
        )
    )
    crawl = parser.add_mutually_exclusive_group(required=False)
    crawl.add_argument("--ids", nargs="+", help="Explicit arXiv IDs to crawl")
    crawl.add_argument("--member", help="Member name in the members file")
    crawl.add_argument("--year-month", help="YYMM part of a numeric ID range")
    parser.add_argument("--start", type=int, help="First ID sequence number")
    parser.add_argument("--end", type=int, help="Last ID sequence number")
    parser.add_argument("--members-file", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        nargs="+",
        default=[Path("data/pipeline/sources")],
        help="One or more directories containing crawled/parsed publications.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/huggingface_dataset"),
        help="Output directory for the Hugging Face Parquet dataset and README card.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/pipeline/work"))
    parser.add_argument(
        "--matching-output-dir",
        type=Path,
        default=None,
        help="Optional directory with existing Milestone 3 outputs (predictions.parquet).",
    )
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=None,
        help="Path to finalized manual_labels.csv (required when running 'match').",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Explicit subset of stages to run (default: crawl parse match export).",
    )
    parser.add_argument("--skip-crawl", action="store_true", help="Skip Milestone 1 crawling")
    parser.add_argument("--skip-parse", action="store_true", help="Skip Milestone 2 parsing")
    parser.add_argument("--skip-match", action="store_true", help="Skip Milestone 3 matching")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export Parquet tables and dataset card from existing outputs",
    )
    parser.add_argument(
        "--filter-selected-ids",
        action="store_true",
        help="Filter exported tables strictly to the IDs resolved by --member/--ids/--year-month",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=DEFAULT_SHARD_ROWS,
        help=f"Maximum rows per Parquet shard (default: {DEFAULT_SHARD_ROWS}).",
    )
    parser.add_argument("--num-members", type=int, default=4)
    parser.add_argument("--auto-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--clean-tex", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argument_parser().parse_args(argv)
    try:
        summary = run_pipeline(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Pipeline failed: %s", error)
        return 1
    print(
        f"Completed stages ({', '.join(summary['stages'])}). "
        f"Hugging Face dataset: {summary['dataset_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
