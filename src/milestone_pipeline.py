"""Run Milestones 1–3 and export their tables for a Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import arxiv_crawler
from . import crawler_common
from .model import run_matching
from .model.data_cleaning import build_references_table, flatten_references_json
from .parser import pipeline as parsing_pipeline

logger = logging.getLogger(__name__)

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


def _table_path(dataset_dir: Path, table_name: str) -> Path:
    return dataset_dir / "data" / table_name / f"{table_name}.parquet"


def _write_table(dataset_dir: Path, table_name: str, rows: Iterable[dict]) -> Path:
    path = _table_path(dataset_dir, table_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pylist(list(rows), schema=TABLE_SCHEMAS[table_name])
    pq.write_table(arrow_table, path)
    logger.info("Wrote %s rows to %s", arrow_table.num_rows, path)
    return path


def export_crawl_tables(source_dir: Path, dataset_dir: Path) -> None:
    """Export the papers and arXiv-keyed references immediately after crawling."""
    papers_by_id: Dict[str, List[dict]] = {}
    for metadata_path in sorted(source_dir.glob("*/v*/metadata.json")):
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, dict) or not metadata.get("arxiv_id"):
            raise ValueError(f"Missing arxiv_id in {metadata_path}")
        base_id = metadata.get("base_id") or metadata_path.parent.parent.name
        papers_by_id.setdefault(str(base_id), []).append(metadata)
    paper_rows = []
    for arxiv_id, versions in sorted(papers_by_id.items()):
        versions.sort(
            key=lambda row: _version_number(row.get("version", "v1"))
        )
        latest = versions[-1]
        first_version = versions[0]
        stored_revised_dates = _string_list(first_version.get("revised_dates"))
        revised_dates = []
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
                "arxiv_id": arxiv_id,
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
    _write_table(dataset_dir, "papers", paper_rows)

    references = flatten_references_json(source_dir)
    references_table = build_references_table(references)
    reference_rows = references_table.to_dict(orient="records")
    for row in reference_rows:
        row["ref_authors"] = _string_list(row.get("ref_authors"))
        for column in (
            "arxiv_id",
            "ref_arxiv_id",
            "ref_title",
            "ref_submission_date",
            "ref_s2_id",
        ):
            if pd.isna(row[column]):
                row[column] = None
            else:
                row[column] = str(row[column])
    _write_table(dataset_dir, "references", reference_rows)


def _load_publication_rows(source_dir: Path, relative_path: str) -> List[dict]:
    rows: List[dict] = []
    for publication_dir in sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"elements", "bibtex"}
    ):
        table_path = publication_dir / relative_path
        if not table_path.is_file():
            raise FileNotFoundError(
                f"Expected parser output {table_path}; did parsing complete?"
            )
        table_rows = _read_json(table_path)
        if not isinstance(table_rows, list):
            raise ValueError(f"Expected a JSON array in {table_path}")
        if not all(isinstance(row, dict) for row in table_rows):
            raise ValueError(f"Expected object rows in {table_path}")
        rows.extend(table_rows)
    return rows


def export_parsing_tables(source_dir: Path, dataset_dir: Path) -> None:
    """Export the deduplicated Milestone 2 tables after parsing."""
    element_rows = _load_publication_rows(source_dir, "elements/elements.json")
    _write_table(
        dataset_dir,
        "elements",
        [
            {
                "element_id": row.get("element_id"),
                "arxiv_id": row.get("arxiv_id"),
                "kind": row.get("kind"),
                "content": row.get("content"),
            }
            for row in element_rows
        ],
    )

    hierarchy_rows = _load_publication_rows(source_dir, "elements/hierarchy.json")
    _write_table(
        dataset_dir,
        "hierarchy",
        [
            {
                "arxiv_id": row.get("arxiv_id"),
                "version": _version_number(row.get("version")),
                "element_id": row.get("element_id"),
                "parent_id": row.get("parent_id"),
            }
            for row in hierarchy_rows
        ],
    )

    bibtex_rows = _load_publication_rows(source_dir, "bibtex/bibtex.json")
    normalized_bibtex = []
    for row in bibtex_rows:
        year = row.get("year")
        normalized_bibtex.append(
            {
                "arxiv_id": row.get("arxiv_id"),
                "citation_key": row.get("citation_key"),
                "entry_type": row.get("entry_type"),
                "title": row.get("title"),
                "authors": _string_list(row.get("authors")),
                "year": int(year) if year not in (None, "") else None,
                "doi": row.get("doi"),
                "journal": row.get("journal"),
                "url": row.get("url"),
            }
        )
    _write_table(dataset_dir, "bibtex", normalized_bibtex)


def export_predictions(matching_output_dir: Path, dataset_dir: Path) -> None:
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
    rows = predictions[required].where(pd.notna(predictions[required]), None).to_dict(
        orient="records"
    )
    for row in rows:
        for column in required:
            if row[column] is not None:
                row[column] = str(row[column])
    _write_table(dataset_dir, "predictions", rows)


def write_dataset_card(dataset_dir: Path) -> Path:
    configs = "\n".join(
        f"- config_name: {name}\n"
        f"  data_files:\n"
        f"  - split: train\n"
        f"    path: data/{name}/*.parquet"
        for name in TABLE_SCHEMAS
    )
    card = (
        "---\n"
        "configs:\n"
        f"{configs}\n"
        "---\n\n"
        "# NMKHDL Nhom 11 Milestone Dataset\n\n"
        "Processed arXiv papers, references, parsed structures, BibTeX entries, "
        "and reference-matching predictions. Each table is provided as Parquet "
        "under `data/<table>/`.\n"
    )
    path = dataset_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute crawl, parse, and matching in order, exporting each stage."""
    manual_labels = args.manual_labels.resolve()
    if not manual_labels.is_file():
        raise FileNotFoundError(f"Finalized manual labels not found: {manual_labels}")
    if args.year_month and (args.start is None or args.end is None):
        raise ValueError("--year-month requires both --start and --end")
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be provided together")
    if args.member and args.members_file is None:
        args.members_file = Path("members.json")
    if args.member and not args.members_file.is_file():
        raise FileNotFoundError(f"Members file not found: {args.members_file}")

    source_dir = args.source_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    work_dir = args.work_dir.resolve()
    matching_output_dir = work_dir / "matching"
    parser_report = work_dir / "parser_benchmark_report.md"
    parser_stats = work_dir / "parser_benchmark_stats.json"
    for directory in (source_dir, dataset_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ids = crawler_common.resolve_ids(args)
    if not ids:
        raise ValueError("The selected crawl range did not contain any arXiv IDs")

    logger.info("Milestone 1/3: crawling %d requested IDs", len(ids))
    crawl_summary = arxiv_crawler.crawl(ids, source_dir, with_refs=True)
    export_crawl_tables(source_dir, dataset_dir)

    logger.info("Milestone 2/3: parsing sources in %s", source_dir)
    parser_args = [
        "--source-dir",
        str(source_dir),
        "--report-md",
        str(parser_report),
        "--stats-json",
        str(parser_stats),
    ]
    if args.clean_tex:
        parser_args.append("--clean-tex")
    parser_result = parsing_pipeline.main(parser_args)
    if parser_result != 0:
        raise RuntimeError(f"Milestone 2 parsing exited with status {parser_result}")
    export_parsing_tables(source_dir, dataset_dir)

    logger.info("Milestone 3/3: matching references")
    matching_summary = run_matching.run(
        source_dir=source_dir,
        manual_labels_path=manual_labels,
        output_dir=matching_output_dir,
        num_members=args.num_members,
        members_file=args.members_file,
        auto_ratio=args.auto_ratio,
        seed=args.seed,
        run_ablation=not args.no_ablation,
    )
    export_predictions(matching_output_dir, dataset_dir)
    write_dataset_card(dataset_dir)

    return {
        "crawl": crawl_summary,
        "matching": matching_summary,
        "dataset_dir": str(dataset_dir),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Milestone 1 crawl, Milestone 2 parsing, and Milestone 3 matching."
    )
    crawl = parser.add_mutually_exclusive_group(required=True)
    crawl.add_argument("--ids", nargs="+", help="Explicit arXiv IDs to crawl")
    crawl.add_argument("--member", help="Member name in the members file")
    crawl.add_argument("--year-month", help="YYMM part of a numeric ID range")
    parser.add_argument("--start", type=int, help="First ID sequence number")
    parser.add_argument("--end", type=int, help="Last ID sequence number")
    parser.add_argument("--members-file", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("data/pipeline/sources"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/huggingface_dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/pipeline/work"))
    parser.add_argument("--manual-labels", type=Path, required=True)
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
    print(f"All milestones completed. Hugging Face dataset: {summary['dataset_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
