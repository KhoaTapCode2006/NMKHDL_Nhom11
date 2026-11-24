import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import milestone_pipeline


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_parser_outputs(source_dir: Path) -> None:
    publication_dir = source_dir / "2407.17839"
    _write_json(
        publication_dir / "elements" / "elements.json",
        [
            {
                "element_id": "2407.17839#s1",
                "arxiv_id": "2407.17839",
                "kind": "sentence",
                "content": "A sentence.",
            }
        ],
    )
    _write_json(
        publication_dir / "elements" / "hierarchy.json",
        [
            {
                "arxiv_id": "2407.17839",
                "version": "v1",
                "element_id": "2407.17839#s1",
                "parent_id": "2407.17839#root1",
            }
        ],
    )
    _write_json(
        publication_dir / "bibtex" / "bibtex.json",
        [
            {
                "arxiv_id": "2407.17839",
                "citation_key": "ref1",
                "entry_type": "article",
                "title": "A paper",
                "authors": ["A Author"],
                "year": 2024,
                "doi": None,
                "journal": None,
                "url": None,
            }
        ],
    )


def _write_crawl_outputs(source_dir: Path) -> None:
    version_dir = source_dir / "2407.17839" / "v1"
    _write_json(
        version_dir / "metadata.json",
        {
            "arxiv_id": "2407.17839v1",
            "base_id": "2407.17839",
            "version": "1",
            "title": "A paper",
            "authors": ["A Author"],
            "submission_date": "2024-07-01",
            "version_date": "2024-07-01",
            "revised_dates": [],
            "venue": None,
            "doi": None,
            "primary_category": "cs.AI",
            "categories": ["cs.AI"],
        },
    )
    _write_json(
        source_dir / "2407.17839" / "v2" / "metadata.json",
        {
            "arxiv_id": "2407.17839v2",
            "base_id": "2407.17839",
            "version": "2",
            "title": "A revised paper",
            "authors": ["A Author", "C Author"],
            "submission_date": "2024-07-01",
            "version_date": "2024-07-14",
            "revised_dates": ["2024-07-14"],
            "venue": "Updated Journal",
        },
    )
    _write_json(
        version_dir / "references.json",
        [
            {
                "paperId": "s2-paper",
                "externalIds": {"ArXiv": "2301.12345"},
                "title": "Referenced paper",
                "authors": [{"name": "B Author"}],
                "year": 2023,
                "publicationDate": "2023-01-02",
                "venue": "Journal",
            },
            {
                "paperId": "not-arxiv",
                "externalIds": {},
                "title": "A non-arXiv reference",
                "authors": [],
            },
        ],
    )


def test_export_tables_preserves_required_columns_and_types(tmp_path):
    source_dir = tmp_path / "sources"
    dataset_dir = tmp_path / "dataset"
    _write_crawl_outputs(source_dir)

    milestone_pipeline.export_crawl_tables(source_dir, dataset_dir)
    _write_parser_outputs(source_dir)
    milestone_pipeline.export_parsing_tables(source_dir, dataset_dir)

    papers = pq.read_table(dataset_dir / "data" / "papers" / "papers.parquet")
    refs = pq.read_table(dataset_dir / "data" / "references" / "references.parquet")
    elements = pq.read_table(dataset_dir / "data" / "elements" / "elements.parquet")
    hierarchy = pq.read_table(dataset_dir / "data" / "hierarchy" / "hierarchy.parquet")
    bibtex = pq.read_table(dataset_dir / "data" / "bibtex" / "bibtex.parquet")

    assert papers.schema.equals(milestone_pipeline.TABLE_SCHEMAS["papers"])
    assert papers.num_rows == 1
    assert papers.column("arxiv_id").to_pylist() == ["2407.17839"]
    assert papers.column("revised_dates").to_pylist() == [["2024-07-14"]]
    assert papers.column("title").to_pylist() == ["A revised paper"]
    assert refs.schema.equals(milestone_pipeline.TABLE_SCHEMAS["references"])
    assert refs.column("ref_arxiv_id").to_pylist() == ["2301.12345"]
    assert elements.schema.equals(milestone_pipeline.TABLE_SCHEMAS["elements"])
    assert hierarchy.column("version").to_pylist() == [1]
    assert bibtex.schema.equals(milestone_pipeline.TABLE_SCHEMAS["bibtex"])


def test_pipeline_runs_stages_in_order_and_writes_dataset_card(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    dataset_dir = tmp_path / "dataset"
    manual_labels = tmp_path / "manual_labels.csv"
    manual_labels.write_text("arxiv_id,citation_key,groundtruth_arxiv_id\n", encoding="utf-8")
    args = milestone_pipeline.build_argument_parser().parse_args(
        [
            "--ids",
            "2407.17839",
            "--manual-labels",
            str(manual_labels),
            "--source-dir",
            str(source_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    stages = []

    def fake_crawl(ids, out_dir, with_refs):
        stages.append("crawl")
        _write_crawl_outputs(out_dir)
        return {"requested": len(ids)}

    def fake_parse(argv):
        stages.append("parse")
        assert (dataset_dir / "data" / "papers" / "papers.parquet").is_file()
        assert (dataset_dir / "data" / "references" / "references.parquet").is_file()
        _write_parser_outputs(source_dir)
        return 0

    def fake_matching(**kwargs):
        stages.append("matching")
        assert all(
            (dataset_dir / "data" / table / f"{table}.parquet").is_file()
            for table in ("elements", "hierarchy", "bibtex")
        )
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "arxiv_id": "2407.17839",
                    "partition": "test",
                    "citation_key": "ref1",
                    "groundtruth_arxiv_id": None,
                    "cand_1": "2301.12345",
                    "cand_2": None,
                    "cand_3": None,
                    "cand_4": None,
                    "cand_5": None,
                }
            ]
        ).to_parquet(output_dir / "predictions.parquet", index=False)
        return {"selected_model": "baseline"}

    monkeypatch.setattr(milestone_pipeline.arxiv_crawler, "crawl", fake_crawl)
    monkeypatch.setattr(milestone_pipeline.parsing_pipeline, "main", fake_parse)
    monkeypatch.setattr(milestone_pipeline.run_matching, "run", fake_matching)

    summary = milestone_pipeline.run_pipeline(args)

    assert stages == ["crawl", "parse", "matching"]
    assert summary["dataset_dir"] == str(dataset_dir.resolve())
    card = (dataset_dir / "README.md").read_text(encoding="utf-8")
    for table in milestone_pipeline.TABLE_SCHEMAS:
        assert f"config_name: {table}" in card
        assert f"path: data/{table}/*.parquet" in card
    predictions = pq.read_table(dataset_dir / "data" / "predictions" / "predictions.parquet")
    assert predictions.schema.equals(milestone_pipeline.TABLE_SCHEMAS["predictions"])
    assert predictions.column("cand_1").to_pylist() == ["2301.12345"]


def test_pipeline_stops_if_parsing_fails(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    manual_labels = tmp_path / "manual_labels.csv"
    manual_labels.write_text("arxiv_id,citation_key,groundtruth_arxiv_id\n", encoding="utf-8")
    args = milestone_pipeline.build_argument_parser().parse_args(
        [
            "--ids",
            "2407.17839",
            "--manual-labels",
            str(manual_labels),
            "--source-dir",
            str(source_dir),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    _write_crawl_outputs(source_dir)
    monkeypatch.setattr(
        milestone_pipeline.arxiv_crawler,
        "crawl",
        lambda ids, out_dir, with_refs: {"requested": 1},
    )
    monkeypatch.setattr(milestone_pipeline.parsing_pipeline, "main", lambda argv: 2)
    matching_called = False

    def fail_if_called(**kwargs):
        nonlocal matching_called
        matching_called = True

    monkeypatch.setattr(milestone_pipeline.run_matching, "run", fail_if_called)

    with pytest.raises(RuntimeError, match="Milestone 2 parsing exited"):
        milestone_pipeline.run_pipeline(args)

    assert not matching_called
