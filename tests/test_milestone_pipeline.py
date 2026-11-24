"""Unit tests for src/milestone_pipeline.py (stages, sharded Parquet, schemas)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    papers = pq.read_table(dataset_dir / "data" / "papers")
    refs = pq.read_table(dataset_dir / "data" / "references")
    elements = pq.read_table(dataset_dir / "data" / "elements")
    hierarchy = pq.read_table(dataset_dir / "data" / "hierarchy")
    bibtex = pq.read_table(dataset_dir / "data" / "bibtex")

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
        assert list((dataset_dir / "data" / "papers").glob("*.parquet"))
        assert list((dataset_dir / "data" / "references").glob("*.parquet"))
        _write_parser_outputs(source_dir)
        return 0

    def fake_matching(**kwargs):
        stages.append("matching")
        assert all(
            list((dataset_dir / "data" / table).glob("*.parquet"))
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
    predictions = pq.read_table(dataset_dir / "data" / "predictions")
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


def _write_sample_crawled_and_parsed_pub(
    source_dir: Path,
    arxiv_id: str = "2407.17839",
    weird_year: object = "2021a",
) -> Path:
    pub_dir = source_dir / arxiv_id
    v1_dir = pub_dir / "v1"
    v2_dir = pub_dir / "v2"
    elements_dir = pub_dir / "elements"
    bibtex_dir = pub_dir / "bibtex"
    for d in (v1_dir, v2_dir, elements_dir, bibtex_dir):
        d.mkdir(parents=True, exist_ok=True)

    _write_json(
        v1_dir / "metadata.json",
        {
            "arxiv_id": f"{arxiv_id}v1",
            "base_id": arxiv_id,
            "version": "1",
            "title": "Sample Paper Title",
            "authors": ["Alice Smith", "Bob Jones"],
            "submission_date": "2024-07-01T00:00:00Z",
            "version_date": "2024-07-01T00:00:00Z",
            "revised_dates": ["2024-07-15T00:00:00Z"],
            "venue": "NeurIPS 2024",
        },
    )
    _write_json(
        v2_dir / "metadata.json",
        {
            "arxiv_id": f"{arxiv_id}v2",
            "base_id": arxiv_id,
            "version": "2",
            "title": "Sample Paper Title Revised",
            "authors": ["Alice Smith", "Bob Jones"],
            "submission_date": "2024-07-01T00:00:00Z",
            "version_date": "2024-07-15T00:00:00Z",
            "revised_dates": ["2024-07-15T00:00:00Z"],
            "venue": "NeurIPS 2024",
        },
    )
    refs = [
        {
            "paperId": "s2_abc",
            "externalIds": {"ArXiv": "1706.03762"},
            "title": "Attention Is All You Need",
            "authors": [{"name": "Ashish Vaswani"}],
            "year": 2017,
            "publicationDate": "2017-06-12",
            "venue": "NeurIPS",
        },
        {
            "paperId": "s2_non_arxiv",
            "externalIds": {"DOI": "10.1000/xyz"},
            "title": "Journal Only Paper",
            "authors": [{"name": "Carol Dan"}],
            "year": 2019,
        },
    ]
    _write_json(v1_dir / "references.json", refs)
    _write_json(v2_dir / "references.json", refs)

    _write_json(
        elements_dir / "elements.json",
        [
            {
                "element_id": f"{arxiv_id}#root1",
                "arxiv_id": arxiv_id,
                "kind": "document",
                "content": "Version 1 root",
            },
            {
                "element_id": f"{arxiv_id}#h1",
                "arxiv_id": arxiv_id,
                "kind": "section",
                "content": "Introduction",
            },
            {
                "element_id": f"{arxiv_id}#s1",
                "arxiv_id": arxiv_id,
                "kind": "sentence",
                "content": r"We build on Transformers \cite{vaswani2017}.",
            },
        ],
    )
    _write_json(
        elements_dir / "hierarchy.json",
        [
            {
                "arxiv_id": arxiv_id,
                "version": 1,
                "element_id": f"{arxiv_id}#h1",
                "parent_id": f"{arxiv_id}#root1",
            },
            {
                "arxiv_id": arxiv_id,
                "version": 1,
                "element_id": f"{arxiv_id}#s1",
                "parent_id": f"{arxiv_id}#h1",
            },
        ],
    )
    _write_json(
        bibtex_dir / "bibtex.json",
        [
            {
                "arxiv_id": arxiv_id,
                "citation_key": "vaswani2017",
                "entry_type": "article",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani"],
                "year": weird_year,
                "doi": "",
                "journal": "NeurIPS",
                "url": "https://arxiv.org/abs/1706.03762",
            }
        ],
    )
    return pub_dir


class MilestonePipelineOptimizationTests(unittest.TestCase):
    def test_sharded_export_multiple_sources_and_noisy_year_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "src1"
            src2 = root / "src2"
            dataset_dir = root / "hf_dataset"
            match_dir = root / "matching_out"
            match_dir.mkdir(parents=True)

            _write_sample_crawled_and_parsed_pub(src1, "2407.17839", weird_year="2017a")
            _write_sample_crawled_and_parsed_pub(src2, "2506.10788", weird_year="invalid")

            pd.DataFrame(
                [
                    {
                        "arxiv_id": "2407.17839",
                        "partition": "test",
                        "citation_key": "vaswani2017",
                        "groundtruth_arxiv_id": "1706.03762",
                        "cand_1": "1706.03762",
                        "cand_2": None,
                        "cand_3": None,
                        "cand_4": None,
                        "cand_5": None,
                    }
                ]
            ).to_parquet(match_dir / "predictions.parquet", index=False)

            rc = milestone_pipeline.main(
                [
                    "--export-only",
                    "--source-dir",
                    str(src1),
                    str(src2),
                    "--matching-output-dir",
                    str(match_dir),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--shard-rows",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)

            elem_shards = sorted((dataset_dir / "data" / "elements").glob("*.parquet"))
            self.assertEqual(len(elem_shards), 3)
            elem_table = pq.read_table(dataset_dir / "data" / "elements")
            self.assertEqual(elem_table.num_rows, 6)
            self.assertEqual(elem_table.schema, milestone_pipeline.TABLE_SCHEMAS["elements"])

            bib_df = pq.read_table(dataset_dir / "data" / "bibtex").to_pandas()
            by_pub = dict(zip(bib_df["arxiv_id"], bib_df["year"]))
            self.assertEqual(int(by_pub["2407.17839"]), 2017)
            self.assertTrue(pd.isna(by_pub["2506.10788"]))

            papers_df = pq.read_table(dataset_dir / "data" / "papers").to_pandas()
            self.assertEqual(sorted(papers_df["arxiv_id"].tolist()), ["2407.17839", "2506.10788"])

            refs_df = pq.read_table(dataset_dir / "data" / "references").to_pandas()
            self.assertEqual(len(refs_df), 2)
            self.assertTrue((refs_df["ref_arxiv_id"] == "1706.03762").all())

            card_text = (dataset_dir / "README.md").read_text(encoding="utf-8")
            for table_name in milestone_pipeline.TABLE_SCHEMAS:
                self.assertIn(f"- config_name: {table_name}", card_text)
                self.assertIn(f"path: data/{table_name}/*.parquet", card_text)

    def test_skip_match_does_not_require_manual_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            dataset_dir = root / "hf"

            def _fake_crawl(ids, save_dir, with_refs=True):
                for pid in ids:
                    pub = Path(save_dir) / pid / "v1"
                    pub.mkdir(parents=True, exist_ok=True)
                    _write_json(
                        pub / "metadata.json",
                        {
                            "arxiv_id": f"{pid}v1",
                            "base_id": pid,
                            "version": "1",
                            "title": "Test",
                            "authors": ["Author A"],
                            "submission_date": "2024-07-01T00:00:00Z",
                            "revised_dates": [],
                        },
                    )
                    (pub / "references.json").write_text("[]", encoding="utf-8")
                    (pub / "main.tex").write_text(
                        r"\documentclass{article}\begin{document}\section{Intro}Hello world.\end{document}",
                        encoding="utf-8",
                    )
                return {"requested": len(ids), "resolved": len(ids), "missing_ids": []}

            with mock.patch.object(milestone_pipeline.arxiv_crawler, "crawl", side_effect=_fake_crawl):
                rc = milestone_pipeline.main(
                    [
                        "--ids",
                        "2407.17839",
                        "--skip-match",
                        "--source-dir",
                        str(source_dir),
                        "--dataset-dir",
                        str(dataset_dir),
                        "--work-dir",
                        str(root / "work"),
                    ]
                )
            self.assertEqual(rc, 0)
            for name in ("papers", "references", "elements", "hierarchy", "bibtex"):
                self.assertTrue(list((dataset_dir / "data" / name).glob("*.parquet")))
            self.assertFalse((dataset_dir / "data" / "predictions").exists())


if __name__ == "__main__":
    unittest.main()
