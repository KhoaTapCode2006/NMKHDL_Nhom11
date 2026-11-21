import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model.data_cleaning import (
    clean_authors,
    clean_bibtex_table,
    clean_references_table,
    clean_title,
    clean_year,
    extract_arxiv_id,
    flatten_references_json,
    normalize_arxiv_id,
    collect_bibtex_tables,
)


class DataCleaningTests(unittest.TestCase):
    def test_collects_all_publication_bibtex_json_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for arxiv_id in ("2501.00001", "2501.00002"):
                bib_dir = root / arxiv_id / "bibtex"
                bib_dir.mkdir(parents=True)
                (bib_dir / "bibtex.json").write_text(
                    json.dumps(
                        [
                            {
                                "arxiv_id": arxiv_id,
                                "citation_key": f"key_{arxiv_id}",
                                "title": "A Study",
                                "authors": [],
                                "year": 2024,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            result = collect_bibtex_tables(root)

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["arxiv_id"]), {"2501.00001", "2501.00002"})

    def test_collect_prefers_json_over_duplicate_csv_in_same_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bib_dir = root / "2501.00003" / "bibtex"
            bib_dir.mkdir(parents=True)
            row = {
                "arxiv_id": "2501.00003",
                "citation_key": "key1",
                "title": "A Study",
                "authors": [],
                "year": 2024,
            }
            (bib_dir / "bibtex.json").write_text(
                json.dumps([row]),
                encoding="utf-8",
            )
            pd.DataFrame([row]).to_csv(bib_dir / "bibtex.csv", index=False)

            result = collect_bibtex_tables(root)

        self.assertEqual(len(result), 1)

    def test_arxiv_id_normalization_and_extraction(self):
        self.assertEqual(normalize_arxiv_id("arXiv:1706.03762v5"), "1706.03762")
        self.assertEqual(normalize_arxiv_id("hep-th/9901001v2"), "hep-th/9901001")
        self.assertEqual(
            extract_arxiv_id("Preprint: https://arxiv.org/abs/2301.01234v3"),
            "2301.01234",
        )

    def test_title_cleanup_is_reproducible(self):
        title = r"\emph{End-to-End} Study: Caf{e} & {\textbf{Attention}}"
        cleaned = clean_title(title)
        self.assertEqual(cleaned, "end to end study cafe attention")
        self.assertEqual(clean_title(title), cleaned)

    def test_year_cleanup_extracts_four_digit_year(self):
        self.assertEqual(clean_year("July 2021"), 2021)
        self.assertEqual(clean_year("2020a"), 2020)
        self.assertIsNone(clean_year("no year"))

    def test_author_cleanup_supports_forward_reverse_and_initial_lists(self):
        self.assertEqual(clean_authors("Ashish Vaswani"), ["a_vaswani"])
        self.assertEqual(clean_authors("Vaswani, Ashish"), ["a_vaswani"])
        self.assertEqual(
            clean_authors("A. Vaswani, N. Shazeer"),
            ["a_vaswani", "n_shazeer"],
        )
        self.assertEqual(clean_authors(["John Smith", "Jane Doe"]), ["j_smith", "j_doe"])
        self.assertEqual(clean_authors("et al."), [])

    def test_bibtex_schema_is_preserved_and_cleaned_columns_added(self):
        bibtex = pd.DataFrame(
            [
                {
                    "arxiv_id": "2301.01234",
                    "citation_key": "key1",
                    "title": r"\emph{A--Reliable Study}",
                    "authors": "Vaswani, Ashish and Smith, John",
                    "year": "2021",
                    "eprint": "arXiv:2401.00001v2",
                }
            ]
        )
        cleaned = clean_bibtex_table(bibtex)
        self.assertEqual(cleaned.iloc[0]["citation_key"], "key1")
        self.assertEqual(cleaned.iloc[0]["arxiv_id"], "2301.01234")
        self.assertEqual(cleaned.iloc[0]["clean_title"], "a reliable study")
        self.assertEqual(
            cleaned.iloc[0]["clean_authors_list"],
            ["a_vaswani", "j_smith"],
        )
        self.assertEqual(cleaned.iloc[0]["clean_year"], 2021)
        self.assertEqual(cleaned.iloc[0]["extracted_arxiv_id"], "2401.00001")

    def test_references_table_filters_invalid_arxiv_and_deduplicates_pairs(self):
        references = pd.DataFrame(
            [
                {
                    "citing_arxiv_id": "1706.03762v5",
                    "ref_arxiv_id": "2301.01234v2",
                    "title": "A Test Paper",
                    "authors": [{"name": "Jane Smith"}],
                    "year": "2020",
                },
                {
                    "citing_arxiv_id": "1706.03762v5",
                    "ref_arxiv_id": "2301.01234v1",
                    "title": "A Test Paper",
                    "authors": [{"name": "Jane Smith"}],
                    "year": "2020",
                },
                {
                    "citing_arxiv_id": "1706.03762",
                    "ref_arxiv_id": "",
                    "title": "Missing reference",
                    "authors": [],
                    "year": None,
                },
            ]
        )
        cleaned = clean_references_table(references)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["citing_arxiv_id"], "1706.03762")
        self.assertEqual(cleaned.iloc[0]["ref_arxiv_id"], "2301.01234")
        self.assertEqual(cleaned.iloc[0]["clean_authors_list"], ["j_smith"])

    def test_flatten_references_json_uses_metadata_citing_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            publication = Path(temp_dir) / "1706.03762"
            version = publication / "v1"
            version.mkdir(parents=True)
            (version / "metadata.json").write_text(
                json.dumps({"arxiv_id": "1706.03762v5"}),
                encoding="utf-8",
            )
            (version / "references.json").write_text(
                json.dumps(
                    [
                        {
                            "title": "Attention Study",
                            "year": 2017,
                            "authors": [{"name": "Ashish Vaswani"}],
                            "externalIds": {"ArXiv": "2301.01234v2"},
                        },
                        {
                            "title": "No arXiv ID",
                            "year": 2020,
                            "authors": [],
                            "externalIds": {},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = flatten_references_json(Path(temp_dir))

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["citing_arxiv_id"], "1706.03762")
        self.assertEqual(result.iloc[0]["ref_arxiv_id"], "2301.01234")


if __name__ == "__main__":
    unittest.main()
