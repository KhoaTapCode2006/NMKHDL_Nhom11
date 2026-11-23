import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model.data_cleaning import clean_bibtex_table, clean_references_table
from src.model.manual_labeling import (
    MANUAL_POOL_20_IDS,
    compute_pair_similarity,
    finalize_manual_labels,
    prepare_manual_review,
)


class ManualLabelingTests(unittest.TestCase):
    def test_manual_pool_has_20_unique_ids_across_4_ranges(self):
        self.assertEqual(len(MANUAL_POOL_20_IDS), 20)
        self.assertEqual(len(set(MANUAL_POOL_20_IDS)), 20)

    def test_compute_pair_similarity_exact_id_and_title_author(self):
        bib = clean_bibtex_table(
            pd.DataFrame(
                [
                    {
                        "arxiv_id": "2407.17850",
                        "citation_key": "rombach2022high",
                        "title": "High-resolution image synthesis with latent diffusion models",
                        "authors": "Rombach, Robin and Blattmann, Andreas",
                        "year": 2022,
                        "url": "https://arxiv.org/abs/2112.10752",
                    }
                ]
            )
        ).iloc[0]
        ref = clean_references_table(
            pd.DataFrame(
                [
                    {
                        "citing_arxiv_id": "2407.17850",
                        "ref_arxiv_id": "2112.10752",
                        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
                        "authors": ["Robin Rombach", "A. Blattmann"],
                        "year": 2021,
                    }
                ]
            )
        ).iloc[0]
        sim = compute_pair_similarity(bib, ref)
        self.assertTrue(sim["exact_id_match"])
        self.assertEqual(sim["total_score"], 1.0)
        self.assertGreater(sim["title_score"], 0.95)
        self.assertGreater(sim["author_score"], 0.95)

    def test_prepare_and_finalize_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            out_dir = root / "out"
            pub_dir = source_dir / "2506.10792"
            (pub_dir / "bibtex").mkdir(parents=True)
            (pub_dir / "v1").mkdir(parents=True)

            bib_rows = [
                {
                    "arxiv_id": "2506.10792",
                    "citation_key": f"ref_{i}",
                    "title": (
                        "Attention Is All You Need"
                        if i == 0
                        else f"Unmatched Paper Title Number {i}"
                    ),
                    "authors": ["Ashish Vaswani"] if i == 0 else [f"Author {i}"],
                    "year": 2017 if i == 0 else 2020,
                    "url": "https://arxiv.org/abs/1706.03762" if i == 0 else "",
                }
                for i in range(20)
            ]
            (pub_dir / "bibtex" / "bibtex.json").write_text(
                json.dumps(bib_rows),
                encoding="utf-8",
            )
            refs_rows = [
                {
                    "paperId": "s2_1",
                    "externalIds": {"ArXiv": "1706.03762"},
                    "title": "Attention Is All You Need",
                    "authors": [{"name": "Ashish Vaswani"}],
                    "year": 2017,
                },
                {
                    "paperId": "s2_2",
                    "externalIds": {"ArXiv": "2301.10140"},
                    "title": "The Semantic Scholar Open Data Platform",
                    "authors": [{"name": "Rodney Kinney"}],
                    "year": 2023,
                },
            ]
            (pub_dir / "v1" / "references.json").write_text(
                json.dumps(refs_rows),
                encoding="utf-8",
            )

            prep = prepare_manual_review(source_dir=source_dir, output_dir=out_dir)
            self.assertEqual(prep["publications_count"], 1)
            self.assertEqual(prep["entries_count"], 20)

            stats = finalize_manual_labels(
                review_csv_path=out_dir / "manual_review_sheet.csv",
                references_csv_path=out_dir / "references_cleaned.csv",
                output_dir=out_dir,
                min_publications=1,
                min_entries_per_pub=20,
                min_total_entries=20,
            )
            self.assertEqual(stats["total_manual_publications"], 1)
            self.assertEqual(stats["total_labeled_entries_Q"], 20)
            self.assertEqual(stats["total_positive_matches"], 1)
            self.assertEqual(stats["total_candidate_matrix_cells"], 40)


if __name__ == "__main__":
    unittest.main()
