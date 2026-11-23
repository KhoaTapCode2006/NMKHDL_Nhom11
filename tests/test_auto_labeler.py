"""Tests for AutoReferenceMatcher and auto-labeling pipeline."""

import unittest
import pandas as pd

from src.model.auto_labeler import (
    AutoReferenceMatcher,
    run_auto_labeling_pipeline,
    select_auto_label_publications,
    split_publications_train_val_test,
)


class AutoLabelerTests(unittest.TestCase):
    def setUp(self):
        self.matcher = AutoReferenceMatcher()

    def test_direct_arxiv_id_match(self):
        bib_row = pd.Series({
            "arxiv_id": "2411.00001",
            "citation_key": "vaswani2017attention",
            "extracted_arxiv_id": "1706.03762",
            "clean_title": "attention is all you need",
            "clean_year": 2017,
            "clean_authors_list": ["a_vaswani", "n_shazeer"],
        })
        ref_candidates = pd.DataFrame([
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1706.03762",
                "clean_title": "attention is all you need",
                "clean_year": 2017,
                "clean_authors_list": ["a_vaswani", "n_shazeer"],
            },
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1810.04805",
                "clean_title": "bert pre-training of deep bidirectional transformers",
                "clean_year": 2018,
                "clean_authors_list": ["j_devlin"],
            },
        ])

        result = self.matcher.match_entry(bib_row, ref_candidates)
        self.assertTrue(result.matched)
        self.assertEqual(result.groundtruth_arxiv_id, "1706.03762")
        self.assertEqual(result.method, "direct_arxiv_id")
        self.assertEqual(result.confidence, 1.0)

    def test_high_title_and_metadata_match(self):
        bib_row = pd.Series({
            "arxiv_id": "2411.00001",
            "citation_key": "he2016deep",
            "extracted_arxiv_id": None,
            "clean_title": "deep residual learning for image recognition",
            "clean_year": 2016,
            "clean_authors_list": ["k_he", "x_zhang", "s_ren", "j_sun"],
        })
        ref_candidates = pd.DataFrame([
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1512.03385",
                "clean_title": "deep residual learning for image recognition",
                "clean_year": 2015,
                "clean_authors_list": ["k_he", "x_zhang"],
            },
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1409.1556",
                "clean_title": "very deep convolutional networks for large scale image recognition",
                "clean_year": 2014,
                "clean_authors_list": ["k_simonyan", "a_zisserman"],
            },
        ])

        result = self.matcher.match_entry(bib_row, ref_candidates)
        self.assertTrue(result.matched)
        self.assertEqual(result.groundtruth_arxiv_id, "1512.03385")
        self.assertIn("title", result.method)

    def test_no_match_returns_negative_row(self):
        bib_row = pd.Series({
            "arxiv_id": "2411.00001",
            "citation_key": "unrelated_book",
            "extracted_arxiv_id": None,
            "clean_title": "classical mechanics and modern quantum field theory",
            "clean_year": 1980,
            "clean_authors_list": ["a_author"],
        })
        ref_candidates = pd.DataFrame([
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1706.03762",
                "clean_title": "attention is all you need",
                "clean_year": 2017,
                "clean_authors_list": ["a_vaswani"],
            }
        ])

        result = self.matcher.match_entry(bib_row, ref_candidates)
        self.assertFalse(result.matched)
        self.assertIsNone(result.groundtruth_arxiv_id)

    def test_select_auto_label_publications(self):
        all_pubs = [f"pub_{i}" for i in range(50)]
        manual_pool = [f"pub_{i}" for i in range(5)]

        selected = select_auto_label_publications(
            available_publications=all_pubs,
            manual_labeled_publications=manual_pool,
            ratio=0.10,
            num_members=4,
        )
        # 45 remaining. 10% is 4, but min 2*k = 8. So min 8.
        self.assertGreaterEqual(len(selected), 8)
        self.assertTrue(all(p not in manual_pool for p in selected))

    def test_split_publications_train_val_test(self):
        k = 4
        manual_pubs = [f"m_{i}" for i in range(20)]  # 5*k = 20
        auto_pubs = [f"a_{i}" for i in range(12)]    # >= 2*k = 8

        splits = split_publications_train_val_test(manual_pubs, auto_pubs, num_members=k, seed=123)

        # Test set: k manual + k auto = 8
        self.assertEqual(len(splits["test"]), 2 * k)
        # Validation set: k manual + k auto = 8
        self.assertEqual(len(splits["valid"]), 2 * k)
        # Training set: remaining manual (12) + remaining auto (4) = 16
        self.assertEqual(len(splits["train"]), (20 - 2 * k) + (12 - 2 * k))

        # Disjoint sets
        all_split_pubs = splits["test"] + splits["valid"] + splits["train"]
        self.assertEqual(len(all_split_pubs), len(set(all_split_pubs)))


if __name__ == "__main__":
    unittest.main()
