"""Tests for FeatureExtractor and feature matrix generation."""

import unittest
import pandas as pd

from src.model.feature_engineering import FeatureExtractor


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_pairwise_feature_extraction(self):
        bib_row = pd.Series({
            "clean_title": "attention is all you need",
            "clean_year": 2017,
            "clean_authors_list": ["a_vaswani", "n_shazeer"],
            "extracted_arxiv_id": "1706.03762",
        })
        ref_row = pd.Series({
            "clean_title": "attention is all you need",
            "clean_year": 2017,
            "clean_authors_list": ["a_vaswani", "n_shazeer"],
            "ref_arxiv_id": "1706.03762",
        })

        features = self.extractor.extract_pair_features(
            bib_row=bib_row,
            ref_row=ref_row,
            pub_id="2411.00001",
            citation_key="vaswani2017",
        )

        self.assertEqual(features["sim_title_exact"], 1.0)
        self.assertAlmostEqual(features["sim_title_ratio"], 1.0)
        self.assertEqual(features["sim_author_exact"], 1.0)
        self.assertEqual(features["feat_year_exact"], 1.0)
        self.assertEqual(features["feat_id_exact_match"], 1.0)
        self.assertEqual(features["feat_has_bib_arxiv_id"], 1.0)

    def test_candidate_matrix_generation_with_labels(self):
        bibtex_df = pd.DataFrame([
            {
                "arxiv_id": "2411.00001",
                "citation_key": "vaswani2017",
                "title": "Attention Is All You Need",
                "authors": "A. Vaswani and N. Shazeer",
                "year": 2017,
                "eprint": "1706.03762",
            },
            {
                "arxiv_id": "2411.00001",
                "citation_key": "unmatched_book",
                "title": "Some random physics book",
                "authors": "John Doe",
                "year": 1999,
            },
        ])

        references_df = pd.DataFrame([
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1706.03762",
                "title": "Attention is all you need",
                "authors": ["A. Vaswani", "N. Shazeer"],
                "year": 2017,
            },
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["J. Devlin"],
                "year": 2018,
            },
        ])

        labels_df = pd.DataFrame([
            {
                "arxiv_id": "2411.00001",
                "citation_key": "vaswani2017",
                "groundtruth_arxiv_id": "1706.03762",
            },
            {
                "arxiv_id": "2411.00001",
                "citation_key": "unmatched_book",
                "groundtruth_arxiv_id": None,
            },
        ])

        matrix = self.extractor.create_candidate_matrix(
            bibtex_df=bibtex_df,
            references_df=references_df,
            labels_df=labels_df,
        )

        # 2 bib entries x 2 candidates = 4 candidate rows
        self.assertEqual(len(matrix), 4)
        self.assertIn("label", matrix.columns)
        self.assertIn("sim_title_ratio", matrix.columns)
        self.assertIn("feat_title_sim_rank", matrix.columns)

        # Only 1 true match (vaswani2017 -> 1706.03762)
        self.assertEqual(matrix["label"].sum(), 1)

    def test_unlabeled_entries_carry_no_rows(self):
        bibtex_df = pd.DataFrame([
            {
                "arxiv_id": "2411.00001",
                "citation_key": "entry_labeled",
                "title": "Title 1",
            },
            {
                "arxiv_id": "2411.00001",
                "citation_key": "entry_unlabeled",
                "title": "Title 2",
            },
        ])
        references_df = pd.DataFrame([
            {
                "citing_arxiv_id": "2411.00001",
                "ref_arxiv_id": "1111.00001",
                "title": "Title 1",
            }
        ])
        # labels_df only has entry_labeled
        labels_df = pd.DataFrame([
            {
                "arxiv_id": "2411.00001",
                "citation_key": "entry_labeled",
                "groundtruth_arxiv_id": "1111.00001",
            }
        ])

        matrix = self.extractor.create_candidate_matrix(
            bibtex_df=bibtex_df,
            references_df=references_df,
            labels_df=labels_df,
        )

        # entry_unlabeled should be excluded per spec: "Unlabeled entries simply carry no rows."
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix.iloc[0]["citation_key"], "entry_labeled")


if __name__ == "__main__":
    unittest.main()
