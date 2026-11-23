import json
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model.feature_engineering import (
    FeatureExtractor,
    parse_hierarchy_for_features,
)
from src.model.auto_labeler import select_auto_label_publications, validate_auto_labels
from src.model.modeling import (
    PREDICTION_COLUMNS,
    assign_partitions,
    attach_labels,
    build_predictions_table,
    evaluate_rankings,
    mrr_at_k,
    postprocess_rankings,
    select_and_evaluate,
)
from src.model.run_matching import run

WORDS = (
    "graph neural network attention transformer quantum learning sparse optimal "
    "bayesian inference robust adaptive stochastic gradient language vision model "
    "kernel spectral manifold causal diffusion contrastive reinforcement policy"
).split()
SURNAMES = "smith nguyen tran garcia muller rossi tanaka kim silva cohen".split()


def synthetic_corpus(n_pubs=12, refs_per_pub=6, seed=0):
    """BibTeX entries and references; most entries cite one reference with noise."""
    rng = random.Random(seed)
    bib_rows, ref_rows, truth = [], [], []
    for p in range(n_pubs):
        pub = f"2501.{p + 1:05d}"
        for r in range(refs_per_pub):
            ref_id = f"2312.{p * 100 + r + 1:05d}"
            title = " ".join(rng.sample(WORDS, 6))
            authors = [f"{rng.choice('ABCDEFG')}. {rng.choice(SURNAMES).title()}" for _ in range(2)]
            year = rng.randint(2015, 2023)
            ref_rows.append({"citing_arxiv_id": pub, "ref_arxiv_id": ref_id, "title": title,
                             "authors": authors, "year": year, "paperId": f"s2-{ref_id}"})
            if r < refs_per_pub - 1:  # last reference is never cited by a BibTeX entry
                words = title.split()
                words.pop(rng.randrange(len(words)))  # noisy title
                bib_rows.append({"arxiv_id": pub, "citation_key": f"key{r}",
                                 "title": " ".join(words).title(),
                                 "authors": " and ".join(authors), "year": str(year)})
                truth.append({"arxiv_id": pub, "citation_key": f"key{r}", "groundtruth_arxiv_id": ref_id})
        # one entry without a true reference
        bib_rows.append({"arxiv_id": pub, "citation_key": "nomatch",
                         "title": "an unrelated book about cooking", "authors": "Jane Roe", "year": "1999"})
        truth.append({"arxiv_id": pub, "citation_key": "nomatch", "groundtruth_arxiv_id": None})
    return pd.DataFrame(bib_rows), pd.DataFrame(ref_rows), pd.DataFrame(truth)


class MetricTests(unittest.TestCase):
    def test_mrr_literal_and_matchable(self):
        rankings = {("p", "a"): ["x", "y"], ("p", "b"): ["y", "z"], ("p", "c"): ["x"]}
        truths = {("p", "a"): "x", ("p", "b"): "z", ("p", "c"): None}

        self.assertAlmostEqual(mrr_at_k(rankings, truths), (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(mrr_at_k(rankings, truths, include_unmatchable=False), 0.75)

    def test_true_reference_beyond_top5_scores_zero(self):
        rankings = {("p", "a"): ["c1", "c2", "c3", "c4", "c5", "truth"]}
        metrics = evaluate_rankings(rankings, {("p", "a"): "truth"})
        self.assertEqual(metrics["mrr@5"], 0.0)
        self.assertEqual(metrics["hit@5_matchable"], 0.0)

    def test_one_to_one_postprocessing_resolves_conflicts(self):
        scored = pd.DataFrame([
            {"arxiv_id": "p", "citation_key": "a", "candidate_ref_id": "r1", "score": 0.9, "feat_id_exact_match": 0},
            {"arxiv_id": "p", "citation_key": "a", "candidate_ref_id": "r2", "score": 0.8, "feat_id_exact_match": 0},
            {"arxiv_id": "p", "citation_key": "b", "candidate_ref_id": "r1", "score": 0.95, "feat_id_exact_match": 0},
            {"arxiv_id": "p", "citation_key": "b", "candidate_ref_id": "r2", "score": 0.1, "feat_id_exact_match": 0},
        ])
        plain = postprocess_rankings(scored, "none")
        fixed = postprocess_rankings(scored, "id_override+one_to_one")

        self.assertEqual(plain[("p", "a")][0], "r1")
        self.assertEqual(fixed[("p", "a")][0], "r2")
        self.assertEqual(fixed[("p", "b")][0], "r1")

    def test_predictions_table_has_one_row_per_entry(self):
        bibtex = pd.DataFrame([
            {"arxiv_id": "p1", "citation_key": "a"},
            {"arxiv_id": "p1", "citation_key": "unlabeled"},
            {"arxiv_id": "p2", "citation_key": "b"},
            {"arxiv_id": "other", "citation_key": "c"},
        ])
        labels = pd.DataFrame([
            {"arxiv_id": "p1", "citation_key": "a", "groundtruth_arxiv_id": "2301.00001"},
            {"arxiv_id": "p2", "citation_key": "b", "groundtruth_arxiv_id": None},
        ])
        rankings = {("p1", "a"): ["2301.00001", "2301.00002"]}
        table = build_predictions_table(bibtex, rankings, {"train": ["p1"], "test": ["p2"]}, labels)

        self.assertEqual(list(table.columns), PREDICTION_COLUMNS)
        self.assertEqual(len(table), 3)
        row = table[table["citation_key"] == "a"].iloc[0]
        self.assertEqual(row["cand_1"], "2301.00001")
        self.assertIsNone(row["cand_3"])
        self.assertIsNone(table[table["citation_key"] == "unlabeled"].iloc[0]["groundtruth_arxiv_id"])


class ModelingTests(unittest.TestCase):
    def test_trained_model_ranks_true_references_first(self):
        bib, refs, truth = synthetic_corpus()
        truth["label_source"] = ["manual" if int(a[-2:]) <= 6 else "auto" for a in truth["arxiv_id"]]
        matrix = FeatureExtractor().create_candidate_matrix(bib, refs, labels_df=None)
        matrix = attach_labels(matrix, truth)
        pubs = sorted(bib["arxiv_id"].unique())
        manual = [p for p in pubs if int(p[-2:]) <= 6]
        auto = [p for p in pubs if int(p[-2:]) > 6]
        split = {"test": manual[:1] + auto[:1], "valid": manual[1:2] + auto[1:2],
                 "train": manual[2:] + auto[2:]}
        results = select_and_evaluate(assign_partitions(matrix, split))

        self.assertIn(results.selected_postprocessing,
                      {"none", "id_override", "id_override+one_to_one"})
        self.assertGreaterEqual(results.test_metrics["all"]["mrr@5_matchable"], 0.9)
        self.assertIn("manual", results.test_metrics)
        self.assertIn("auto", results.test_metrics)
        self.assertTrue({"model", "postprocessing", "valid_manual_mrr@5_matchable"}
                        <= set(results.comparison.columns))


class HoangFixTests(unittest.TestCase):
    def test_hierarchy_features_read_milestone2_tables(self):
        elements = pd.DataFrame([
            {"element_id": "p#root1", "arxiv_id": "p", "kind": "document", "content": "root"},
            {"element_id": "p#sec1", "arxiv_id": "p", "kind": "section", "content": "Introduction"},
            {"element_id": "p#sec2", "arxiv_id": "p", "kind": "section", "content": "Method"},
            {"element_id": "p#s1", "arxiv_id": "p", "kind": "sentence", "content": r"As in \cite{a, b}."},
            {"element_id": "p#s2", "arxiv_id": "p", "kind": "sentence", "content": r"We use \citep[p.~2]{b}."},
        ])
        hierarchy = pd.DataFrame([
            {"arxiv_id": "p", "version": 1, "element_id": "p#sec1", "parent_id": "p#root1"},
            {"arxiv_id": "p", "version": 1, "element_id": "p#sec2", "parent_id": "p#root1"},
            {"arxiv_id": "p", "version": 1, "element_id": "p#s1", "parent_id": "p#sec1"},
            {"arxiv_id": "p", "version": 1, "element_id": "p#s2", "parent_id": "p#sec2"},
        ])
        summary = parse_hierarchy_for_features(elements, hierarchy)

        self.assertEqual(summary["p"]["a"], {"depth": 2, "frequency": 1, "is_intro_or_related": True})
        self.assertEqual(summary["p"]["b"]["frequency"], 2)
        self.assertTrue(summary["p"]["b"]["is_intro_or_related"])

    def test_auto_pool_is_random_rounded_up_and_balanced(self):
        pubs = [f"a{i}" for i in range(55)] + [f"b{i}" for i in range(55)]
        groups = {p: p[0] for p in pubs}
        selected = select_auto_label_publications(pubs, [], ratio=0.10, num_members=2, groups=groups)

        self.assertEqual(len(selected), 11)  # ceil(110 * 0.1), not floor
        self.assertNotEqual(selected, sorted(pubs)[:11])
        counts = pd.Series([p[0] for p in selected]).value_counts()
        self.assertLessEqual(abs(counts.get("a", 0) - counts.get("b", 0)), 1)

    def test_auto_label_validation_against_manual(self):
        manual = pd.DataFrame([
            {"arxiv_id": "p", "citation_key": "a", "groundtruth_arxiv_id": "2301.00001"},
            {"arxiv_id": "p", "citation_key": "b", "groundtruth_arxiv_id": "2301.00002"},
            {"arxiv_id": "p", "citation_key": "c", "groundtruth_arxiv_id": None},
        ])
        auto = pd.DataFrame([
            {"arxiv_id": "p", "citation_key": "a", "groundtruth_arxiv_id": "2301.00001"},
            {"arxiv_id": "p", "citation_key": "b", "groundtruth_arxiv_id": None},
            {"arxiv_id": "p", "citation_key": "c", "groundtruth_arxiv_id": None},
        ])
        result = validate_auto_labels(auto, manual)

        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["no_match_agreement"], 1.0)


class PipelineTests(unittest.TestCase):
    def test_run_matching_end_to_end(self):
        bib, refs, truth = synthetic_corpus(n_pubs=10, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "corpus"
            for pub, group in bib.groupby("arxiv_id"):
                (source / pub / "bibtex").mkdir(parents=True)
                (source / pub / "v1").mkdir()
                (source / pub / "bibtex" / "bibtex.json").write_text(
                    group.to_json(orient="records"), encoding="utf-8")
                pub_refs = refs[refs["citing_arxiv_id"] == pub]
                payload = [
                    {"paperId": r.paperId, "title": r.title, "year": r.year,
                     "publicationDate": f"{r.year}-01-15", "venue": "Venue",
                     "authors": [{"name": a} for a in r.authors],
                     "externalIds": {"ArXiv": r.ref_arxiv_id}}
                    for r in pub_refs.itertuples()
                ]
                (source / pub / "v1" / "references.json").write_text(json.dumps(payload), encoding="utf-8")
                (source / pub / "v1" / "metadata.json").write_text(
                    json.dumps({"arxiv_id": f"{pub}v1"}), encoding="utf-8")
            manual_pubs = sorted(bib["arxiv_id"].unique())[:4]
            manual_path = Path(tmp) / "manual_labels.csv"
            truth[truth["arxiv_id"].isin(manual_pubs)].to_csv(manual_path, index=False)

            summary = run(source, manual_path, Path(tmp) / "out", num_members=1,
                          auto_ratio=0.5, run_ablation=True)
            out = Path(tmp) / "out"
            predictions = pd.read_parquet(out / "predictions.parquet")
            references_table = pd.read_parquet(out / "references_table.parquet")

            for name in ("labels.csv", "split.json", "candidate_matrix.parquet",
                         "model_comparison.csv", "feature_importance.csv", "metrics.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(list(predictions.columns), PREDICTION_COLUMNS)
            split_pubs = {p for part in json.loads((out / "split.json").read_text()).values() for p in part}
            self.assertEqual(len(predictions), int(bib["arxiv_id"].isin(split_pubs).sum()))
            self.assertEqual(
                list(references_table.columns),
                ["arxiv_id", "ref_arxiv_id", "ref_title", "ref_authors", "ref_submission_date", "ref_s2_id"],
            )
            self.assertTrue(references_table["ref_s2_id"].str.startswith("s2-").all())
            self.assertGreaterEqual(summary["auto_publications"], 2)
            self.assertGreater(summary["test_metrics"]["all"]["mrr@5_matchable"], 0.8)


if __name__ == "__main__":
    unittest.main()
