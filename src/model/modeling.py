"""Milestone 3 Phase 4: modeling and evaluation of the reference matcher.

Framing
-------
Reference matching is treated as a **ranking** problem. For every BibTeX entry
of a publication, the candidates are the arXiv-keyed references crawled for the
same publication (the m x n candidate matrix of the statement). A model gives
each (entry, candidate) pair a score; sorting the candidates of an entry by
score gives its ranked list, whose top 5 are ``cand_1`` .. ``cand_5``.

Models are pointwise scorers trained on the labeled pairs (label 1 for the true
reference, 0 otherwise, all-zero rows for entries without a true reference):

* ``heuristic_title``: no training, title similarity plus an arXiv-ID bonus
  (the baseline to beat);
* logistic regression (scaled features, balanced classes);
* histogram gradient boosting and random forest (non-linear interactions).

No pretrained model or external knowledge is used.

Evaluation
----------
MRR@5 = mean over labeled entries of 1/rank of the true reference, 0 when it is
not in the top 5. The statement does not say how entries *without* a true
reference count, so two numbers are reported:

* ``mrr@5``: literal reading, such entries count as 0;
* ``mrr@5_matchable``: only entries that have a true reference.

Every metric is also split by label source. Automatic labels come from the same
title/author similarities used as features, so the auto part of a split is
optimistic; model selection therefore uses the **manual** part of the
validation set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.model.data_cleaning import normalize_arxiv_id
from src.model.feature_engineering import FeatureExtractor

logger = logging.getLogger(__name__)

KEY = ["arxiv_id", "citation_key"]
TOP_K = 5
EntryKey = Tuple[str, str]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "title": [
        "sim_title_exact", "sim_title_ratio", "sim_title_token_sort", "sim_title_token_set",
        "sim_title_jaccard_words", "sim_title_char_3gram", "feat_title_length_diff",
        "feat_title_word_count_diff",
    ],
    "author": [
        "sim_author_exact", "sim_author_jaccard", "feat_author_first_match",
        "feat_author_count_diff", "feat_author_missing",
    ],
    "year": ["feat_year_exact", "feat_year_diff", "feat_year_missing"],
    "identifier": ["feat_id_exact_match", "feat_has_bib_arxiv_id"],
    "context": ["feat_citation_depth", "feat_citation_frequency", "feat_is_in_intro_or_related"],
    "relative": ["feat_title_sim_rank", "feat_title_sim_diff_to_max", "feat_candidate_count"],
}

POSTPROCESSING_MODES = ("none", "id_override", "id_override+one_to_one")


def feature_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    excluded = set(exclude)
    return [c for c in FeatureExtractor.FEATURE_NAMES if c in frame.columns and c not in excluded]


# --------------------------------------------------------------------------- #
# Labels and partitions
# --------------------------------------------------------------------------- #
def normalize_labels(labels: pd.DataFrame, default_source: str = "manual") -> pd.DataFrame:
    """Keep ``arxiv_id, citation_key, groundtruth_arxiv_id, label_source``."""
    frame = labels.copy()
    if "label_source" not in frame.columns:
        frame["label_source"] = default_source
    frame["arxiv_id"] = frame["arxiv_id"].astype(str).str.strip()
    frame["citation_key"] = frame["citation_key"].astype(str).str.strip()

    def norm(value: object) -> Optional[str]:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        return normalize_arxiv_id(value)

    truth = frame["groundtruth_arxiv_id"].map(norm).astype(object)
    frame["groundtruth_arxiv_id"] = truth.where(truth.notna(), None)
    frame = frame.drop_duplicates(subset=KEY, keep="first")
    return frame[KEY + ["groundtruth_arxiv_id", "label_source"]].reset_index(drop=True)


def attach_labels(matrix: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add ``groundtruth_arxiv_id``, ``label_source``, ``is_labeled`` and ``label``.

    ``label`` is 1 for the true candidate, 0 for the others, and NaN for the
    rows of unlabeled entries (they are scored for the predictions table but
    never used for training or evaluation).
    """
    frame = matrix.copy()
    frame["arxiv_id"] = frame["arxiv_id"].astype(str)
    frame["citation_key"] = frame["citation_key"].astype(str)
    frame = frame.drop(columns=[c for c in ("label", "groundtruth_arxiv_id", "label_source")
                                if c in frame.columns])
    frame = frame.merge(normalize_labels(labels), on=KEY, how="left", indicator=True)
    frame["is_labeled"] = frame.pop("_merge").eq("both")
    candidate = frame["candidate_ref_id"].map(lambda v: normalize_arxiv_id(v) or str(v))
    frame["label"] = np.where(
        frame["is_labeled"],
        (candidate == frame["groundtruth_arxiv_id"]).astype(float),
        np.nan,
    )
    return frame


def assign_partitions(frame: pd.DataFrame, split: Dict[str, Sequence[str]]) -> pd.DataFrame:
    partition_of = {str(pub): part for part, pubs in split.items() for pub in pubs}
    frame = frame.copy()
    frame["partition"] = frame["arxiv_id"].astype(str).map(partition_of)
    return frame


# --------------------------------------------------------------------------- #
# Ranking and metrics
# --------------------------------------------------------------------------- #
def rank_candidates(scored: pd.DataFrame, score_col: str = "score") -> Dict[EntryKey, List[str]]:
    """Candidates of every entry sorted by score (ties broken by candidate ID)."""
    if scored.empty:
        return {}
    ordered = scored.sort_values(KEY + [score_col, "candidate_ref_id"],
                                 ascending=[True, True, False, True])
    return {
        (str(pub), str(key)): list(group["candidate_ref_id"].astype(str))
        for (pub, key), group in ordered.groupby(KEY, sort=False)
    }


def truth_by_entry(frame: pd.DataFrame) -> Dict[EntryKey, Optional[str]]:
    """True reference of every labeled entry (None when it has none)."""
    labeled = frame[frame["is_labeled"]] if "is_labeled" in frame.columns else frame
    entries = labeled.drop_duplicates(subset=KEY)
    return {
        (str(row.arxiv_id), str(row.citation_key)): (
            row.groundtruth_arxiv_id if isinstance(row.groundtruth_arxiv_id, str) else None
        )
        for row in entries.itertuples(index=False)
    }


def reciprocal_rank(ranking: Sequence[str], truth: Optional[str], k: int = TOP_K) -> float:
    if not truth:
        return 0.0
    for position, candidate in enumerate(list(ranking)[:k], start=1):
        if normalize_arxiv_id(candidate) == truth or candidate == truth:
            return 1.0 / position
    return 0.0


def mrr_at_k(
    rankings: Dict[EntryKey, Sequence[str]],
    truths: Dict[EntryKey, Optional[str]],
    k: int = TOP_K,
    include_unmatchable: bool = True,
) -> float:
    """MRR@k over the entries in ``truths``.

    ``include_unmatchable=True`` follows the statement literally: |Q| counts
    every labeled entry and entries without a true reference score 0.
    """
    entries = [e for e, t in truths.items() if include_unmatchable or t]
    if not entries:
        return float("nan")
    return float(np.mean([reciprocal_rank(rankings.get(e, []), truths[e], k) for e in entries]))


def evaluate_rankings(
    rankings: Dict[EntryKey, Sequence[str]],
    truths: Dict[EntryKey, Optional[str]],
    k: int = TOP_K,
) -> Dict[str, float]:
    matchable = {e: t for e, t in truths.items() if t}

    def hit(n: int) -> float:
        if not matchable:
            return float("nan")
        return float(np.mean([reciprocal_rank(rankings.get(e, []), t, n) > 0
                              for e, t in matchable.items()]))

    return {
        "entries": len(truths),
        "matchable_entries": len(matchable),
        f"mrr@{k}": mrr_at_k(rankings, truths, k, include_unmatchable=True),
        f"mrr@{k}_matchable": mrr_at_k(rankings, truths, k, include_unmatchable=False),
        "hit@1_matchable": hit(1),
        f"hit@{k}_matchable": hit(k),
    }


def evaluate_by_source(
    rankings: Dict[EntryKey, Sequence[str]], frame: pd.DataFrame, k: int = TOP_K
) -> Dict[str, Dict[str, float]]:
    """Metrics on all labeled entries and separately per label source."""
    result = {"all": evaluate_rankings(rankings, truth_by_entry(frame), k)}
    if "label_source" in frame.columns:
        for source in sorted(frame.loc[frame["is_labeled"], "label_source"].dropna().unique()):
            part = frame[frame["label_source"] == source]
            result[str(source)] = evaluate_rankings(rankings, truth_by_entry(part), k)
    return result


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class HeuristicModel:
    """Baseline: rank by title similarity, an extracted arXiv ID wins outright."""

    name = "heuristic_title"

    def __init__(self, feature: str = "sim_title_token_sort"):
        self.feature = feature
        self.features: List[str] = [feature, "feat_id_exact_match"]

    def fit(self, frame: pd.DataFrame) -> "HeuristicModel":
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        base = frame[self.feature].fillna(0.0).to_numpy(dtype=float)
        bonus = frame.get("feat_id_exact_match", pd.Series(0.0, index=frame.index))
        return base + 10.0 * bonus.fillna(0.0).to_numpy(dtype=float)


class SklearnModel:
    """Pointwise scorer: probability that a candidate is the true reference."""

    def __init__(self, name: str, estimator_factory: Callable[[], object],
                 features: Optional[List[str]] = None):
        self.name = name
        self.estimator_factory = estimator_factory
        self.features = features
        self.estimator = None

    def fit(self, frame: pd.DataFrame) -> "SklearnModel":
        train = frame[frame["label"].notna()]
        if self.features is None:
            self.features = feature_columns(train)
        y = train["label"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError(f"{self.name}: training data needs positive and negative pairs")
        self.estimator = self.estimator_factory()
        self.estimator.fit(train[self.features].fillna(0.0).to_numpy(dtype=float), y)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].fillna(0.0).to_numpy(dtype=float)
        return self.estimator.predict_proba(x)[:, 1]


def default_model_factories(seed: int = 42) -> Dict[str, Callable[[Optional[List[str]]], object]]:
    """Candidate models compared on the validation set."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def logreg(c: float):
        return lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
        )

    def hgb(learning_rate: float, max_depth: Optional[int]):
        return lambda: HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth, max_iter=300,
            class_weight="balanced", random_state=seed,
        )

    def forest():
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )

    return {
        "heuristic_title": lambda features=None: HeuristicModel(),
        "logreg_C1": lambda features=None: SklearnModel("logreg_C1", logreg(1.0), features),
        "logreg_C0.1": lambda features=None: SklearnModel("logreg_C0.1", logreg(0.1), features),
        "hgb_lr0.1": lambda features=None: SklearnModel("hgb_lr0.1", hgb(0.1, None), features),
        "hgb_lr0.05_d4": lambda features=None: SklearnModel("hgb_lr0.05_d4", hgb(0.05, 4), features),
        "random_forest": lambda features=None: SklearnModel("random_forest", forest, features),
    }


# --------------------------------------------------------------------------- #
# Post-processing (allowed by the statement)
# --------------------------------------------------------------------------- #
def postprocess_rankings(scored: pd.DataFrame, mode: str = "none") -> Dict[EntryKey, List[str]]:
    """Turn scores into rankings, optionally with two corrections.

    ``id_override``: a candidate whose arXiv ID was found inside the BibTeX
    entry (eprint, url, journal) goes to rank 1.
    ``one_to_one``: after deduplication every reference is cited by at most
    one entry, so within a publication each entry's rank 1 becomes its
    candidate in the best one-to-one assignment (Hungarian algorithm).
    """
    if mode not in POSTPROCESSING_MODES:
        raise ValueError(f"Unknown post-processing mode: {mode}")
    frame = scored.copy()
    if mode.startswith("id_override") and "feat_id_exact_match" in frame.columns:
        frame["score"] = frame["score"] + 1000.0 * frame["feat_id_exact_match"].fillna(0.0)
    rankings = rank_candidates(frame)
    if not mode.endswith("one_to_one"):
        return rankings

    from scipy.optimize import linear_sum_assignment

    for pub, group in frame.groupby("arxiv_id", sort=False):
        entries = sorted(group["citation_key"].astype(str).unique())
        candidates = sorted(group["candidate_ref_id"].astype(str).unique())
        if len(entries) < 2 or len(candidates) < 2:
            continue
        e_index = {e: i for i, e in enumerate(entries)}
        c_index = {c: j for j, c in enumerate(candidates)}
        scores = np.full((len(entries), len(candidates)), -1e9)
        for row in group.itertuples(index=False):
            scores[e_index[str(row.citation_key)], c_index[str(row.candidate_ref_id)]] = row.score
        rows, cols = linear_sum_assignment(-scores)
        for i, j in zip(rows, cols):
            if scores[i, j] <= -1e8:
                continue
            key = (str(pub), entries[i])
            assigned = candidates[j]
            ranking = rankings.get(key, [])
            rankings[key] = [assigned] + [c for c in ranking if c != assigned]
    return rankings


# --------------------------------------------------------------------------- #
# Training, selection, evaluation
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentResults:
    selected_model: str
    selected_postprocessing: str
    comparison: pd.DataFrame
    test_metrics: Dict[str, Dict[str, float]]
    valid_metrics: Dict[str, Dict[str, float]]
    final_model: object
    selection_metric: str
    notes: List[str] = field(default_factory=list)


def _metric(metrics: Dict[str, Dict[str, float]], source: str, name: str) -> float:
    value = metrics.get(source, {}).get(name, float("nan"))
    return float(value) if value is not None else float("nan")


def score_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.score(frame) if len(frame) else []
    return scored


def select_and_evaluate(
    frame: pd.DataFrame,
    factories: Optional[Dict[str, Callable]] = None,
    postprocessing_modes: Sequence[str] = POSTPROCESSING_MODES,
    selection_metric: str = "mrr@5_matchable",
    seed: int = 42,
) -> ExperimentResults:
    """Train every model on ``train``, pick model + post-processing on ``valid``.

    Selection uses the manual part of the validation set when it has matchable
    entries (auto labels are optimistic), otherwise the whole validation set.
    The selected configuration is refit on train + valid and evaluated once on
    ``test``. Test scores of the other models are reported for analysis only.
    """
    factories = factories or default_model_factories(seed)
    labeled = frame[frame["is_labeled"]]
    train = labeled[labeled["partition"] == "train"]
    valid = labeled[labeled["partition"] == "valid"]
    test = labeled[labeled["partition"] == "test"]
    notes: List[str] = []

    manual_valid = valid[valid["label_source"] == "manual"]
    use_manual = bool(len(manual_valid)) and any(truth_by_entry(manual_valid).values())
    selection_source = "manual" if use_manual else "all"
    if not use_manual:
        notes.append("Validation has no matchable manual entry: selection uses all of valid.")

    rows = []
    for name, factory in factories.items():
        try:
            model = factory(None).fit(train)
        except ValueError as exc:
            notes.append(f"{name} skipped: {exc}")
            continue
        valid_scored = score_frame(model, valid)
        test_scored = score_frame(model, test)
        for mode in postprocessing_modes:
            v = evaluate_by_source(postprocess_rankings(valid_scored, mode), valid)
            t = evaluate_by_source(postprocess_rankings(test_scored, mode), test)
            row = {"model": name, "postprocessing": mode}
            for split_name, metrics in (("valid", v), ("test", t)):
                for source in ("all", "manual", "auto"):
                    for metric in ("mrr@5", "mrr@5_matchable", "hit@1_matchable"):
                        row[f"{split_name}_{source}_{metric}"] = _metric(metrics, source, metric)
            row["selection_score"] = _metric(v, selection_source, selection_metric)
            rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("No model could be trained: check that train has positive pairs.")
    ordered = comparison.assign(
        _score=comparison["selection_score"].fillna(-1.0),
        _tiebreak=comparison[f"valid_all_{selection_metric}"].fillna(-1.0),
    ).sort_values(["_score", "_tiebreak"], ascending=False)
    best = ordered.iloc[0]
    comparison = ordered.drop(columns=["_score", "_tiebreak"]).reset_index(drop=True)

    final_model = factories[best["model"]](None).fit(pd.concat([train, valid]))
    test_rankings = postprocess_rankings(score_frame(final_model, test), best["postprocessing"])
    valid_rankings = postprocess_rankings(score_frame(final_model, valid), best["postprocessing"])
    return ExperimentResults(
        selected_model=str(best["model"]),
        selected_postprocessing=str(best["postprocessing"]),
        comparison=comparison,
        test_metrics=evaluate_by_source(test_rankings, test),
        valid_metrics=evaluate_by_source(valid_rankings, valid),
        final_model=final_model,
        selection_metric=f"{selection_source}:{selection_metric}",
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Analysis for the report / poster
# --------------------------------------------------------------------------- #
def permutation_importance_mrr(
    model,
    frame: pd.DataFrame,
    postprocessing: str = "none",
    n_repeats: int = 3,
    seed: int = 42,
    metric: str = "mrr@5_matchable",
) -> pd.DataFrame:
    """Drop of MRR when one feature is shuffled across the rows of ``frame``."""
    features = list(getattr(model, "features", None) or [])
    labeled = frame[frame["is_labeled"]]
    if labeled.empty or not features:
        return pd.DataFrame(columns=["feature", "group", "mrr_drop_mean", "mrr_drop_std"])
    rng = np.random.default_rng(seed)

    def mrr(data: pd.DataFrame) -> float:
        rankings = postprocess_rankings(score_frame(model, data), postprocessing)
        return evaluate_rankings(rankings, truth_by_entry(data))[metric]

    baseline = mrr(labeled)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    rows = []
    for feature in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = labeled.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            drops.append(baseline - mrr(shuffled))
        rows.append({
            "feature": feature,
            "group": group_of.get(feature, "other"),
            "mrr_drop_mean": float(np.mean(drops)),
            "mrr_drop_std": float(np.std(drops)),
        })
    return pd.DataFrame(rows).sort_values("mrr_drop_mean", ascending=False).reset_index(drop=True)


def feature_group_ablation(
    frame: pd.DataFrame,
    model_name: str,
    factories: Optional[Dict[str, Callable]] = None,
    postprocessing: str = "none",
    eval_partition: str = "valid",
    metric: str = "mrr@5_matchable",
    seed: int = 42,
) -> pd.DataFrame:
    """MRR with every feature, without each group, and with the title group only."""
    factories = factories or default_model_factories(seed)
    labeled = frame[frame["is_labeled"]]
    train = labeled[labeled["partition"] == "train"]
    evaluation = labeled[labeled["partition"] == eval_partition]
    all_features = feature_columns(labeled)
    settings = [("all features", all_features)]
    settings += [(f"without {g}", [f for f in all_features if f not in fs])
                 for g, fs in FEATURE_GROUPS.items() if set(fs) & set(all_features)]
    settings.append(("title only", [f for f in all_features if f in FEATURE_GROUPS["title"]]))

    rows = []
    for label, features in settings:
        if not features:
            continue
        try:
            model = factories[model_name](features).fit(train)
        except ValueError as exc:
            logger.warning("Ablation %s skipped: %s", label, exc)
            continue
        if not hasattr(model, "features") or model.features != features:
            # the heuristic baseline ignores feature subsets
            continue
        rankings = postprocess_rankings(score_frame(model, evaluation), postprocessing)
        metrics = evaluate_by_source(rankings, evaluation)
        rows.append({
            "setting": label,
            "n_features": len(features),
            f"{eval_partition}_all_{metric}": _metric(metrics, "all", metric),
            f"{eval_partition}_manual_{metric}": _metric(metrics, "manual", metric),
            f"{eval_partition}_auto_{metric}": _metric(metrics, "auto", metric),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Predictions table
# --------------------------------------------------------------------------- #
PREDICTION_COLUMNS = ["arxiv_id", "partition", "citation_key", "groundtruth_arxiv_id"] + [
    f"cand_{i}" for i in range(1, TOP_K + 1)
]


def build_predictions_table(
    bibtex: pd.DataFrame,
    rankings: Dict[EntryKey, Sequence[str]],
    split: Dict[str, Sequence[str]],
    labels: pd.DataFrame,
    k: int = TOP_K,
) -> pd.DataFrame:
    """One row per BibTeX entry of every publication in train/valid/test.

    Entries without candidates get empty ``cand_*``; ``groundtruth_arxiv_id``
    is filled only for labeled entries with a true reference.
    """
    partition_of = {str(pub): part for part, pubs in split.items() for pub in pubs}
    truths = {
        (row.arxiv_id, row.citation_key): row.groundtruth_arxiv_id
        for row in normalize_labels(labels).itertuples(index=False)
    }
    entries = bibtex[["arxiv_id", "citation_key"]].astype(str).drop_duplicates()
    entries = entries[entries["arxiv_id"].isin(partition_of)]

    rows = []
    for pub, key in entries.itertuples(index=False):
        ranking = list(rankings.get((pub, key), []))[:k]
        ranking += [None] * (k - len(ranking))
        rows.append([pub, partition_of[pub], key, truths.get((pub, key))] + ranking)
    table = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    # nullable string columns: an all-empty column must stay null, not float NaN
    for column in ["groundtruth_arxiv_id"] + [f"cand_{i}" for i in range(1, k + 1)]:
        table[column] = table[column].astype(object).where(table[column].notna(), None)
    table["partition"] = pd.Categorical(table["partition"], categories=["train", "valid", "test"])
    return table.sort_values(["partition", "arxiv_id", "citation_key"]).reset_index(drop=True)


def json_ready(value):
    """Convert numpy / NaN values so that metrics can be written as JSON."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
