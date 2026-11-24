"""Milestone 3 Phase 2: Automatic ground-truth reference labeling.

Implements high-precision heuristics and string similarity rules to match
extracted BibTeX entries against crawled arXiv references for Milestone 3.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    # Fallback to difflib if rapidfuzz is not installed
    import difflib

    class _FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0

        @staticmethod
        def token_sort_ratio(s1: str, s2: str) -> float:
            t1 = " ".join(sorted(s1.split()))
            t2 = " ".join(sorted(s2.split()))
            return difflib.SequenceMatcher(None, t1, t2).ratio() * 100.0

    fuzz = _FuzzFallback()

from src.model.data_cleaning import (
    clean_authors,
    clean_bibtex_table,
    clean_references_table,
    clean_title,
    clean_year,
    extract_arxiv_id,
    normalize_arxiv_id,
    read_table,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Detailed matching result for a single BibTeX entry."""

    arxiv_id: str
    citation_key: str
    groundtruth_arxiv_id: Optional[str]
    matched: bool
    method: str
    confidence: float
    details: Dict[str, object] = field(default_factory=dict)


class AutoReferenceMatcher:
    """Heuristic and string similarity-based automatic matcher."""

    def __init__(
        self,
        min_title_sim_exact_meta: float = 88.0,
        min_title_sim_alone: float = 94.0,
        min_margin_alone: float = 12.0,
    ) -> None:
        """Initialize threshold parameters for high-precision labeling.

        Args:
            min_title_sim_exact_meta: Minimum title similarity if year or author matches.
            min_title_sim_alone: Minimum title similarity if metadata is missing or mismatching.
            min_margin_alone: Score difference between top 1 and top 2 to avoid ambiguity.
        """
        self.min_title_sim_exact_meta = min_title_sim_exact_meta
        self.min_title_sim_alone = min_title_sim_alone
        self.min_margin_alone = min_margin_alone

    @staticmethod
    def _compute_author_overlap(authors_a: Sequence[str], authors_b: Sequence[str]) -> float:
        """Compute Jaccard overlap on normalized author keys."""
        set_a = {a for a in authors_a if a}
        set_b = {b for b in authors_b if b}
        if not set_a or not set_b:
            return 0.0
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _first_author_match(authors_a: Sequence[str], authors_b: Sequence[str]) -> bool:
        """Check if the first authors have matching surname parts."""
        if not authors_a or not authors_b:
            return False
        first_a = authors_a[0].split("_")[-1]
        first_b = authors_b[0].split("_")[-1]
        return bool(first_a and first_b and first_a == first_b)

    def match_entry(
        self,
        bib_row: pd.Series,
        ref_candidates: pd.DataFrame,
    ) -> MatchResult:
        """Match one BibTeX entry against candidate references of the same publication."""
        arxiv_id = str(bib_row.get("arxiv_id") or bib_row.get("citing_arxiv_id") or "")
        citation_key = str(bib_row.get("citation_key") or bib_row.get("key") or "")
        extracted_arxiv_id = normalize_arxiv_id(bib_row.get("extracted_arxiv_id"))
        bib_title = str(bib_row.get("clean_title") or "")
        bib_year = bib_row.get("clean_year")
        bib_authors = list(bib_row.get("clean_authors_list") or [])

        # Default: No match found (all-negative row in candidate matrix)
        no_match = MatchResult(
            arxiv_id=arxiv_id,
            citation_key=citation_key,
            groundtruth_arxiv_id=None,
            matched=False,
            method="none",
            confidence=0.0,
        )

        if ref_candidates.empty:
            return no_match

        # -------------------------------------------------------------
        # Rule 1: Direct arXiv ID match (Regex / URL / Eprint extraction)
        # -------------------------------------------------------------
        if extracted_arxiv_id:
            exact_id_matches = ref_candidates[
                ref_candidates["ref_arxiv_id"] == extracted_arxiv_id
            ]
            if not exact_id_matches.empty:
                best_ref = exact_id_matches.iloc[0]
                return MatchResult(
                    arxiv_id=arxiv_id,
                    citation_key=citation_key,
                    groundtruth_arxiv_id=str(best_ref["ref_arxiv_id"]),
                    matched=True,
                    method="direct_arxiv_id",
                    confidence=1.0,
                    details={"matched_id": extracted_arxiv_id},
                )

        if not bib_title or len(bib_title) < 5:
            return no_match

        # -------------------------------------------------------------
        # Rule 2: Title & Author / Year heuristic matching
        # -------------------------------------------------------------
        candidate_scores: List[Tuple[float, str, dict]] = []

        for _, ref_row in ref_candidates.iterrows():
            ref_id = str(ref_row.get("ref_arxiv_id"))
            ref_title = str(ref_row.get("clean_title") or "")
            ref_year = ref_row.get("clean_year")
            ref_authors = list(ref_row.get("clean_authors_list") or [])

            if not ref_title:
                continue

            # Title similarities
            token_sim = fuzz.token_sort_ratio(bib_title, ref_title)
            direct_sim = fuzz.ratio(bib_title, ref_title)
            title_score = max(token_sim, direct_sim)

            # Year compatibility
            year_match = False
            year_diff = 999
            if pd.notna(bib_year) and pd.notna(ref_year):
                try:
                    year_diff = abs(int(bib_year) - int(ref_year))
                    year_match = year_diff <= 1
                except (ValueError, TypeError):
                    pass

            # Author overlap
            author_overlap = self._compute_author_overlap(bib_authors, ref_authors)
            first_auth_match = self._first_author_match(bib_authors, ref_authors)

            candidate_scores.append(
                (
                    title_score,
                    ref_id,
                    {
                        "token_sim": token_sim,
                        "direct_sim": direct_sim,
                        "year_match": year_match,
                        "year_diff": year_diff,
                        "author_overlap": author_overlap,
                        "first_auth_match": first_auth_match,
                    },
                )
            )

        if not candidate_scores:
            return no_match

        # Sort candidate scores descending by title similarity
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_ref_id, top_info = candidate_scores[0]
        second_score = candidate_scores[1][0] if len(candidate_scores) > 1 else 0.0
        margin = top_score - second_score

        # Rule 2a: Exact normalized title match + at least 1 author or year match
        if top_score >= 98.0:
            if top_info["author_overlap"] > 0 or top_info["first_auth_match"] or top_info["year_match"] or margin >= 10:
                return MatchResult(
                    arxiv_id=arxiv_id,
                    citation_key=citation_key,
                    groundtruth_arxiv_id=top_ref_id,
                    matched=True,
                    method="exact_title_verified",
                    confidence=0.98,
                    details=top_info,
                )

        # Rule 2b: High title similarity with author/year validation
        if top_score >= self.min_title_sim_exact_meta:
            has_strong_meta = (
                (top_info["author_overlap"] >= 0.20 or top_info["first_auth_match"])
                and top_info["year_match"]
            )
            has_moderate_meta = (
                top_info["author_overlap"] > 0 or top_info["first_auth_match"]
            ) and margin >= 5.0

            if has_strong_meta or has_moderate_meta:
                return MatchResult(
                    arxiv_id=arxiv_id,
                    citation_key=citation_key,
                    groundtruth_arxiv_id=top_ref_id,
                    matched=True,
                    method="high_title_and_metadata",
                    confidence=round(top_score / 100.0, 3),
                    details=top_info,
                )

        # Rule 2c: Very high title similarity alone with large margin over 2nd best
        if top_score >= self.min_title_sim_alone and margin >= self.min_margin_alone:
            return MatchResult(
                arxiv_id=arxiv_id,
                citation_key=citation_key,
                groundtruth_arxiv_id=top_ref_id,
                matched=True,
                method="unambiguous_title_match",
                confidence=round(top_score / 100.0, 3),
                details=top_info,
            )

        return no_match

    def label_publication(
        self,
        bibtex_df: pd.DataFrame,
        references_df: pd.DataFrame,
    ) -> List[MatchResult]:
        """Label every BibTeX entry of a given publication."""
        results: List[MatchResult] = []
        for _, bib_row in bibtex_df.iterrows():
            result = self.match_entry(bib_row, references_df)
            results.append(result)
        return results


def select_auto_label_publications(
    available_publications: Sequence[str],
    manual_labeled_publications: Iterable[str],
    ratio: float = 0.10,
    num_members: int = 4,
    seed: int = 42,
    groups: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Select at least max(ceil(10%), 2*k) publications outside the manual pool.

    The publications are drawn at random (reproducible through ``seed``), not
    taken from the start of the sorted list, which would favour the beginning of
    one member's ID range. When ``groups`` maps each publication to its member,
    members are visited in turn so that every member's range is represented.

    Args:
        available_publications: All publication IDs available in the dataset.
        manual_labeled_publications: Set of publication IDs used in manual labeling.
        ratio: Minimum fraction of remaining publications to label (default 0.10).
        num_members: Number of group members (default 4, needing 2*k = 8 pubs min).
        seed: Random seed of the draw.
        groups: Optional ``{publication_id: member}`` for a balanced draw.

    Returns:
        Sorted list of publication IDs selected for automatic labeling.
    """
    import math
    import random

    manual_set = set(manual_labeled_publications)
    remaining = sorted({pid for pid in available_publications if pid not in manual_set})

    # "at least 10 percent": round up, never down
    min_required = max(math.ceil(len(remaining) * ratio), 2 * num_members)
    selected_count = min(len(remaining), min_required)

    rng = random.Random(seed)
    if not groups:
        return sorted(rng.sample(remaining, selected_count))

    buckets: Dict[str, List[str]] = {}
    for pid in remaining:
        buckets.setdefault(groups.get(pid, "_unassigned"), []).append(pid)
    for members in buckets.values():
        rng.shuffle(members)
    order = sorted(buckets)
    selected: List[str] = []
    while len(selected) < selected_count:
        progressed = False
        for member in order:
            if buckets[member] and len(selected) < selected_count:
                selected.append(buckets[member].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def split_publications_train_val_test(
    manual_publications: Sequence[str],
    auto_publications: Sequence[str],
    num_members: int = 4,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition labeled publications into train, valid, and test sets per Milestone 3 spec.

    Spec:
    - Test set: k manual + k auto publications.
    - Validation set: k manual + k auto publications.
    - Training set: all remaining manual + auto publications.

    Args:
        manual_publications: List of manually labeled publication IDs (min 5*k).
        auto_publications: List of automatically labeled publication IDs (min 2*k).
        num_members: Number of members in group (k).
        seed: Random seed for deterministic partition.

    Returns:
        Dict with keys 'train', 'valid', 'test' mapping to list of publication IDs.
    """
    import random

    rng = random.Random(seed)
    m_list = list(manual_publications)
    a_list = list(auto_publications)
    rng.shuffle(m_list)
    rng.shuffle(a_list)

    k = num_members
    if len(m_list) < 2 * k:
        raise ValueError(f"Manual publications ({len(m_list)}) must be at least 2*k ({2*k}) for val and test splits.")
    if len(a_list) < 2 * k:
        raise ValueError(f"Auto publications ({len(a_list)}) must be at least 2*k ({2*k}) for val and test splits.")

    test_pubs = m_list[:k] + a_list[:k]
    valid_pubs = m_list[k : 2 * k] + a_list[k : 2 * k]
    train_pubs = m_list[2 * k :] + a_list[2 * k :]

    return {
        "train": sorted(train_pubs),
        "valid": sorted(valid_pubs),
        "test": sorted(test_pubs),
    }


def run_auto_labeling_pipeline(
    bibtex_df: pd.DataFrame,
    references_df: pd.DataFrame,
    target_publication_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Execute automatic labeling over target publications.

    Returns:
        DataFrame with columns:
        ['arxiv_id', 'citation_key', 'groundtruth_arxiv_id', 'matched', 'method', 'confidence']
    """
    clean_bib = clean_bibtex_table(bibtex_df)
    clean_ref = clean_references_table(references_df)

    matcher = AutoReferenceMatcher()

    # Determine publications to process
    bib_col = "arxiv_id" if "arxiv_id" in clean_bib.columns else "citing_arxiv_id"
    ref_col = "citing_arxiv_id" if "citing_arxiv_id" in clean_ref.columns else "arxiv_id"

    if target_publication_ids is None:
        target_pubs = sorted(clean_bib[bib_col].dropna().unique())
    else:
        target_pubs = list(target_publication_ids)

    all_results: List[MatchResult] = []

    for pub_id in target_pubs:
        pub_bib = clean_bib[clean_bib[bib_col] == pub_id]
        pub_ref = clean_ref[clean_ref[ref_col] == pub_id]

        if pub_bib.empty:
            continue

        pub_results = matcher.label_publication(pub_bib, pub_ref)
        all_results.extend(pub_results)

    result_df = pd.DataFrame(
        [
            {
                "arxiv_id": r.arxiv_id,
                "citation_key": r.citation_key,
                "groundtruth_arxiv_id": r.groundtruth_arxiv_id,
                "matched": r.matched,
                "method": r.method,
                "confidence": r.confidence,
                "label_source": "auto",
            }
            for r in all_results
        ],
        columns=[
            "arxiv_id", "citation_key", "groundtruth_arxiv_id",
            "matched", "method", "confidence", "label_source",
        ],
    )

    return result_df


def validate_auto_labels(auto_labels: pd.DataFrame, manual_labels: pd.DataFrame) -> Dict[str, object]:
    """Compare automatic labels with manual labels on the same entries.

    Run the auto labeler on the manually labeled publications and pass both
    tables here. Leakage note: the auto labels come from the same title/author
    similarities that later serve as model features, so a model evaluated on
    auto-labeled publications looks better than it is; these numbers tell how
    far the auto labels can be trusted.

    Returns precision (correct among auto matches), recall (manual matches the
    auto labeler recovered), agreement on "no match", and the entry count.
    """
    def norm(value: object) -> Optional[str]:
        return normalize_arxiv_id(value) if value is not None and not pd.isna(value) else None

    auto = auto_labels[["arxiv_id", "citation_key", "groundtruth_arxiv_id"]].copy()
    manual = manual_labels[["arxiv_id", "citation_key", "groundtruth_arxiv_id"]].copy()
    for frame in (auto, manual):
        frame["arxiv_id"] = frame["arxiv_id"].astype(str)
        frame["citation_key"] = frame["citation_key"].astype(str)
        frame["groundtruth_arxiv_id"] = frame["groundtruth_arxiv_id"].map(norm)
    merged = manual.merge(auto, on=["arxiv_id", "citation_key"], suffixes=("_manual", "_auto"))

    auto_pos = merged["groundtruth_arxiv_id_auto"].notna()
    manual_pos = merged["groundtruth_arxiv_id_manual"].notna()
    same = merged["groundtruth_arxiv_id_auto"] == merged["groundtruth_arxiv_id_manual"]
    correct_pos = int((auto_pos & same).sum())
    both_neg = int((~auto_pos & ~manual_pos).sum())
    return {
        "entries_compared": int(len(merged)),
        "auto_matches": int(auto_pos.sum()),
        "manual_matches": int(manual_pos.sum()),
        "precision": round(correct_pos / auto_pos.sum(), 4) if auto_pos.sum() else None,
        "recall": round(correct_pos / manual_pos.sum(), 4) if manual_pos.sum() else None,
        "no_match_agreement": round(both_neg / (~manual_pos).sum(), 4) if (~manual_pos).sum() else None,
        "accuracy": round(float((same | (~auto_pos & ~manual_pos)).mean()), 4) if len(merged) else None,
    }


def main() -> None:
    """CLI runner for automatic labeling."""
    parser = argparse.ArgumentParser(description="Auto Reference Labeler for Milestone 3")
    parser.add_argument("--bibtex-path", type=str, required=True, help="Path to bibtex table (csv/json/parquet)")
    parser.add_argument("--references-path", type=str, required=True, help="Path to references table (csv/json/parquet)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save auto-labeled ground truth")
    args = parser.parse_args()

    bib_df = read_table(Path(args.bibtex_path))
    ref_df = read_table(Path(args.references_path))

    labels_df = run_auto_labeling_pipeline(bib_df, ref_df)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        labels_df.to_parquet(output_path, index=False)
    else:
        labels_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    matched_count = labels_df["matched"].sum()
    total_count = len(labels_df)
    print(f"Auto-labeling completed: {matched_count}/{total_count} matched ({matched_count/max(1, total_count)*100:.2f}%)")


if __name__ == "__main__":
    main()
