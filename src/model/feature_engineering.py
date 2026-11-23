"""Milestone 3 Phase 3: Feature Engineering for Reference Matching.

Constructs comprehensive feature vectors for candidate pairs (bibtex_entry, reference)
including string similarities, author overlap, year signals, identifier matches,
and hierarchy / document-context features from Milestone 2.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
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

        @staticmethod
        def token_set_ratio(s1: str, s2: str) -> float:
            s1_words = set(s1.split())
            s2_words = set(s2.split())
            intersection = " ".join(sorted(s1_words.intersection(s2_words)))
            diff1 = " ".join(sorted(s1_words - s2_words))
            diff2 = " ".join(sorted(s2_words - s1_words))
            base = intersection
            sim1 = difflib.SequenceMatcher(None, base, f"{base} {diff1}".strip()).ratio()
            sim2 = difflib.SequenceMatcher(None, base, f"{base} {diff2}".strip()).ratio()
            return max(sim1, sim2) * 100.0

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


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class FeatureExtractor:
    """Extracts features for (BibTeX entry, candidate reference) pairs."""

    FEATURE_NAMES = [
        # Title features
        "sim_title_exact",
        "sim_title_ratio",
        "sim_title_token_sort",
        "sim_title_token_set",
        "sim_title_jaccard_words",
        "sim_title_char_3gram",
        "feat_title_length_diff",
        "feat_title_word_count_diff",
        # Author features
        "sim_author_exact",
        "sim_author_jaccard",
        "feat_author_first_match",
        "feat_author_count_diff",
        "feat_author_missing",
        # Year features
        "feat_year_exact",
        "feat_year_diff",
        "feat_year_missing",
        # Identifier features
        "feat_id_exact_match",
        "feat_has_bib_arxiv_id",
        # Context / Hierarchy features
        "feat_citation_depth",
        "feat_citation_frequency",
        "feat_is_in_intro_or_related",
        # Group-relative features
        "feat_title_sim_rank",
        "feat_title_sim_diff_to_max",
        "feat_candidate_count",
    ]

    def __init__(self, hierarchy_summary: Optional[Dict[str, Dict[str, dict]]] = None) -> None:
        """Initialize feature extractor.

        Args:
            hierarchy_summary: Optional mapping of
                publication_id -> citation_key -> {depth, frequency, is_intro_or_related}
                extracted from Milestone 2 hierarchy.
        """
        self.hierarchy_summary = hierarchy_summary or {}

    def extract_pair_features(
        self,
        bib_row: pd.Series,
        ref_row: pd.Series,
        pub_id: str,
        citation_key: str,
    ) -> Dict[str, float]:
        """Compute base pairwise features for a single candidate pair."""
        # 1. Clean Title Features
        bib_title = str(bib_row.get("clean_title") or "")
        ref_title = str(ref_row.get("clean_title") or "")

        exact_title = 1.0 if bib_title and ref_title and bib_title == ref_title else 0.0
        ratio_title = fuzz.ratio(bib_title, ref_title) / 100.0 if (bib_title and ref_title) else 0.0
        token_sort_title = (
            fuzz.token_sort_ratio(bib_title, ref_title) / 100.0 if (bib_title and ref_title) else 0.0
        )
        token_set_title = (
            fuzz.token_set_ratio(bib_title, ref_title) / 100.0 if (bib_title and ref_title) else 0.0
        )

        words_bib = set(bib_title.split())
        words_ref = set(ref_title.split())
        jaccard_words = _jaccard_similarity(words_bib, words_ref)

        ngrams_bib = _char_ngrams(bib_title, n=3)
        ngrams_ref = _char_ngrams(ref_title, n=3)
        jaccard_3gram = _jaccard_similarity(ngrams_bib, ngrams_ref)

        len_diff = abs(len(bib_title) - len(ref_title))
        word_diff = abs(len(words_bib) - len(words_ref))

        # 2. Author Features
        bib_authors = list(bib_row.get("clean_authors_list") or [])
        ref_authors = list(ref_row.get("clean_authors_list") or [])
        author_missing = 1.0 if not bib_authors or not ref_authors else 0.0

        exact_author = 1.0 if (bib_authors and ref_authors and bib_authors == ref_authors) else 0.0
        author_jaccard = _jaccard_similarity(set(bib_authors), set(ref_authors))

        first_match = 0.0
        if bib_authors and ref_authors:
            s_bib = bib_authors[0].split("_")[-1]
            s_ref = ref_authors[0].split("_")[-1]
            if s_bib and s_ref and s_bib == s_ref:
                first_match = 1.0

        author_count_diff = abs(len(bib_authors) - len(ref_authors))

        # 3. Year Features
        bib_year = bib_row.get("clean_year")
        ref_year = ref_row.get("clean_year")
        year_missing = 1.0 if pd.isna(bib_year) or pd.isna(ref_year) else 0.0

        year_exact = 0.0
        year_diff = 10.0  # Default penalization for missing / large gap
        if not year_missing:
            try:
                diff = abs(int(bib_year) - int(ref_year))
                year_diff = min(float(diff), 20.0)
                year_exact = 1.0 if diff == 0 else 0.0
            except (ValueError, TypeError):
                year_missing = 1.0

        # 4. Identifier Features
        extracted_arxiv_id = normalize_arxiv_id(bib_row.get("extracted_arxiv_id"))
        ref_arxiv_id = normalize_arxiv_id(ref_row.get("ref_arxiv_id"))
        has_bib_id = 1.0 if extracted_arxiv_id else 0.0
        id_exact = 1.0 if extracted_arxiv_id and ref_arxiv_id and extracted_arxiv_id == ref_arxiv_id else 0.0

        # 5. Hierarchy Context Features
        pub_hierarchy = self.hierarchy_summary.get(pub_id, {})
        entry_hierarchy = pub_hierarchy.get(citation_key, {})
        citation_depth = float(entry_hierarchy.get("depth", 1.0))
        citation_frequency = float(entry_hierarchy.get("frequency", 1.0))
        is_intro_or_related = 1.0 if entry_hierarchy.get("is_intro_or_related", False) else 0.0

        return {
            "sim_title_exact": exact_title,
            "sim_title_ratio": ratio_title,
            "sim_title_token_sort": token_sort_title,
            "sim_title_token_set": token_set_title,
            "sim_title_jaccard_words": jaccard_words,
            "sim_title_char_3gram": jaccard_3gram,
            "feat_title_length_diff": float(len_diff),
            "feat_title_word_count_diff": float(word_diff),
            "sim_author_exact": exact_author,
            "sim_author_jaccard": author_jaccard,
            "feat_author_first_match": first_match,
            "feat_author_count_diff": float(author_count_diff),
            "feat_author_missing": author_missing,
            "feat_year_exact": year_exact,
            "feat_year_diff": year_diff,
            "feat_year_missing": year_missing,
            "feat_id_exact_match": id_exact,
            "feat_has_bib_arxiv_id": has_bib_id,
            "feat_citation_depth": citation_depth,
            "feat_citation_frequency": citation_frequency,
            "feat_is_in_intro_or_related": is_intro_or_related,
        }

    def create_candidate_matrix(
        self,
        bibtex_df: pd.DataFrame,
        references_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate full candidate feature matrix for every (bibtex, reference) pair.

        Args:
            bibtex_df: Cleaned or raw BibTeX table.
            references_df: Cleaned or raw references table.
            labels_df: Optional ground truth labels with columns:
                ['arxiv_id', 'citation_key', 'groundtruth_arxiv_id']

        Returns:
            DataFrame with metadata columns:
            ['arxiv_id', 'citation_key', 'candidate_ref_id'],
            all engineered feature columns,
            and 'label' (1 if matched, 0 otherwise) if labels_df is provided.
        """
        clean_bib = clean_bibtex_table(bibtex_df)
        clean_ref = clean_references_table(references_df)

        bib_col = "arxiv_id" if "arxiv_id" in clean_bib.columns else "citing_arxiv_id"
        ref_col = "citing_arxiv_id" if "citing_arxiv_id" in clean_ref.columns else "arxiv_id"

        # Build label lookup: (arxiv_id, citation_key) -> true_ref_arxiv_id
        ground_truth: Dict[Tuple[str, str], Optional[str]] = {}
        has_labels = labels_df is not None and not labels_df.empty
        if has_labels:
            for _, row in labels_df.iterrows():
                p_id = str(row.get("arxiv_id") or "")
                c_key = str(row.get("citation_key") or "")
                true_id = normalize_arxiv_id(row.get("groundtruth_arxiv_id"))
                ground_truth[(p_id, c_key)] = true_id

        rows: List[Dict[str, object]] = []

        all_pubs = sorted(clean_bib[bib_col].dropna().unique())

        for pub_id in all_pubs:
            pub_bib = clean_bib[clean_bib[bib_col] == pub_id]
            pub_ref = clean_ref[clean_ref[ref_col] == pub_id]

            if pub_bib.empty or pub_ref.empty:
                continue

            num_candidates = len(pub_ref)

            for _, bib_row in pub_bib.iterrows():
                citation_key = str(bib_row.get("citation_key") or bib_row.get("key") or "")

                # Per Milestone 3 spec: Unlabeled entries simply carry no rows
                if has_labels and (pub_id, citation_key) not in ground_truth:
                    continue

                true_ref_id = ground_truth.get((pub_id, citation_key)) if has_labels else None

                entry_pairs: List[Dict[str, object]] = []

                for _, ref_row in pub_ref.iterrows():
                    ref_id = str(ref_row.get("ref_arxiv_id") or "")
                    pair_feats = self.extract_pair_features(
                        bib_row=bib_row,
                        ref_row=ref_row,
                        pub_id=pub_id,
                        citation_key=citation_key,
                    )
                    pair_feats["feat_candidate_count"] = float(num_candidates)

                    record: Dict[str, object] = {
                        "arxiv_id": pub_id,
                        "citation_key": citation_key,
                        "candidate_ref_id": ref_id,
                        **pair_feats,
                    }

                    if labels_df is not None:
                        # 1 if this candidate is the true reference, 0 otherwise
                        is_positive = 1 if (true_ref_id and ref_id == true_ref_id) else 0
                        record["label"] = is_positive

                    entry_pairs.append(record)

                # Add group-relative features across all candidates for this BibTeX entry
                if entry_pairs:
                    # Sort candidates by title ratio descending to assign rank
                    entry_pairs.sort(key=lambda x: x["sim_title_ratio"], reverse=True)
                    max_title_sim = entry_pairs[0]["sim_title_ratio"]

                    for rank_idx, cand_record in enumerate(entry_pairs, start=1):
                        cand_record["feat_title_sim_rank"] = float(rank_idx)
                        cand_record["feat_title_sim_diff_to_max"] = float(
                            cand_record["sim_title_ratio"] - max_title_sim
                        )

                    rows.extend(entry_pairs)

        df = pd.DataFrame(rows)
        return df


def parse_hierarchy_for_features(elements_df: pd.DataFrame, hierarchy_df: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    """Helper to extract citation hierarchy depth, section, and frequency from Milestone 2 tables.

    Returns:
        Dict: publication_id -> citation_key -> {depth, frequency, is_intro_or_related}
    """
    summary: Dict[str, Dict[str, dict]] = {}
    if elements_df.empty or hierarchy_df.empty:
        return summary

    # Milestone 2 hierarchy node depth and parent mapping
    depth_map: Dict[str, int] = {}
    if "child" in hierarchy_df.columns and "parent" in hierarchy_df.columns:
        parent_map = dict(zip(hierarchy_df["child"], hierarchy_df["parent"]))
        for node in hierarchy_df["child"]:
            curr = node
            d = 1
            visited = set()
            while curr in parent_map and curr not in visited and d < 20:
                visited.add(curr)
                curr = parent_map[curr]
                d += 1
            depth_map[str(node)] = d

    # Inspect citations in elements
    if "element_type" in elements_df.columns or "type" in elements_df.columns:
        type_col = "element_type" if "element_type" in elements_df.columns else "type"
        id_col = "element_id" if "element_id" in elements_df.columns else "id"
        pub_col = "arxiv_id" if "arxiv_id" in elements_df.columns else "publication_id"
        content_col = "content" if "content" in elements_df.columns else "text"

        for _, elem in elements_df.iterrows():
            elem_type = str(elem.get(type_col) or "").lower()
            elem_content = str(elem.get(content_col) or "")
            pub = str(elem.get(pub_col) or "")
            elem_id = str(elem.get(id_col) or "")

            if "cite" in elem_type or "citation" in elem_type:
                # Key is usually in content
                c_key = elem_content.strip()
                if not c_key or not pub:
                    continue
                if pub not in summary:
                    summary[pub] = {}
                if c_key not in summary[pub]:
                    summary[pub][c_key] = {
                        "depth": depth_map.get(elem_id, 1),
                        "frequency": 1,
                        "is_intro_or_related": False,
                    }
                else:
                    summary[pub][c_key]["frequency"] += 1

    return summary


def main() -> None:
    """CLI runner to generate candidate feature matrix."""
    parser = argparse.ArgumentParser(description="Feature Engineering for Reference Matching")
    parser.add_argument("--bibtex-path", type=str, required=True, help="Path to bibtex table")
    parser.add_argument("--references-path", type=str, required=True, help="Path to references table")
    parser.add_argument("--labels-path", type=str, default=None, help="Path to ground truth labels table")
    parser.add_argument("--output-path", type=str, required=True, help="Output file path (parquet/csv)")
    args = parser.parse_args()

    bib_df = read_table(Path(args.bibtex_path))
    ref_df = read_table(Path(args.references_path))
    labels_df = read_table(Path(args.labels_path)) if args.labels_path else None

    extractor = FeatureExtractor()
    feature_matrix = extractor.create_candidate_matrix(bib_df, ref_df, labels_df)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        feature_matrix.to_parquet(out_path, index=False)
    else:
        feature_matrix.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Feature matrix generated: {len(feature_matrix)} candidate pairs across {feature_matrix['arxiv_id'].nunique()} publications.")
    if "label" in feature_matrix.columns:
        positives = int(feature_matrix["label"].sum())
        total = len(feature_matrix)
        print(f"Positive candidate pairs: {positives}/{total} ({positives/max(1, total)*100:.2f}%)")


if __name__ == "__main__":
    main()
