"""End-to-end Milestone 3 pipeline: cleaning -> labels -> features -> model -> predictions.

Inputs
------
``--source-dir``: the publication folders produced by Milestones 1 and 2::

    <source-dir>/<arxiv_id>/v<N>/references.json   (Semantic Scholar references)
    <source-dir>/<arxiv_id>/bibtex/bibtex.json     (unified BibTeX entries)
    <source-dir>/<arxiv_id>/elements/              (elements + hierarchy, optional)

``--manual-labels``: the manual ground truth (``arxiv_id, citation_key,
groundtruth_arxiv_id``), e.g. ``data/manual_labeling/manual_labels.csv``.

Steps
-----
1. clean the BibTeX and reference tables (``data_cleaning``);
2. select the automatic pool, at least max(ceil(10% of the non-manual
   publications), 2k), drawn at random and balanced across members, and label
   it (``auto_labeler``); the auto labeler is also run on the manual pool and
   compared with the manual labels;
3. split: k manual + k auto publications for test and for valid, the rest train;
4. build the candidate matrix of every BibTeX entry of those publications
   (``feature_engineering``, with citation-context features from elements/);
5. train, select on valid, evaluate on test (``modeling``);
6. write the ``predictions`` table and every number used by the report.

Usage::

    python -m src.model.run_matching \
        --source-dir src/sourcesArxiv \
        --manual-labels data/manual_labeling/manual_labels.csv \
        --members-file members.json --num-members 4 \
        --output-dir data/matching
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.model.auto_labeler import (
    run_auto_labeling_pipeline,
    select_auto_label_publications,
    split_publications_train_val_test,
    validate_auto_labels,
)
from src.model.data_cleaning import (
    build_references_table,
    clean_references_table,
    collect_bibtex_tables,
    flatten_references_json,
    normalize_arxiv_id,
    read_table,
)
from src.model.feature_engineering import FeatureExtractor, load_hierarchy_summary
from src.model.modeling import (
    KEY,
    assign_partitions,
    attach_labels,
    build_predictions_table,
    feature_group_ablation,
    json_ready,
    normalize_labels,
    permutation_importance_mrr,
    postprocess_rankings,
    score_frame,
    select_and_evaluate,
)

logger = logging.getLogger(__name__)


def load_member_groups(members_file: Optional[Path]) -> Optional[Dict[str, str]]:
    """``{arxiv_id: member}`` from members.json, or None."""
    if not members_file:
        return None
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import crawler_common  # noqa: E402

    return crawler_common.member_of(Path(members_file))


def publication_key(value: object) -> str:
    """Publication IDs as stored in the tables (versionless, dots kept)."""
    text = str(value).strip()
    return normalize_arxiv_id(text) or text


def run(
    source_dir: Path,
    manual_labels_path: Path,
    output_dir: Path,
    num_members: int = 4,
    members_file: Optional[Path] = None,
    auto_ratio: float = 0.10,
    seed: int = 42,
    run_ablation: bool = True,
) -> Dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tables ------------------------------------------------------------
    bibtex = collect_bibtex_tables(source_dir)
    if bibtex.empty:
        raise SystemExit(f"No bibtex/bibtex.json found under {source_dir}")
    bibtex["arxiv_id"] = bibtex["arxiv_id"].map(publication_key)
    references = clean_references_table(flatten_references_json(source_dir))
    build_references_table(references).to_parquet(output_dir / "references_table.parquet", index=False)

    manual = normalize_labels(read_table(manual_labels_path), default_source="manual")
    manual["arxiv_id"] = manual["arxiv_id"].map(publication_key)
    manual["label_source"] = "manual"
    manual_pubs = sorted(manual["arxiv_id"].unique())

    # 2. Automatic labels --------------------------------------------------
    all_pubs = sorted(bibtex["arxiv_id"].unique())
    with_refs = set(references["citing_arxiv_id"].dropna().astype(str))
    non_manual = [p for p in all_pubs if p not in set(manual_pubs)]
    usable = [p for p in non_manual if p in with_refs]
    required = max(math.ceil(len(non_manual) * auto_ratio), 2 * num_members)
    # Draw from publications that have candidate references first: an auto
    # publication without any arXiv-keyed reference has nothing to rank.
    ratio = min(1.0, required / max(1, len(usable)))
    auto_pubs = select_auto_label_publications(
        usable, manual_pubs, ratio=ratio, num_members=num_members, seed=seed,
        groups=load_member_groups(members_file),
    )
    auto = run_auto_labeling_pipeline(bibtex, references, target_publication_ids=auto_pubs)
    auto_on_manual = run_auto_labeling_pipeline(bibtex, references, target_publication_ids=manual_pubs)
    auto_validation = validate_auto_labels(auto_on_manual, manual) if len(auto_on_manual) else {}
    auto_methods = auto["method"].value_counts().to_dict() if len(auto) else {}

    labels = pd.concat(
        [manual, normalize_labels(auto, default_source="auto")], ignore_index=True
    ).drop_duplicates(subset=KEY, keep="first")
    labels.to_csv(output_dir / "labels.csv", index=False, encoding="utf-8")

    # 3. Split ---------------------------------------------------------------
    split = split_publications_train_val_test(manual_pubs, auto_pubs, num_members=num_members, seed=seed)
    split_pubs = [p for part in split.values() for p in part]
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    # 4. Candidate matrix (every entry of the split publications) -----------
    hierarchy = load_hierarchy_summary(source_dir, split_pubs)
    extractor = FeatureExtractor(hierarchy_summary=hierarchy)
    split_bibtex = bibtex[bibtex["arxiv_id"].isin(split_pubs)]
    split_refs = references[references["citing_arxiv_id"].isin(split_pubs)]
    matrix = extractor.create_candidate_matrix(split_bibtex, split_refs, labels_df=None)
    matrix = assign_partitions(attach_labels(matrix, labels), split)
    matrix.to_parquet(output_dir / "candidate_matrix.parquet", index=False)

    # 5. Modeling --------------------------------------------------------------
    results = select_and_evaluate(matrix, seed=seed)
    results.comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    labeled = matrix[matrix["is_labeled"]]
    importance = permutation_importance_mrr(
        results.final_model, labeled[labeled["partition"] == "test"],
        postprocessing=results.selected_postprocessing, seed=seed,
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    if run_ablation and results.selected_model != "heuristic_title":
        feature_group_ablation(
            matrix, results.selected_model, postprocessing=results.selected_postprocessing, seed=seed,
        ).to_csv(output_dir / "ablation.csv", index=False)

    # 6. Predictions -------------------------------------------------------------
    rankings = postprocess_rankings(score_frame(results.final_model, matrix), results.selected_postprocessing)
    predictions = build_predictions_table(split_bibtex, rankings, split, labels)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")

    summary = {
        "publications_in_corpus": len(all_pubs),
        "manual_publications": len(manual_pubs),
        "manual_entries": int(len(manual)),
        "auto_publications": len(auto_pubs),
        "auto_required_min": required,
        "auto_entries": int(len(auto)),
        "auto_methods": auto_methods,
        "auto_label_validation_on_manual_pool": auto_validation,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "candidate_pairs": int(len(matrix)),
        "labeled_pairs": int(matrix["is_labeled"].sum()),
        "selected_model": results.selected_model,
        "selected_postprocessing": results.selected_postprocessing,
        "selection_metric": results.selection_metric,
        "valid_metrics": results.valid_metrics,
        "test_metrics": results.test_metrics,
        "predictions_rows": int(len(predictions)),
        "notes": results.notes,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Milestone 3 reference matching pipeline")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/matching"))
    parser.add_argument("--num-members", type=int, default=4)
    parser.add_argument("--members-file", type=Path, default=None,
                        help="members.json, to balance the automatic pool across members")
    parser.add_argument("--auto-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = run(
        args.source_dir, args.manual_labels, args.output_dir, args.num_members,
        args.members_file, args.auto_ratio, args.seed, not args.no_ablation,
    )
    test = summary["test_metrics"]
    print(f"Selected: {summary['selected_model']} + {summary['selected_postprocessing']}")
    for source, metrics in test.items():
        print(
            f"  test[{source}] MRR@5={metrics.get('mrr@5')} "
            f"MRR@5(matchable)={metrics.get('mrr@5_matchable')} entries={metrics.get('entries')}"
        )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
