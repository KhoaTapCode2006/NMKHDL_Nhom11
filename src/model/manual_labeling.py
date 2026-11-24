"""Milestone 3 Phase 2 : Assisted Manual Labeling & Custom-Folder Crawler Helper.

This module provides:
1. crawl: Crawl an arXiv ID range or explicit ID list into any custom directory
   (e.g. `sourcesManual20`) using existing crawler functions
   without modifying `src/arxiv_crawler.py` or `src/oai_crawler.py`.
   Supports `--target-valid N` to automatically parse each crawled paper with Milestone 2's
   ParsingPipeline and stop as soon as N eligible manual-labeling papers
   (>= 20 BibTeX entries and >= 1 arXiv-keyed reference) are collected.
2. copy-eligible: Copy already-crawled & parsed eligible publications from an existing
   folder (such as `sourcesArxiv`) into `sourcesManual20`.
3. prepare: Run `src.model.data_cleaning` normalizers on the manual pool,
   score all within-paper `(BibTeX entry, crawled arXiv reference)` candidate pairs,
   and generate:
   - `manual_review_sheet.csv` (spreadsheet for human verification)
   - `references_lookup_by_paper.csv` (all valid arXiv references per paper)
   - `manual_review_ui.html` (standalone offline interactive HTML review tool)
4. finalize: Validate human-reviewed labels against all Milestone 3 requirements
   (per-paper floor Q >= 20, valid within-paper ref_arxiv_id, single association per entry)
   and export:
   - `manual_labels.csv` & `manual_labels.json` (entry-level ground truth)
   - `manual_candidate_pairs.csv` (m x n binary candidate matrix for Modeling)
   - `manual_labeling_stats.json` (summary statistics for Report & Poster)
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from src.model.data_cleaning import (
    clean_bibtex_table,
    clean_references_table,
    collect_bibtex_tables,
    flatten_references_json,
    normalize_arxiv_id,
)


STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "with",
}


def _tokenize_title(clean_title_str: object) -> List[str]:
    if not clean_title_str or pd.isna(clean_title_str):
        return []
    tokens = [
        tok
        for tok in str(clean_title_str).lower().split()
        if len(tok) > 1 and tok not in STOPWORDS
    ]
    return tokens


def _extract_surnames(authors_list: object) -> List[str]:
    if not isinstance(authors_list, (list, tuple)):
        return []
    surnames: List[str] = []
    for item in authors_list:
        if not item:
            continue
        s = str(item).strip().lower()
        surname = s.split("_")[-1] if "_" in s else s
        if surname and surname not in surnames:
            surnames.append(surname)
    return surnames


def compute_pair_similarity(
    bib_row: Mapping[str, Any],
    ref_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute interpretable matching signals between one BibTeX entry and one arXiv reference."""
    bib_extracted_id = normalize_arxiv_id(bib_row.get("extracted_arxiv_id"))
    ref_arxiv_id = normalize_arxiv_id(ref_row.get("ref_arxiv_id"))

    exact_id_match = bool(
        bib_extracted_id and ref_arxiv_id and bib_extracted_id == ref_arxiv_id
    )

    bib_title = str(bib_row.get("clean_title") or "").strip()
    ref_title = str(ref_row.get("clean_title") or "").strip()

    if bib_title and ref_title:
        seq_ratio = difflib.SequenceMatcher(None, bib_title, ref_title).ratio()
        bib_tokens = set(_tokenize_title(bib_title))
        ref_tokens = set(_tokenize_title(ref_title))
        if bib_tokens and ref_tokens:
            inter = len(bib_tokens & ref_tokens)
            union = len(bib_tokens | ref_tokens)
            jaccard = inter / union if union else 0.0
            containment = inter / min(len(bib_tokens), len(ref_tokens))
            title_score = 0.45 * seq_ratio + 0.35 * jaccard + 0.20 * containment
        else:
            title_score = seq_ratio
    else:
        title_score = 0.0

    bib_authors = bib_row.get("clean_authors_list")
    if not isinstance(bib_authors, list):
        bib_authors = []
    ref_authors = ref_row.get("clean_authors_list")
    if not isinstance(ref_authors, list):
        ref_authors = []

    bib_surnames = _extract_surnames(bib_authors)
    ref_surnames = _extract_surnames(ref_authors)

    if bib_surnames and ref_surnames:
        bib_set = set(bib_surnames)
        ref_set = set(ref_surnames)
        surname_overlap = len(bib_set & ref_set) / min(len(bib_set), len(ref_set))
        first_author_match = 1.0 if bib_surnames[0] == ref_surnames[0] else 0.0
        author_score = 0.65 * surname_overlap + 0.35 * first_author_match
    else:
        author_score = 0.0

    bib_year = bib_row.get("clean_year")
    ref_year = ref_row.get("clean_year")
    year_score = 0.0
    year_diff: Optional[int] = None
    try:
        if bib_year is not None and ref_year is not None and not pd.isna(bib_year) and not pd.isna(ref_year):
            year_diff = abs(int(bib_year) - int(ref_year))
            if year_diff == 0:
                year_score = 1.0
            elif year_diff == 1:
                year_score = 0.8
            elif year_diff == 2:
                year_score = 0.4
    except (TypeError, ValueError):
        year_score = 0.0

    if exact_id_match:
        total_score = 1.0
    else:
        # Weighted combination emphasizing title + authors + year consistency
        if bib_title and ref_title:
            total_score = 0.72 * title_score + 0.20 * author_score + 0.08 * year_score
        else:
            # When BibTeX entry lacks a parsed title (e.g. raw \bibitem with only author/year)
            total_score = 0.75 * author_score + 0.25 * year_score

    return {
        "exact_id_match": exact_id_match,
        "title_score": round(float(title_score), 4),
        "author_score": round(float(author_score), 4),
        "year_score": round(float(year_score), 4),
        "year_diff": year_diff,
        "total_score": round(float(total_score), 4),
    }


def inspect_publication_eligibility(pub_dir: Path) -> Dict[str, Any]:
    """Check whether a publication folder has >= 20 BibTeX entries and >= 1 arXiv reference."""
    pub_dir = Path(pub_dir)
    arxiv_id = normalize_arxiv_id(pub_dir.name) or pub_dir.name

    bib_json = pub_dir / "bibtex" / "bibtex.json"
    bib_csv = pub_dir / "bibtex" / "bibtex.csv"
    elem_bib_json = pub_dir / "elements" / "bibtex.json"

    bibtex_count = 0
    for candidate in (bib_json, elem_bib_json):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    bibtex_count = len(data)
                    break
            except Exception:
                pass
    if bibtex_count == 0 and bib_csv.is_file():
        try:
            bibtex_count = len(pd.read_csv(bib_csv))
        except Exception:
            bibtex_count = 0

    total_refs = 0
    arxiv_refs = 0
    seen_ref_ids: Set[str] = set()
    for ref_path in sorted(pub_dir.rglob("references.json")):
        try:
            refs = json.loads(ref_path.read_text(encoding="utf-8"))
            if isinstance(refs, list):
                total_refs = max(total_refs, len(refs))
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    ext = ref.get("externalIds") or {}
                    rid = normalize_arxiv_id(ext.get("ArXiv") if isinstance(ext, dict) else None)
                    if rid:
                        seen_ref_ids.add(rid)
        except Exception:
            continue
    arxiv_refs = len(seen_ref_ids)

    return {
        "arxiv_id": arxiv_id,
        "folder_name": pub_dir.name,
        "bibtex_count": bibtex_count,
        "total_refs": total_refs,
        "arxiv_refs_count": arxiv_refs,
        "is_eligible": 20 <= bibtex_count <= 250 and arxiv_refs >= 3,
    }


TEAM_MEMBER_RANGES: Dict[str, List[Tuple[str, int, int]]] = {
    "member1_2506": [("2506", 10788, 15787)],
    "member2_2505_2506": [("2505", 20666, 24878), ("2506", 1, 787)],
    "member3_2411": [("2411", 13799, 18798)],
    "member4_2407_2408": [("2407", 17839, 21794), ("2408", 1, 1044)],
}

MANUAL_POOL_20_IDS: Tuple[str, ...] = (
    "2506.10792",
    "2506.10793",
    "2506.10795",
    "2506.10798",
    "2506.10806",
    "2505.20674",
    "2505.20678",
    "2505.20679",
    "2505.20686",
    "2505.20690",
    "2411.13800",
    "2411.13807",
    "2411.13817",
    "2411.13820",
    "2411.13821",
    "2407.17850",
    "2407.17852",
    "2407.17855",
    "2407.17859",
    "2407.17860",
)


def expand_segments_to_ids(
    segments: Sequence[Tuple[str, int, int]],
    max_ids: int = 0,
) -> List[str]:
    """Expand [(yymm, start_seq, end_seq), ...] into base arXiv IDs."""
    ids: List[str] = []
    for yymm, start_seq, end_seq in segments:
        for seq in range(start_seq, end_seq + 1):
            ids.append(f"{yymm}.{seq:05d}")
            if max_ids > 0 and len(ids) >= max_ids:
                return ids
    return ids


def _count_arxiv_refs(references: Optional[List[Dict[str, Any]]]) -> int:
    if not isinstance(references, list):
        return 0
    seen: Set[str] = set()
    for ref in references:
        if not isinstance(ref, dict):
            continue
        ext = ref.get("externalIds") or {}
        rid = normalize_arxiv_id(ext.get("ArXiv") if isinstance(ext, dict) else None)
        if rid:
            seen.add(rid)
    return len(seen)


def crawl_custom_range(
    out_dir: Path,
    ids: List[str],
    parse_after_crawl: bool = False,
    target_valid_papers: int = 0,
    prune_ineligible: bool = False,
    member_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Crawl specified arXiv IDs into `out_dir` using `src.arxiv_crawler` without modifying it."""
    import arxiv
    from src import arxiv_crawler
    from src.parser.pipeline import ParsingPipeline

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    id_set = set(ids)

    valid_papers: List[Dict[str, Any]] = []
    # First check already existing valid papers in out_dir belonging to `ids` (for resume)
    if target_valid_papers > 0:
        for existing_pub in sorted(out_dir.iterdir()):
            if not existing_pub.is_dir() or existing_pub.name.startswith("."):
                continue
            if existing_pub.name not in id_set:
                continue
            info = inspect_publication_eligibility(existing_pub)
            if info["is_eligible"]:
                if member_tag:
                    info["member"] = member_tag
                valid_papers.append(info)
        if len(valid_papers) >= target_valid_papers:
            print(
                f"[crawl] Already have {len(valid_papers)}/{target_valid_papers} "
                f"eligible papers for {member_tag or 'range'} in {out_dir}."
            )
            return {"valid_papers": valid_papers[:target_valid_papers], "out_dir": str(out_dir)}

    batch_size = 25 if target_valid_papers > 0 else 50
    for start_idx in range(0, len(ids), batch_size):
        batch = ids[start_idx : start_idx + batch_size]
        search = arxiv.Search(id_list=batch)
        for paper in arxiv_crawler.client.results(search):
            latest = paper.entry_id.rsplit("/", 1)[-1]
            base_id, _, latest_version = latest.rpartition("v")
            if not base_id:
                base_id = latest
                latest_version = "1"
            paper_dir = out_dir / base_id.replace("/", "-")

            # Skip if already counted as valid
            if any(p["arxiv_id"] == base_id for p in valid_papers):
                continue

            print(f"[{base_id}] (Latest: v{latest_version}) -> {paper_dir}")

            ref_file = paper_dir / "v1" / "references.json"
            references = None
            if ref_file.exists():
                try:
                    references = json.loads(ref_file.read_text(encoding="utf-8"))
                except Exception:
                    references = None

            if references is None:
                references = arxiv_crawler.get_paper_references(base_id)

            # Fast pre-filter when hunting for eligible manual-labeling papers:
            # if Semantic Scholar returns < 3 arXiv references, skip downloading .tar.gz!
            arxiv_ref_cnt = _count_arxiv_refs(references)
            if target_valid_papers > 0 and prune_ineligible and arxiv_ref_cnt < 3:
                print(
                    f"  [fast-skip] {base_id} has only {arxiv_ref_cnt} arXiv refs "
                    f"(need >= 3 for rich manual labeling); skipping TeX download."
                )
                if paper_dir.exists():
                    shutil.rmtree(paper_dir, ignore_errors=True)
                continue

            all_versions_exist = True
            for v_num in range(1, int(latest_version) + 1):
                v_meta = paper_dir / f"v{v_num}" / "metadata.json"
                if not v_meta.exists():
                    all_versions_exist = False
                    break
                try:
                    saved = json.loads(v_meta.read_text(encoding="utf-8"))
                    if saved.get("source_status") not in {"downloaded", "no_tex_or_bib"}:
                        all_versions_exist = False
                        break
                except Exception:
                    all_versions_exist = False
                    break

            if not all_versions_exist:
                # Date of every version (one batched API call; none for single-version papers)
                version_dates = arxiv_crawler.fetch_version_dates({base_id: paper}).get(base_id)
                for version_number in range(1, int(latest_version) + 1):
                    arxiv_crawler.get_source(
                        f"{base_id}v{version_number}",
                        save_dir=out_dir,
                        paper=paper,
                        references=references,
                        version_dates=version_dates,
                    )
            else:
                print(f"  Skipped download for {base_id} (already downloaded)")

            if parse_after_crawl or target_valid_papers > 0:
                bib_json = paper_dir / "bibtex" / "bibtex.json"
                if not bib_json.is_file():
                    pipeline = ParsingPipeline(
                        source_dir=out_dir,
                        clean_redundant_tex=False,
                    )
                    try:
                        pipeline.process_publication(paper_dir)
                    except Exception as exc:
                        print(f"  [parse] Warning on {base_id}: {exc}")

                info = inspect_publication_eligibility(paper_dir)
                if member_tag:
                    info["member"] = member_tag
                print(
                    f"  [check] {base_id}: bibtex={info['bibtex_count']}, "
                    f"arxiv_refs={info['arxiv_refs_count']} -> eligible={info['is_eligible']}"
                )
                if info["is_eligible"]:
                    if not any(p["arxiv_id"] == info["arxiv_id"] for p in valid_papers):
                        valid_papers.append(info)
                    if target_valid_papers > 0 and len(valid_papers) >= target_valid_papers:
                        print(
                            f"[crawl] Reached target of {target_valid_papers} eligible papers for {member_tag or 'range'} "
                            f"(total BibTeX entries: {sum(p['bibtex_count'] for p in valid_papers)})."
                        )
                        return {"valid_papers": valid_papers, "out_dir": str(out_dir)}
                elif prune_ineligible and target_valid_papers > 0:
                    print(f"  [prune] Removing ineligible paper {paper_dir.name} from {out_dir.name}")
                    shutil.rmtree(paper_dir, ignore_errors=True)

    return {"valid_papers": valid_papers, "out_dir": str(out_dir)}


def crawl_team20_manual_pool(
    out_dir: Path,
    papers_per_member: int = 5,
    scan_window_per_member: int = 40,
    manifest_out: Path = Path("data/manual_labeling/manual_pool_manifest.json"),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Crawl & parse 5 eligible manual-labeling papers for each of the 4 team members (total 20)."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_selected: Dict[str, List[Dict[str, Any]]] = {}

    for idx, (member_key, segments) in enumerate(TEAM_MEMBER_RANGES.items()):
        print(f"\n============================================================")
        print(f"[team20] Collecting {papers_per_member} eligible papers for {member_key}...")
        print(f"============================================================")
        candidate_ids = expand_segments_to_ids(segments, max_ids=scan_window_per_member)
        if seed is not None:
            rng = random.Random(seed + idx)
            rng.shuffle(candidate_ids)
        res = crawl_custom_range(
            out_dir=out_dir,
            ids=candidate_ids,
            parse_after_crawl=True,
            target_valid_papers=papers_per_member,
            prune_ineligible=True,
            member_tag=member_key,
        )
        all_selected[member_key] = res["valid_papers"]

    manifest_out = Path(manifest_out).resolve()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    total_papers = sum(len(v) for v in all_selected.values())
    total_entries = sum(p["bibtex_count"] for v in all_selected.values() for p in v)
    manifest_data = {
        "total_papers": total_papers,
        "total_bibtex_entries": total_entries,
        "by_member": all_selected,
    }
    manifest_out.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[team20] Finished! Collected {total_papers} papers ({total_entries} BibTeX entries).")
    print(f"[team20] Manifest saved to: {manifest_out}")
    return manifest_data


def copy_eligible_papers(
    from_dir: Path,
    out_dir: Path,
    limit: int = 5,
    member_tag: Optional[str] = None,
    specific_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Copy top eligible publications (sorted by arXiv ref count and BibTeX count) to `out_dir`."""
    from_dir = Path(from_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    allowed_set = {normalize_arxiv_id(x) or x for x in specific_ids} if specific_ids else None

    for pub_dir in sorted(from_dir.iterdir()):
        if not pub_dir.is_dir() or pub_dir.name.startswith("."):
            continue
        info = inspect_publication_eligibility(pub_dir)
        if allowed_set is not None and info["arxiv_id"] not in allowed_set and pub_dir.name not in allowed_set:
            continue
        if info["is_eligible"]:
            if member_tag:
                info["member"] = member_tag
            candidates.append((pub_dir, info))

    # Prefer papers with rich arXiv reference pools and comfortable BibTeX entry counts (>= 25)
    candidates.sort(
        key=lambda item: (item[1]["arxiv_refs_count"], item[1]["bibtex_count"]),
        reverse=True,
    )
    selected = candidates[:limit] if limit > 0 else candidates

    copied_infos: List[Dict[str, Any]] = []
    for src_pub, info in selected:
        dst_pub = out_dir / src_pub.name
        if dst_pub.resolve() != src_pub.resolve():
            if dst_pub.exists():
                shutil.rmtree(dst_pub)
            shutil.copytree(src_pub, dst_pub)
        copied_infos.append(info)
        print(
            f"[copy-eligible] {src_pub.name} -> {dst_pub} "
            f"(bibtex={info['bibtex_count']}, arxiv_refs={info['arxiv_refs_count']})"
        )
    return copied_infos


def _build_html_reviewer(
    review_rows: List[Dict[str, Any]],
    refs_by_paper: Dict[str, List[Dict[str, Any]]],
    output_html_path: Path,
) -> None:
    """Create a standalone offline HTML interactive UI for fast human review of manual labels."""
    payload = {
        "rows": review_rows,
        "refs_by_paper": refs_by_paper,
    }
    json_payload = json.dumps(payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Milestone 3 — Assisted Manual Labeling Reviewer</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --card: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  header {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: #020617;
    border-bottom: 1px solid #334155;
    padding: 12px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }}
  select, input, button {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 7px 11px;
    font-size: 13px;
  }}
  button {{
    cursor: pointer;
    font-weight: 600;
  }}
  button.primary {{
    background: #0284c7;
    border-color: #38bdf8;
  }}
  button.success {{
    background: #15803d;
    border-color: #22c55e;
  }}
  .stats-bar {{
    font-size: 13px;
    color: var(--muted);
  }}
  .container {{
    max-width: 1400px;
    margin: 16px auto;
    padding: 0 16px 60px;
  }}
  .entry-card {{
    background: var(--panel);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    display: grid;
    grid-template-columns: 1fr 1.15fr;
    gap: 16px;
  }}
  .entry-card.status-AUTO_EXACT_ID {{ border-left: 5px solid var(--green); }}
  .entry-card.status-HIGH_CONF_MATCH {{ border-left: 5px solid var(--accent); }}
  .entry-card.status-NEEDS_HUMAN_CHECK {{ border-left: 5px solid var(--amber); }}
  .entry-card.status-LIKELY_NO_ARXIV_REF {{ border-left: 5px solid #64748b; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 6px;
  }}
  .badge.AUTO_EXACT_ID {{ background: rgba(34,197,94,0.2); color: #4ade80; }}
  .badge.HIGH_CONF_MATCH {{ background: rgba(56,189,248,0.2); color: #38bdf8; }}
  .badge.NEEDS_HUMAN_CHECK {{ background: rgba(245,158,11,0.25); color: #fbbf24; }}
  .badge.LIKELY_NO_ARXIV_REF {{ background: rgba(148,163,184,0.2); color: #cbd5e1; }}
  .title {{ font-weight: 600; font-size: 15px; margin: 6px 0; color: #ffffff; }}
  .meta {{ font-size: 12.5px; color: var(--muted); margin: 3px 0; }}
  .cand-list {{ display: flex; flex-direction: column; gap: 6px; }}
  .cand-item {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 8px 10px;
    cursor: pointer;
    transition: border-color 0.15s;
  }}
  .cand-item:hover {{ border-color: var(--accent); }}
  .cand-item.selected {{
    border: 2px solid var(--green);
    background: rgba(21, 128, 61, 0.18);
  }}
  .cand-header {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
  }}
  .cand-title {{ font-size: 13.5px; font-weight: 500; margin: 3px 0; }}
  .cand-meta {{ font-size: 11.5px; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <div>
    <strong>Milestone 3 — Manual Labeling Reviewer</strong>
    <div class="stats-bar" id="statsBar">Đang tải...</div>
  </div>
  <div class="controls">
    <select id="paperFilter"><option value="ALL">Tất cả bài báo</option></select>
    <select id="statusFilter">
      <option value="ALL">Tất cả trạng thái</option>
      <option value="NEEDS_HUMAN_CHECK">⚠️ Cần kiểm tra tay (NEEDS_HUMAN_CHECK)</option>
      <option value="HIGH_CONF_MATCH">🔹 Gợi ý độ tin cậy cao (HIGH_CONF_MATCH)</option>
      <option value="AUTO_EXACT_ID">✅ Khớp trực tiếp arXiv ID (AUTO_EXACT_ID)</option>
      <option value="LIKELY_NO_ARXIV_REF">⬜ Không có arXiv Ref khớp (LIKELY_NO_ARXIV_REF)</option>
    </select>
    <input type="text" id="searchInput" placeholder="Tìm key / tiêu đề / tác giả..." />
    <button class="success" onclick="exportCSV()">💾 Tải xuống manual_review_sheet.csv</button>
  </div>
</header>
<div class="container" id="listContainer"></div>

<script>
const DATA = {json_payload};
const STORAGE_KEY = "m3_manual_labels_v1";
let savedOverrides = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");

DATA.rows.forEach(r => {{
  const key = r.arxiv_id + "::" + r.citation_key;
  if (key in savedOverrides) {{
    r.groundtruth_arxiv_id = savedOverrides[key];
    r.human_verified = true;
    r.reviewed = "yes";
  }}
}});

const paperSelect = document.getElementById("paperFilter");
const papers = [...new Set(DATA.rows.map(r => r.arxiv_id))];
papers.forEach(p => {{
  const count = DATA.rows.filter(r => r.arxiv_id === p).length;
  const opt = document.createElement("option");
  opt.value = p;
  opt.textContent = `${{p}} (${{count}} entries)`;
  paperSelect.appendChild(opt);
}});

paperSelect.addEventListener("change", render);
document.getElementById("statusFilter").addEventListener("change", render);
document.getElementById("searchInput").addEventListener("input", render);

function setLabel(arxivId, citationKey, refArxivId) {{
  const key = arxivId + "::" + citationKey;
  savedOverrides[key] = refArxivId || "";
  localStorage.setItem(STORAGE_KEY, JSON.stringify(savedOverrides));
  const row = DATA.rows.find(r => r.arxiv_id === arxivId && r.citation_key === citationKey);
  if (row) {{
    row.groundtruth_arxiv_id = refArxivId || "";
    row.human_verified = true;
    row.reviewed = "yes";
  }}
  render();
}}

function updateStats() {{
  const total = DATA.rows.length;
  const matched = DATA.rows.filter(r => r.groundtruth_arxiv_id && r.groundtruth_arxiv_id.trim() !== "").length;
  const needsCheck = DATA.rows.filter(r => r.review_status === "NEEDS_HUMAN_CHECK").length;
  const reviewed = DATA.rows.filter(r => r.reviewed === "yes").length;
  document.getElementById("statsBar").textContent =
    `Tổng số bài: ${{papers.length}} | Tổng số BibTeX entries: ${{total}} | Đã duyệt: ${{reviewed}}/${{total}} | Đang gán khớp (Positive): ${{matched}} | Không khớp (None): ${{total - matched}} | Cần chú ý kiểm tra: ${{needsCheck}}`;
}}

function render() {{
  updateStats();
  const pFilter = paperSelect.value;
  const sFilter = document.getElementById("statusFilter").value;
  const q = document.getElementById("searchInput").value.trim().toLowerCase();

  const filtered = DATA.rows.filter(r => {{
    if (pFilter !== "ALL" && r.arxiv_id !== pFilter) return false;
    if (sFilter !== "ALL" && r.review_status !== sFilter) return false;
    if (q) {{
      const hay = `${{r.arxiv_id}} ${{r.citation_key}} ${{r.bib_title}} ${{r.bib_authors}} ${{r.cand1_title}} ${{r.groundtruth_arxiv_id}}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});

  const container = document.getElementById("listContainer");
  container.innerHTML = "";
  const shown = filtered.slice(0, 250);

  shown.forEach(r => {{
    const card = document.createElement("div");
    card.className = `entry-card status-${{r.review_status}}`;

    const left = document.createElement("div");
    left.innerHTML = `
      <div>
        <span class="badge ${{r.review_status}}">${{r.review_status}}</span>
        <strong>[${{r.arxiv_id}}]</strong> <code>${{r.citation_key}}</code>
        ${{(r.reviewed === "yes" || r.human_verified) ? '<span class="badge AUTO_EXACT_ID">✓ Đã duyệt</span>' : '<span class="badge NEEDS_HUMAN_CHECK">Chưa duyệt</span>'}}
      </div>
      <div class="title">${{r.bib_title || '<i>(Không có tiêu đề tách riêng)</i>'}}</div>
      <div class="meta">👤 <b>Authors:</b> ${{r.bib_authors || 'N/A'}} | 📅 <b>Year:</b> ${{r.bib_year || 'N/A'}}</div>
      <div class="meta">🔗 <b>Extracted arXiv ID:</b> ${{r.extracted_arxiv_id || 'None'}} | <b>Score #1:</b> ${{r.cand1_score}}</div>
      <div class="meta" style="margin-top:8px;">
        🎯 <b>Nhãn hiện tại (groundtruth_arxiv_id):</b>
        <input type="text" value="${{r.groundtruth_arxiv_id || ''}}"
          placeholder="Trống = Không khớp (None)"
          onchange="setLabel('${{r.arxiv_id}}', '${{r.citation_key}}', this.value.trim())"
          style="width:170px; margin-left:6px; font-weight:bold; color:#4ade80;" />
        <button onclick="setLabel('${{r.arxiv_id}}', '${{r.citation_key}}', '')" style="margin-left:4px;">Xóa (None)</button>
        <button onclick="setLabel('${{r.arxiv_id}}', '${{r.citation_key}}', '${{r.groundtruth_arxiv_id || ''}}')" style="margin-left:4px;">✓ Xác nhận nhãn này</button>
      </div>
    `;

    const right = document.createElement("div");
    right.className = "cand-list";
    const cands = [
      {{ rank: 1, id: r.cand1_arxiv_id, score: r.cand1_score, title: r.cand1_title, authors: r.cand1_authors, year: r.cand1_year }},
      {{ rank: 2, id: r.cand2_arxiv_id, score: r.cand2_score, title: r.cand2_title, authors: r.cand2_authors, year: r.cand2_year }},
      {{ rank: 3, id: r.cand3_arxiv_id, score: r.cand3_score, title: r.cand3_title, authors: r.cand3_authors, year: r.cand3_year }},
    ].filter(c => c.id);

    if (cands.length === 0) {{
      right.innerHTML = `<div class="meta">Bài báo này không có reference ứng viên nào khớp.</div>`;
    }} else {{
      cands.forEach(c => {{
        const item = document.createElement("div");
        const isSel = r.groundtruth_arxiv_id === c.id;
        item.className = "cand-item" + (isSel ? " selected" : "");
        item.onclick = () => setLabel(r.arxiv_id, r.citation_key, isSel ? "" : c.id);
        item.innerHTML = `
          <div class="cand-header">
            <span>#${{c.rank}} — arXiv:${{c.id}} ${{isSel ? '✅ [ĐANG CHỌN]' : '(Click để chọn)'}}</span>
            <span>Score: ${{c.score}}</span>
          </div>
          <div class="cand-title">${{c.title || ''}}</div>
          <div class="cand-meta">👤 ${{c.authors || ''}} | 📅 ${{c.year || ''}}</div>
        `;
        right.appendChild(item);
      }});
    }}

    // Dropdown to choose from ALL arXiv references of this paper
    const allRefs = DATA.refs_by_paper[r.arxiv_id] || [];
    const selBox = document.createElement("select");
    selBox.style.marginTop = "4px";
    selBox.innerHTML = `<option value="">-- Hoặc chọn từ toàn bộ ${{allRefs.length}} arXiv references của bài ${{r.arxiv_id}} --</option>`;
    allRefs.forEach(ref => {{
      const opt = document.createElement("option");
      opt.value = ref.ref_arxiv_id;
      opt.selected = (r.groundtruth_arxiv_id === ref.ref_arxiv_id);
      opt.textContent = `[${{ref.ref_arxiv_id}}] (${{ref.year || ''}}) ${{ref.title ? ref.title.slice(0, 85) : ''}}`;
      selBox.appendChild(opt);
    }});
    selBox.onchange = (e) => setLabel(r.arxiv_id, r.citation_key, e.target.value);
    right.appendChild(selBox);

    card.appendChild(left);
    card.appendChild(right);
    container.appendChild(card);
  }});
}}

function csvEscape(val) {{
  if (val === null || val === undefined) return "";
  const s = String(val);
  if (s.includes(",") || s.includes('"') || s.includes("\\n")) {{
    return '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}

function exportCSV() {{
  if (!DATA.rows.length) return;
  const cols = Object.keys(DATA.rows[0]);
  const lines = [cols.join(",")];
  DATA.rows.forEach(r => {{
    lines.push(cols.map(c => csvEscape(r[c])).join(","));
  }});
  const blob = new Blob(["\\uFEFF" + lines.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "manual_review_sheet.csv";
  a.click();
  URL.revokeObjectURL(url);
}}

render();
</script>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_content, encoding="utf-8")


def prepare_manual_review(
    source_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run data cleaning and produce the assisted manual labeling review sheet & HTML UI."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_bibtex = collect_bibtex_tables(source_dir)
    if raw_bibtex.empty:
        raise RuntimeError(f"No BibTeX tables found under {source_dir}")
    bibtex_df = clean_bibtex_table(raw_bibtex)
    bibtex_df["arxiv_id"] = bibtex_df["arxiv_id"].map(
        lambda x: normalize_arxiv_id(x) or str(x)
    )

    raw_refs = flatten_references_json(source_dir)
    if raw_refs.empty:
        raise RuntimeError(f"No valid references.json files found under {source_dir}")
    refs_df = clean_references_table(raw_refs)

    # Save cleaned tables for inspection
    bibtex_df.to_csv(output_dir / "bibtex_cleaned.csv", index=False, encoding="utf-8-sig")
    refs_df.to_csv(output_dir / "references_cleaned.csv", index=False, encoding="utf-8-sig")

    refs_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    for citing_id, group in refs_df.groupby("citing_arxiv_id"):
        refs_by_paper[str(citing_id)] = group.to_dict(orient="records")

    review_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {
        "AUTO_EXACT_ID": 0,
        "HIGH_CONF_MATCH": 0,
        "NEEDS_HUMAN_CHECK": 0,
        "LIKELY_NO_ARXIV_REF": 0,
    }

    for _, bib_row in bibtex_df.iterrows():
        paper_id = str(bib_row["arxiv_id"])
        citation_key = str(bib_row["citation_key"])
        paper_refs = refs_by_paper.get(paper_id, [])

        scored_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for ref_row in paper_refs:
            sim = compute_pair_similarity(bib_row, ref_row)
            scored_candidates.append((sim, ref_row))

        scored_candidates.sort(
            key=lambda item: (
                1 if item[0]["exact_id_match"] else 0,
                item[0]["total_score"],
                item[0]["title_score"],
                item[0]["author_score"],
            ),
            reverse=True,
        )

        top1 = scored_candidates[0] if len(scored_candidates) >= 1 else None
        top2 = scored_candidates[1] if len(scored_candidates) >= 2 else None
        top3 = scored_candidates[2] if len(scored_candidates) >= 3 else None

        groundtruth_id = ""
        suggested_id = ""
        review_status = "LIKELY_NO_ARXIV_REF"

        if top1 is not None:
            sim1, ref1 = top1
            ref1_id = str(ref1["ref_arxiv_id"])
            score1 = sim1["total_score"]
            title1 = sim1["title_score"]
            author1 = sim1["author_score"]

            if sim1["exact_id_match"]:
                review_status = "AUTO_EXACT_ID"
                groundtruth_id = ref1_id
                suggested_id = ref1_id
            elif (title1 >= 0.84 and (author1 >= 0.35 or score1 >= 0.78)) or score1 >= 0.85:
                review_status = "HIGH_CONF_MATCH"
                groundtruth_id = ref1_id
                suggested_id = ref1_id
            elif score1 >= 0.45 or title1 >= 0.52 or (not bib_row.get("clean_title") and author1 >= 0.65 and sim1["year_score"] >= 0.8):
                review_status = "NEEDS_HUMAN_CHECK"
                suggested_id = ref1_id
                # Pre-fill when confidence is fairly strong (>= 0.68), leave blank when ambiguous
                groundtruth_id = ref1_id if score1 >= 0.68 else ""
            else:
                review_status = "LIKELY_NO_ARXIV_REF"
                groundtruth_id = ""
                suggested_id = ref1_id if score1 >= 0.30 else ""

        status_counts[review_status] = status_counts.get(review_status, 0) + 1

        def _fmt_authors(val: object) -> str:
            if isinstance(val, list):
                return ", ".join(str(x) for x in val[:5])
            return str(val or "")

        row_dict: Dict[str, Any] = {
            "arxiv_id": paper_id,
            "citation_key": citation_key,
            "groundtruth_arxiv_id": groundtruth_id,
            "suggested_arxiv_id": suggested_id,
            "review_status": review_status,
            "extracted_arxiv_id": bib_row.get("extracted_arxiv_id") or "",
            "bib_title": html.unescape(str(bib_row.get("title") or "")),
            "bib_authors": _fmt_authors(bib_row.get("authors")),
            "bib_year": "" if pd.isna(bib_row.get("clean_year")) else int(bib_row.get("clean_year")),
            "cand1_arxiv_id": top1[1]["ref_arxiv_id"] if top1 else "",
            "cand1_score": top1[0]["total_score"] if top1 else 0.0,
            "cand1_title_score": top1[0]["title_score"] if top1 else 0.0,
            "cand1_author_score": top1[0]["author_score"] if top1 else 0.0,
            "cand1_title": str(top1[1].get("title") or "") if top1 else "",
            "cand1_authors": _fmt_authors(top1[1].get("authors")) if top1 else "",
            "cand1_year": "" if (not top1 or pd.isna(top1[1].get("clean_year"))) else int(top1[1].get("clean_year")),
            "cand2_arxiv_id": top2[1]["ref_arxiv_id"] if top2 else "",
            "cand2_score": top2[0]["total_score"] if top2 else 0.0,
            "cand2_title": str(top2[1].get("title") or "") if top2 else "",
            "cand2_authors": _fmt_authors(top2[1].get("authors")) if top2 else "",
            "cand2_year": "" if (not top2 or pd.isna(top2[1].get("clean_year"))) else int(top2[1].get("clean_year")),
            "cand3_arxiv_id": top3[1]["ref_arxiv_id"] if top3 else "",
            "cand3_score": top3[0]["total_score"] if top3 else 0.0,
            "cand3_title": str(top3[1].get("title") or "") if top3 else "",
            "cand3_authors": _fmt_authors(top3[1].get("authors")) if top3 else "",
            "cand3_year": "" if (not top3 or pd.isna(top3[1].get("clean_year"))) else int(top3[1].get("clean_year")),
        }
        review_rows.append(row_dict)

    # Resolve duplicate assignments of the same ref_arxiv_id inside the same paper
    by_paper_ref: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in review_rows:
        gt = str(row["groundtruth_arxiv_id"] or "").strip()
        if gt:
            by_paper_ref.setdefault((row["arxiv_id"], gt), []).append(row)
    for (_, _), dup_rows in by_paper_ref.items():
        if len(dup_rows) > 1:
            dup_rows.sort(key=lambda r: float(r["cand1_score"]), reverse=True)
            for secondary in dup_rows[1:]:
                secondary["groundtruth_arxiv_id"] = ""
                secondary["review_status"] = "NEEDS_HUMAN_CHECK"

    for row in review_rows:
        # The pre-filled value is only a suggestion. Keep it apart so finalize can
        # report how often the reviewer changed it, and start every row as not
        # reviewed: a person must confirm or correct each label.
        row["prefilled_arxiv_id"] = row["groundtruth_arxiv_id"]
        row["reviewed"] = ""
        row["reviewer_note"] = ""

    review_df = pd.DataFrame(review_rows)
    sheet_path = output_dir / "manual_review_sheet.csv"
    review_df.to_csv(sheet_path, index=False, encoding="utf-8-sig")

    lookup_path = output_dir / "references_lookup_by_paper.csv"
    refs_df[
        ["citing_arxiv_id", "ref_arxiv_id", "title", "authors", "year", "clean_title"]
    ].to_csv(lookup_path, index=False, encoding="utf-8-sig")

    html_path = output_dir / "manual_review_ui.html"
    _build_html_reviewer(review_rows, refs_by_paper, html_path)

    print(f"\n[prepare] Processed {bibtex_df['arxiv_id'].nunique()} publications, {len(review_df)} BibTeX entries.")
    for st, cnt in status_counts.items():
        print(f"  - {st}: {cnt}")
    print(f"[prepare] Review CSV : {sheet_path}")
    print(f"[prepare] Lookup CSV : {lookup_path}")
    print(f"[prepare] Web UI     : {html_path}")
    return {
        "publications_count": int(bibtex_df["arxiv_id"].nunique()),
        "entries_count": len(review_df),
        "status_counts": status_counts,
        "sheet_path": str(sheet_path),
        "html_path": str(html_path),
    }


def finalize_manual_labels(
    review_csv_path: Path,
    references_csv_path: Path,
    output_dir: Path,
    min_publications: int = 5,
    min_entries_per_pub: int = 20,
    min_total_entries: int = 100,
    allow_unreviewed: bool = False,
) -> Dict[str, Any]:
    """Validate reviewed manual labels and export entry-level labels + m x n binary candidate matrix.

    Every row must be marked ``reviewed`` (yes/y/1/true/x) by the person who
    checked it; otherwise the labels would only be the pre-filled heuristic
    suggestions. ``allow_unreviewed`` turns that error into a warning (for
    tests or drafts only, never for the submitted ground truth).
    """
    review_df = pd.read_csv(review_csv_path, dtype=str).fillna("")
    refs_df = pd.read_csv(references_csv_path, dtype=str).fillna("")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review_df["arxiv_id"] = review_df["arxiv_id"].map(lambda x: normalize_arxiv_id(x) or x.strip())
    review_df["groundtruth_arxiv_id"] = review_df["groundtruth_arxiv_id"].map(
        lambda x: normalize_arxiv_id(x) or "" if str(x).strip() else ""
    )
    refs_df["citing_arxiv_id"] = refs_df["citing_arxiv_id"].map(
        lambda x: normalize_arxiv_id(x) or x.strip()
    )
    refs_df["ref_arxiv_id"] = refs_df["ref_arxiv_id"].map(
        lambda x: normalize_arxiv_id(x) or x.strip()
    )

    valid_refs_by_paper: Dict[str, Set[str]] = {}
    for citing_id, group in refs_df.groupby("citing_arxiv_id"):
        valid_refs_by_paper[str(citing_id)] = set(group["ref_arxiv_id"].tolist())

    errors: List[str] = []
    warnings: List[str] = []

    pub_counts = review_df.groupby("arxiv_id")["citation_key"].count().to_dict()
    if len(pub_counts) < min_publications:
        warnings.append(
            f"Manual pool currently has {len(pub_counts)} publications "
            f"(target >= {min_publications})."
        )

    for pub_id, cnt in sorted(pub_counts.items()):
        if cnt < min_entries_per_pub:
            errors.append(
                f"Publication {pub_id} has only {cnt} labeled BibTeX entries (must be >= {min_entries_per_pub})."
            )

    if len(review_df) < min_total_entries:
        warnings.append(
            f"Total labeled entries is {len(review_df)} (target >= {min_total_entries})."
        )

    # Check every non-empty groundtruth_arxiv_id belongs to that publication's crawled references
    for idx, row in review_df.iterrows():
        pub_id = row["arxiv_id"]
        ckey = row["citation_key"]
        gt = row["groundtruth_arxiv_id"]
        if not gt:
            continue
        allowed = valid_refs_by_paper.get(pub_id, set())
        if gt not in allowed:
            errors.append(
                f"Row {idx} ({pub_id} :: {ckey}) has groundtruth_arxiv_id='{gt}' "
                f"which does NOT exist in crawled arXiv references of {pub_id}."
            )

    # Every row must have been checked by a person
    reviewed_values = {"yes", "y", "1", "true", "x", "co", "có"}
    if "reviewed" in review_df.columns:
        is_reviewed = review_df["reviewed"].str.strip().str.lower().isin(reviewed_values)
    else:
        is_reviewed = pd.Series(False, index=review_df.index)
    unreviewed = int((~is_reviewed).sum())
    if unreviewed:
        message = (
            f"{unreviewed}/{len(review_df)} rows are not marked reviewed=yes; "
            "check them in manual_review_ui.html (or the CSV) before finalizing."
        )
        (warnings if allow_unreviewed else errors).append(message)

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        raise ValueError(f"Manual label validation failed with {len(errors)} error(s).")

    # How much the reviewer changed the pre-filled suggestions
    if "prefilled_arxiv_id" in review_df.columns:
        prefilled = review_df["prefilled_arxiv_id"].map(
            lambda x: (normalize_arxiv_id(x) or "") if str(x).strip() else ""
        )
        changed = prefilled != review_df["groundtruth_arxiv_id"]
        status_col = review_df.get("review_status", pd.Series("", index=review_df.index))
        review_summary = {
            "reviewed_rows": int(is_reviewed.sum()),
            "suggestions_changed": int(changed.sum()),
            "suggestions_changed_pct": round(float(changed.mean()) * 100, 2) if len(review_df) else 0.0,
            "changed_by_status": {
                str(status): int(changed[status_col == status].sum())
                for status in sorted(status_col.unique())
            },
            "added_match": int((changed & (prefilled == "") & (review_df["groundtruth_arxiv_id"] != "")).sum()),
            "removed_match": int((changed & (prefilled != "") & (review_df["groundtruth_arxiv_id"] == "")).sum()),
        }
    else:
        review_summary = {"reviewed_rows": int(is_reviewed.sum())}

    for warn in warnings:
        print(f"[WARNING] {warn}")

    # Export 1: Entry-level ground truth table
    labels_df = pd.DataFrame({
        "arxiv_id": review_df["arxiv_id"],
        "citation_key": review_df["citation_key"],
        "groundtruth_arxiv_id": review_df["groundtruth_arxiv_id"].map(lambda x: x if x else None),
        "is_positive": review_df["groundtruth_arxiv_id"].map(lambda x: 1 if x else 0),
        "label_source": "manual",
    })
    labels_csv = output_dir / "manual_labels.csv"
    labels_json = output_dir / "manual_labels.json"
    labels_df.to_csv(labels_csv, index=False, encoding="utf-8-sig")
    labels_json.write_text(
        json.dumps(labels_df.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Export 2: Full m x n binary candidate matrix (Data Modeling section specification)
    pair_rows: List[Dict[str, Any]] = []
    for _, row in review_df.iterrows():
        pub_id = row["arxiv_id"]
        ckey = row["citation_key"]
        gt = row["groundtruth_arxiv_id"]
        for ref_id in sorted(valid_refs_by_paper.get(pub_id, set())):
            pair_rows.append({
                "arxiv_id": pub_id,
                "citation_key": ckey,
                "ref_arxiv_id": ref_id,
                "label": 1 if (gt and gt == ref_id) else 0,
                "label_source": "manual",
            })
    pairs_df = pd.DataFrame(pair_rows)
    pairs_csv = output_dir / "manual_candidate_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False, encoding="utf-8-sig")

    # Export 3: Summary statistics per publication for Report & Poster
    per_paper_stats: List[Dict[str, Any]] = []
    for pub_id, group in labels_df.groupby("arxiv_id"):
        m_entries = len(group)
        n_refs = len(valid_refs_by_paper.get(str(pub_id), set()))
        pos_entries = int(group["is_positive"].sum())
        per_paper_stats.append({
            "arxiv_id": str(pub_id),
            "labeled_entries_m": m_entries,
            "crawled_arxiv_refs_n": n_refs,
            "candidate_pairs_m_x_n": m_entries * n_refs,
            "positive_matches": pos_entries,
            "unmatched_entries": m_entries - pos_entries,
            "match_rate_pct": round(pos_entries / m_entries * 100, 2) if m_entries else 0.0,
        })

    stats_summary = {
        "total_manual_publications": len(per_paper_stats),
        "total_labeled_entries_Q": len(labels_df),
        "total_positive_matches": int(labels_df["is_positive"].sum()),
        "total_unmatched_entries": int(len(labels_df) - labels_df["is_positive"].sum()),
        "overall_match_rate_pct": round(
            float(labels_df["is_positive"].sum()) / len(labels_df) * 100, 2
        )
        if len(labels_df)
        else 0.0,
        "total_candidate_matrix_cells": len(pairs_df),
        "review": review_summary,
        "manual_publication_ids": [p["arxiv_id"] for p in per_paper_stats],
        "per_publication": per_paper_stats,
    }
    stats_path = output_dir / "manual_labeling_stats.json"
    stats_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n[finalize] SUCCESS! Exported {len(labels_df)} labeled entries across "
        f"{len(per_paper_stats)} publications ({stats_summary['total_positive_matches']} positives, "
        f"{stats_summary['total_unmatched_entries']} all-negative rows)."
    )
    print(f"[finalize] Entry labels CSV   : {labels_csv}")
    print(f"[finalize] Entry labels JSON  : {labels_json}")
    print(f"[finalize] Candidate matrix   : {pairs_csv} ({len(pairs_df)} pairs)")
    print(f"[finalize] Summary statistics : {stats_path}")
    return stats_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 3 Manual Labeling & Custom-Directory Crawler Helper."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. crawl
    p_crawl = sub.add_parser(
        "crawl",
        help="Crawl arXiv papers into a custom directory without modifying src/arxiv_crawler.py",
    )
    p_crawl.add_argument("--out", type=Path, required=True, help="Target directory (e.g. sourcesManual20)")
    p_crawl.add_argument(
        "--member-range",
        choices=tuple(TEAM_MEMBER_RANGES.keys()),
        default=None,
        help="Use one of the 4 official team member 5000-paper ID ranges directly",
    )
    p_crawl.add_argument("--year-month", type=str, default="2506", help="arXiv YYMM prefix (default: 2506)")
    p_crawl.add_argument("--start", type=int, default=None, help="Start sequence number (e.g. 10788)")
    p_crawl.add_argument("--end", type=int, default=None, help="End sequence number (e.g. 15787)")
    p_crawl.add_argument("--ids", nargs="*", default=None, help="Explicit arXiv IDs to crawl")
    p_crawl.add_argument("--max-ids", type=int, default=0, help="Optional cap on how many IDs from the range to scan")
    p_crawl.add_argument(
        "--parse",
        action="store_true",
        help="Run Milestone 2 ParsingPipeline immediately after downloading each paper",
    )
    p_crawl.add_argument(
        "--target-valid",
        type=int,
        default=0,
        help="Stop automatically once N eligible manual-labeling papers (>=20 BibTeX & >=1 arXiv ref) are collected",
    )
    p_crawl.add_argument(
        "--prune-ineligible",
        action="store_true",
        help="Remove papers that do not meet the >=20 BibTeX & >=1 arXiv ref floor when searching for --target-valid",
    )
    p_crawl.add_argument("--member", type=str, default=None, help="Optional member name tag")

    # 1b. crawl-team20
    p_t20 = sub.add_parser(
        "crawl-team20",
        help="Automatically crawl & parse 5 eligible manual-labeling papers for each of the 4 members (total 20) into sourcesManual20",
    )
    p_t20.add_argument("--out", type=Path, default=Path("sourcesManual20"))
    p_t20.add_argument("--papers-per-member", type=int, default=5)
    p_t20.add_argument("--scan-window", type=int, default=40)
    p_t20.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manual_labeling/manual_pool_manifest.json"),
    )
    p_t20.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to shuffle candidate IDs within the scan window",
    )

    # 2. copy-eligible
    p_copy = sub.add_parser(
        "copy-eligible",
        help="Copy already-crawled & parsed eligible papers (>=20 BibTeX, >=1 arXiv ref) to sourcesManual20",
    )
    p_copy.add_argument("--from-dir", type=Path, default=Path("sourcesArxiv"))
    p_copy.add_argument("--out", type=Path, default=Path("sourcesManual20"))
    p_copy.add_argument("--limit", type=int, default=5, help="Max papers to copy (0 = all eligible)")
    p_copy.add_argument("--member", type=str, default=None)
    p_copy.add_argument("--ids", nargs="*", default=None, help="Optional specific arXiv IDs to copy")

    # 3. prepare
    p_prep = sub.add_parser(
        "prepare",
        help="Clean tables and generate assisted manual review sheet + interactive HTML reviewer",
    )
    p_prep.add_argument("--source-dir", type=Path, default=Path("sourcesManual20"))
    p_prep.add_argument("--output-dir", type=Path, default=Path("data/manual_labeling"))

    # 4. finalize
    p_fin = sub.add_parser(
        "finalize",
        help="Validate reviewed manual labels and export official Ground Truth & candidate matrix",
    )
    p_fin.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/manual_labeling/manual_review_sheet.csv"),
    )
    p_fin.add_argument(
        "--references-csv",
        type=Path,
        default=Path("data/manual_labeling/references_cleaned.csv"),
    )
    p_fin.add_argument("--output-dir", type=Path, default=Path("data/manual_labeling"))
    p_fin.add_argument("--min-publications", type=int, default=5)
    p_fin.add_argument("--min-entries-per-pub", type=int, default=20)
    p_fin.add_argument("--min-total-entries", type=int, default=100)
    p_fin.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Only for drafts: accept rows not marked reviewed=yes (warning instead of error)",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "crawl":
        target_ids: List[str] = []
        if args.member_range:
            target_ids.extend(
                expand_segments_to_ids(
                    TEAM_MEMBER_RANGES[args.member_range],
                    max_ids=args.max_ids,
                )
            )
        if args.ids:
            target_ids.extend(args.ids)
        if args.start is not None and args.end is not None:
            target_ids.extend(
                f"{args.year_month}.{seq:05d}"
                for seq in range(args.start, args.end + 1)
            )
        if args.max_ids and args.max_ids > 0:
            target_ids = target_ids[: args.max_ids]
        if not target_ids:
            raise SystemExit(
                "Provide --member-range, --start/--end, or --ids for `crawl`."
            )
        crawl_custom_range(
            out_dir=args.out,
            ids=target_ids,
            parse_after_crawl=args.parse,
            target_valid_papers=args.target_valid,
            prune_ineligible=args.prune_ineligible,
            member_tag=args.member or args.member_range,
        )
        return 0

    if args.command == "crawl-team20":
        crawl_team20_manual_pool(
            out_dir=args.out,
            papers_per_member=args.papers_per_member,
            scan_window_per_member=args.scan_window,
            manifest_out=args.manifest_out,
            seed=args.seed,
        )
        return 0

    if args.command == "copy-eligible":
        copy_eligible_papers(
            from_dir=args.from_dir,
            out_dir=args.out,
            limit=args.limit,
            member_tag=args.member,
            specific_ids=args.ids,
        )
        return 0

    if args.command == "prepare":
        prepare_manual_review(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
        return 0

    if args.command == "finalize":
        finalize_manual_labels(
            review_csv_path=args.review_csv,
            references_csv_path=args.references_csv,
            output_dir=args.output_dir,
            min_publications=args.min_publications,
            min_entries_per_pub=args.min_entries_per_pub,
            min_total_entries=args.min_total_entries,
            allow_unreviewed=args.allow_unreviewed,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
