"""arXiv crawler: discovery through the arXiv API, then source + references.

For each assigned base ID the crawler

1. resolves the paper with one ``id_list`` API call per batch of IDs and reads
   the latest version ``vN`` from the returned identifier;
2. asks the API once more for the versioned IDs ``v1..vN`` (batched) to get the
   date of every version, which fills ``submission_date`` and ``revised_dates``;
3. downloads every version from ``export.arxiv.org/src/<id>vK`` and keeps only
   the ``.tex`` / ``.bib`` files;
4. fetches the reference list once per paper from Semantic Scholar and stores a
   copy beside every version.

Output layout::

    <out>/<base_id>/v<K>/{metadata.json, references.json, *.tex, *.bib}
    <out>/crawl_log.json   (missing IDs, failed versions, reference status)

Usage::

    python arxiv_crawler.py --year-month 2510 --start 10788 --end 10807
    python arxiv_crawler.py --member tan --members-file ../members.json
    python arxiv_crawler.py --ids 2510.10788 2510.10790 --out sourcesArxiv
"""

from __future__ import annotations

import argparse
import re
import time  # noqa: F401  (patched by the tests through this module)
import urllib.request  # noqa: F401  (patched by the tests through this module)
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import arxiv

try:
    from . import crawler_common as common
except ImportError:  # executed as a script from inside src/
    import crawler_common as common

# Re-exported for callers such as benchmark.py.
chunk_list = common.chunk_list
fetch_references = common.fetch_references
get_paper_references = common.get_paper_references
save_json_file = common.save_json

DISCOVERY_BATCH_SIZE = 100
DEFAULT_OUTPUT_DIR = "./sourcesArxiv"

client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _iso(value) -> Optional[str]:
    return value.isoformat() if value else None


def latest_version_of(paper) -> Tuple[str, int]:
    """(base_id, latest version) from the identifier the API returned."""
    entry_id = getattr(paper, "entry_id", "") or ""
    return common.split_version(entry_id.rsplit("/abs/", 1)[-1])


def discover_papers(ids: List[str]) -> Tuple[Dict[str, object], List[str]]:
    """Resolve base IDs with the arXiv API.

    Returns ``(papers_by_base_id, missing_ids)``; an ID is missing when the API
    returned no entry for it (withdrawn, mistyped or not yet public).
    """
    papers: Dict[str, object] = {}
    for batch in chunk_list(ids, DISCOVERY_BATCH_SIZE):
        search = arxiv.Search(id_list=batch, max_results=len(batch))
        for paper in client.results(search):
            base_id, _ = latest_version_of(paper)
            papers[base_id] = paper
    missing = [paper_id for paper_id in ids if paper_id not in papers]
    for paper_id in missing:
        print(f"[MISSING] {paper_id}: not returned by the arXiv API")
    return papers, missing


def fetch_version_dates(papers: Dict[str, object]) -> Dict[str, Dict[int, str]]:
    """Date of every version, keyed by base ID then version number.

    Querying ``id_list`` with versioned IDs returns one entry per version whose
    ``updated`` field is the date of that version. Papers with a single version
    need no extra request: their only date is ``published``.
    """
    dates: Dict[str, Dict[int, str]] = {}
    versioned_ids: List[str] = []
    for base_id, paper in papers.items():
        _, latest = latest_version_of(paper)
        dates[base_id] = {1: _iso(getattr(paper, "published", None))}
        if latest > 1:
            dates[base_id][latest] = _iso(getattr(paper, "updated", None))
            versioned_ids.extend(f"{base_id}v{v}" for v in range(2, latest))

    for batch in chunk_list(versioned_ids, DISCOVERY_BATCH_SIZE):
        search = arxiv.Search(id_list=batch, max_results=len(batch))
        try:
            for entry in client.results(search):
                base_id, version = latest_version_of(entry)
                if base_id in dates:
                    dates[base_id][version] = _iso(getattr(entry, "updated", None))
        except Exception as exc:  # dates are metadata; never abort the crawl
            print(f"[DATES] version-date lookup failed for a batch: {exc}")
    return dates


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def build_paper_metadata(
    paper,
    versioned_arxiv_id: str,
    version_dates: Optional[Dict[int, str]] = None,
) -> dict:
    """Metadata of one version, following the columns of the ``papers`` table."""
    base_id, version = common.split_version(versioned_arxiv_id)
    version_dates = version_dates or {}
    submission_date = version_dates.get(1) or _iso(getattr(paper, "published", None))
    _, latest = latest_version_of(paper)
    version_date = version_dates.get(version)
    if version_date is None and version == 1:
        version_date = submission_date
    if version_date is None and version == latest:
        version_date = _iso(getattr(paper, "updated", None))
    revised_dates = [
        version_dates[v] for v in sorted(version_dates) if v > 1 and version_dates[v]
    ]
    journal_ref = getattr(paper, "journal_ref", None)

    return {
        "arxiv_id": versioned_arxiv_id,
        "base_id": base_id,
        "version": str(version),
        "title": re.sub(r"\s+", " ", paper.title).strip(),
        "authors": [author.name for author in paper.authors],
        "submission_date": submission_date,
        "version_date": version_date,
        "updated_date": version_date,  # old name, still read by the M1 report notebook
        "revised_dates": revised_dates,
        "venue": journal_ref or None,
        "journal_ref": journal_ref,
        "doi": getattr(paper, "doi", None),
        "primary_category": getattr(paper, "primary_category", None),
        "categories": list(getattr(paper, "categories", []) or []),
    }


# --------------------------------------------------------------------------- #
# One version
# --------------------------------------------------------------------------- #
def get_source(
    arxiv_id: str,
    save_dir: str | Path = DEFAULT_OUTPUT_DIR,
    paper=None,
    references: Optional[list] = None,
    references_status: Optional[str] = None,
    version_dates: Optional[Dict[int, str]] = None,
) -> dict:
    """Download one version and save only its TeX/Bib files and JSON metadata."""
    base_id, version = common.split_version(arxiv_id)
    versioned_arxiv_id = f"{base_id}v{version}"

    if paper is None:
        paper = next(client.results(arxiv.Search(id_list=[base_id], max_results=1)))

    paper_dir = Path(save_dir) / common.paper_folder_name(base_id)
    version_dir = paper_dir / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = version_dir / "metadata.json"

    if common.version_is_complete(version_dir):
        print(f"  {versioned_arxiv_id}: skipped (already downloaded)")
        return {
            "paper_dir": str(paper_dir),
            "version_dir": str(version_dir),
            "source_status": "skipped",
        }

    metadata = build_paper_metadata(paper, versioned_arxiv_id, version_dates)
    metadata["source_status"] = "downloading"
    common.save_json(metadata, metadata_path)
    if references is not None:
        common.save_json(references, version_dir / "references.json")

    download = common.download_source(versioned_arxiv_id)
    files = (
        common.extract_source_archive(download.data, version_dir)
        if download.succeeded
        else None
    )
    status, error = common.source_status(download, files)

    if references_status is None:
        references_status = "success" if isinstance(references, list) else "failed"
    metadata.update(
        {
            "source_status": status,
            "source_format": files["source_format"] if files else None,
            "download_seconds": round(download.seconds, 3),
            "raw_archive_bytes": len(download.data),
            "total_uncompressed_bytes": files["total_uncompressed_bytes"] if files else 0,
            "filtered_uncompressed_bytes": (
                files["filtered_uncompressed_bytes"] if files else 0
            ),
            "tex_files": files["tex_files"] if files else [],
            "bib_files": files["bib_files"] if files else [],
            "references_status": references_status,
            **common.reference_summary(references),
        }
    )
    if error:
        metadata["download_error"] = error
    common.save_json(metadata, metadata_path)

    return {
        "paper_dir": str(paper_dir),
        "version_dir": str(version_dir),
        "tex_files": metadata["tex_files"],
        "bib_files": metadata["bib_files"],
        "raw_archive_bytes": metadata["raw_archive_bytes"],
        "total_uncompressed_bytes": metadata["total_uncompressed_bytes"],
        "filtered_uncompressed_bytes": metadata["filtered_uncompressed_bytes"],
        "source_status": status,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# One paper / whole crawl
# --------------------------------------------------------------------------- #
def _load_saved_references(paper_dir: Path) -> Optional[list]:
    saved = common.read_json(paper_dir / "v1" / "references.json")
    return saved if isinstance(saved, list) else None


def needs_references(paper, save_dir: str | Path = DEFAULT_OUTPUT_DIR) -> bool:
    """True when crawling this paper will have to ask Semantic Scholar for references.

    That includes a paper whose sources are all downloaded but whose references
    are missing (they failed, or the run used ``--no-refs``): a re-run fills them in.
    """
    base_id, _latest = latest_version_of(paper)
    paper_dir = Path(save_dir) / common.paper_folder_name(base_id)
    return _load_saved_references(paper_dir) is None


def attach_references(
    paper_dir: Path, latest: int, references: list, status: str
) -> None:
    """Store late references in every version folder and update their metadata."""
    summary = common.reference_summary(references)
    for version in range(1, latest + 1):
        version_dir = Path(paper_dir) / f"v{version}"
        metadata_path = version_dir / "metadata.json"
        metadata = common.read_json(metadata_path)
        if not isinstance(metadata, dict):
            continue
        common.save_json(references, version_dir / "references.json")
        metadata["references_status"] = status
        metadata.update(summary)
        common.save_json(metadata, metadata_path)


def start_reference_prefetcher(
    papers: Iterable,
    save_dir: str | Path = DEFAULT_OUTPUT_DIR,
    with_refs: bool = True,
    lookahead: int = 25,
) -> Optional[common.ReferencePrefetcher]:
    """Start fetching the references of the papers that need them, in crawl order."""
    if not with_refs:
        return None
    ids = [latest_version_of(p)[0] for p in papers if needs_references(p, save_dir)]
    return common.ReferencePrefetcher(ids, lookahead=lookahead).start() if ids else None


def crawl_paper(
    paper,
    save_dir: str | Path = DEFAULT_OUTPUT_DIR,
    with_refs: bool = True,
    version_dates: Optional[Dict[int, str]] = None,
    prefetcher: Optional[common.ReferencePrefetcher] = None,
) -> dict:
    """Fetch references once, then download every version ``v1..vN``.

    With a ``prefetcher`` the references were requested in the background while
    earlier papers were downloading; without one they are requested here.
    """
    base_id, latest = latest_version_of(paper)
    paper_dir = Path(save_dir) / common.paper_folder_name(base_id)
    print(f"[{base_id}] latest: v{latest}")

    version_dirs = [paper_dir / f"v{v}" for v in range(1, latest + 1)]
    sources_done = all(common.version_is_complete(d) for d in version_dirs)
    references, ref_status = _load_saved_references(paper_dir), "success"
    if sources_done and (references is not None or not with_refs):
        print(f"  {base_id}: all {latest} versions already downloaded")
        return {"base_id": base_id, "versions": latest, "results": [], "skipped": True}

    if references is None and with_refs:
        try:
            if prefetcher is not None and prefetcher.has(base_id):
                references, ref_status = prefetcher.get(base_id)
            else:
                references, ref_status = fetch_references(base_id)
        except Exception as exc:  # one paper must never stop the crawl
            print(f"[S2] {base_id}: {type(exc).__name__}: {exc}")
            references, ref_status = None, "failed"
    elif references is None:
        ref_status = "skipped"

    if sources_done:
        # Only the references were missing: attach them, no source to download.
        if references is not None:
            attach_references(paper_dir, latest, references, ref_status)
        print(f"  {base_id}: references {ref_status} (sources were already downloaded)")
        return {
            "base_id": base_id, "versions": latest, "references_status": ref_status,
            "results": [], "skipped": False,
        }

    results = []
    for version in range(1, latest + 1):
        try:
            results.append(
                get_source(
                    f"{base_id}v{version}",
                    save_dir=save_dir,
                    paper=paper,
                    references=references,
                    references_status=ref_status,
                    version_dates=version_dates,
                )
            )
        except Exception as exc:
            print(f"  {base_id}v{version}: ERROR {type(exc).__name__}: {exc}")
            results.append({"source_status": "error", "error": str(exc)})
    return {
        "base_id": base_id,
        "versions": latest,
        "references_status": ref_status,
        "results": results,
        "skipped": False,
    }


def crawl(
    ids: Iterable[str],
    save_dir: str | Path = DEFAULT_OUTPUT_DIR,
    with_refs: bool = True,
) -> dict:
    """Crawl every base ID and write ``crawl_log.json`` next to the output."""
    ids = list(dict.fromkeys(common.strip_version(i) for i in ids))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    papers, missing = discover_papers(ids)
    dates = fetch_version_dates(papers)
    common.note_arxiv_request()  # the API client made the last arXiv request

    log = {"requested": len(ids), "resolved": len(papers), "missing_ids": missing, "papers": {}}
    ordered = [papers[i] for i in ids if i in papers]
    prefetcher = start_reference_prefetcher(ordered, save_dir, with_refs)
    try:
        for base_id in ids:
            if base_id not in papers:
                continue
            result = crawl_paper(
                papers[base_id], save_dir, with_refs, dates.get(base_id), prefetcher
            )
            log["papers"][base_id] = {
                "versions": result["versions"],
                "references_status": result.get("references_status"),
                "statuses": [r.get("source_status") for r in result["results"]],
            }
    finally:
        if prefetcher is not None:
            prefetcher.stop()
    common.save_json(log, save_dir / "crawl_log.json")
    return log


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="arXiv API crawler (sources + references)")
    common.add_id_arguments(parser)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-refs", action="store_true", help="Skip Semantic Scholar")
    args = parser.parse_args(argv)

    log = crawl(common.resolve_ids(args), args.out, with_refs=not args.no_refs)
    print(
        f"Done: {log['resolved']}/{log['requested']} papers resolved, "
        f"{len(log['missing_ids'])} missing. Log: {Path(args.out) / 'crawl_log.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
