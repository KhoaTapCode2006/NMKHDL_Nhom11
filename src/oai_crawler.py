"""arXiv crawler based on OAI-PMH (arXivRaw) + source download + Semantic Scholar.

Discovery uses the OAI-PMH endpoint ``https://oaipmh.arxiv.org/oai``:
``GetRecord`` for single IDs, ``ListRecords`` with resumption tokens for
date-window harvests. The ``arXivRaw`` format lists every version with its own
date, so ``revised_dates`` comes for free. Download, extraction and references
are shared with ``arxiv_crawler.py`` through ``crawler_common.py``.

Output layout::

    <out>/<base_id>/v<N>/{metadata.json, references.json, *.tex, *.bib}

Usage::

    # explicit IDs
    python oai_crawler.py --ids 2510.10788 2510.10790
    # a sequence range inside one month, or the range of one member
    python oai_crawler.py --year-month 2510 --start 10788 --end 10795
    python oai_crawler.py --member tan --members-file ../members.json
    # bulk harvest by date window (and optional set)
    python oai_crawler.py harvest --from 2025-10-01 --until 2025-10-02 --set cs:cs:CL --limit 50
"""
from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator, List, Optional

import requests

try:
    from . import crawler_common as common
except ImportError:  # executed as a script from inside src/
    import crawler_common as common

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}
USER_AGENT = common.USER_AGENT
DEFAULT_OUTPUT_DIR = "./sourcesOAI"

OAI_MIN_DELAY = 3.0          # arXiv asks for >= 3 s between requests
MAX_RETRIES = 8
BACKOFF_BASE = 3.0
BACKOFF_MAX = 120.0

class OAIError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"OAI error [{code}]: {message}")
        self.code = code


def _throttle(min_delay: float = OAI_MIN_DELAY) -> None:
    """Share the arXiv pacer with the source downloads: 3 s from the start of the
    previous request, whichever arXiv endpoint it went to."""
    common.wait_for_arxiv_slot(min_delay)


def backoff(attempt: int) -> float:
    return min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)


def oai_request(params: dict) -> ET.Element:
    """GET the OAI endpoint with throttling, Retry-After handling and backoff."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            resp = requests.get(OAI_ENDPOINT, params=params, headers=headers, timeout=60)
        except (requests.Timeout, requests.ConnectionError) as exc:
            delay = backoff(attempt)
            print(f"[OAI] {type(exc).__name__}; retry in {delay:.0f}s ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            delay = max(
                backoff(attempt),
                common._retry_after_header(resp.headers.get("Retry-After")),
            )
            print(f"[OAI] HTTP {resp.status_code}; retry in {delay:.0f}s ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            continue

        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        _raise_on_oai_error(root)
        return root

    raise RuntimeError(f"OAI request failed after {MAX_RETRIES} attempts: {params}")


def _raise_on_oai_error(root: ET.Element) -> None:
    err = root.find("oai:error", OAI_NS)
    if err is not None:
        raise OAIError(err.get("code", "unknown"), (err.text or "").strip())


def fetch_oai_record(arxiv_id: str) -> ET.Element:
    """GetRecord for a single arXiv ID (version suffix is stripped)."""
    return oai_request({
        "verb": "GetRecord",
        "identifier": f"oai:arXiv.org:{common.strip_version(arxiv_id)}",
        "metadataPrefix": "arXivRaw",
    })


def list_records(
    date_from: str | None = None,
    date_until: str | None = None,
    oai_set: str | None = None,
    metadata_prefix: str = "arXivRaw",
) -> Iterator[ET.Element]:
    """ListRecords generator following resumptionTokens. Yields <record> elements.

    Resumption tokens of the 2025 endpoint expire daily, so a long harvest
    should finish within a day or restart from its last ``from`` date.
    """
    params: dict = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
    if date_from:
        params["from"] = date_from
    if date_until:
        params["until"] = date_until
    if oai_set:
        params["set"] = oai_set

    while True:
        try:
            root = oai_request(params)
        except OAIError as exc:
            if exc.code == "noRecordsMatch":
                return
            raise

        for record in root.findall(".//oai:record", OAI_NS):
            yield record

        token_el = root.find(".//oai:resumptionToken", OAI_NS)
        token = (token_el.text or "").strip() if token_el is not None else ""
        if not token:
            return
        # Per OAI-PMH, follow-up requests carry only the verb and the token.
        params = {"verb": "ListRecords", "resumptionToken": token}


def list_identifiers(
    date_from: str | None = None,
    date_until: str | None = None,
    oai_set: str | None = None,
) -> Iterator[str]:
    """ListIdentifiers generator yielding base arXiv IDs."""
    params: dict = {"verb": "ListIdentifiers", "metadataPrefix": "arXivRaw"}
    if date_from:
        params["from"] = date_from
    if date_until:
        params["until"] = date_until
    if oai_set:
        params["set"] = oai_set

    while True:
        try:
            root = oai_request(params)
        except OAIError as exc:
            if exc.code == "noRecordsMatch":
                return
            raise
        for ident in root.findall(".//oai:header/oai:identifier", OAI_NS):
            if ident.text:
                yield ident.text.replace("oai:arXiv.org:", "")
        token_el = root.find(".//oai:resumptionToken", OAI_NS)
        token = (token_el.text or "").strip() if token_el is not None else ""
        if not token:
            return
        params = {"verb": "ListIdentifiers", "resumptionToken": token}


# --------------------------------------------------------------------------- #
# arXivRaw XML -> per-version metadata
# --------------------------------------------------------------------------- #
def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(f"raw:{tag}", OAI_NS)
    if child is None or child.text is None:
        return None
    return re.sub(r"\s+", " ", child.text).strip() or None


def parse_authors(raw: str | None) -> list[str]:
    """Split arXivRaw's free-text author string ('A, B (Aff) and C')."""
    if not raw:
        return []
    cleaned = re.sub(r"\([^)]*\)", "", raw)          # drop affiliations in parentheses
    cleaned = re.sub(r"\s+and\s+", ", ", cleaned)
    return [a.strip() for a in cleaned.split(",") if a.strip()]


def _to_iso(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_versions_from_oai(xml_root: ET.Element) -> list[dict]:
    """One metadata dict per version for every record of a GetRecord/ListRecords reply."""
    results: list[dict] = []
    for record in xml_root.iter(f"{{{OAI_NS['oai']}}}record"):
        results.extend(parse_record(record))
    return results


def parse_record(record: ET.Element) -> list[dict]:
    header = record.find("oai:header", OAI_NS)
    if header is not None and header.get("status") == "deleted":
        return []
    raw = record.find("oai:metadata/raw:arXivRaw", OAI_NS)
    if raw is None:
        return []

    base_id = _text(raw, "id")
    if not base_id:
        return []

    categories = (_text(raw, "categories") or "").split()
    version_nodes = raw.findall("raw:version", OAI_NS)
    labels = [node.get("version", "v1") for node in version_nodes]
    dates = [_to_iso(_text(node, "date")) for node in version_nodes]
    first_date = dates[0] if dates else None
    revised_dates = [d for d in dates[1:] if d]
    journal_ref = _text(raw, "journal-ref")

    common_fields = {
        "title": _text(raw, "title"),
        "authors": parse_authors(_text(raw, "authors")),
        "submission_date": first_date,
        "revised_dates": revised_dates,
        "venue": journal_ref,
        "journal_ref": journal_ref,
        "doi": _text(raw, "doi"),
        "primary_category": categories[0] if categories else None,
        "categories": categories,
    }

    versions: list[dict] = []
    for label, date in zip(labels, dates):
        versions.append({
            "arxiv_id": f"{base_id}{label}",
            "base_id": base_id,
            "version": label.lstrip("v"),
            "version_date": date,
            "updated_date": date,  # old name, still read by the M1 report notebook
            **common_fields,
        })
    return versions


def group_by_paper(version_records: list[dict]) -> dict[str, list[dict]]:
    papers: dict[str, list[dict]] = {}
    for rec in version_records:
        papers.setdefault(rec["base_id"], []).append(rec)
    return papers


# --------------------------------------------------------------------------- #
# One version
# --------------------------------------------------------------------------- #
def get_source(
    meta: dict,
    save_dir: Path,
    references: list | None,
    references_status: str | None = None,
) -> dict:
    """Download and store one version: metadata.json, references.json, .tex/.bib."""
    base_id = meta["base_id"]
    version_dir = Path(save_dir) / common.paper_folder_name(base_id) / f"v{meta['version']}"
    version_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = version_dir / "metadata.json"

    if common.version_is_complete(version_dir):
        return {"version_dir": str(version_dir), "source_status": "skipped"}

    metadata = dict(meta)
    metadata["source_status"] = "downloading"
    common.save_json(metadata, metadata_path)
    if references is not None:
        common.save_json(references, version_dir / "references.json")

    download = common.download_source(meta["arxiv_id"])
    files = (
        common.extract_source_archive(download.data, version_dir)
        if download.succeeded
        else None
    )
    status, error = common.source_status(download, files)
    if references_status is None:
        references_status = "success" if isinstance(references, list) else "failed"

    metadata.update({
        "source_status": status,
        "source_format": files["source_format"] if files else None,
        "raw_archive_bytes": len(download.data),
        "total_uncompressed_bytes": files["total_uncompressed_bytes"] if files else 0,
        "filtered_uncompressed_bytes": files["filtered_uncompressed_bytes"] if files else 0,
        "tex_files": files["tex_files"] if files else [],
        "bib_files": files["bib_files"] if files else [],
        "references_status": references_status,
        **common.reference_summary(references),
    })
    if error:
        metadata["download_error"] = error
    common.save_json(metadata, metadata_path)
    return {
        "version_dir": str(version_dir),
        "source_status": status,
        "error": error,
        **(files or {}),
    }


# Shared Semantic Scholar client (kept under the old name for callers).
get_paper_references = common.get_paper_references


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def crawl_paper(
    versions: list[dict],
    save_dir: str | Path = DEFAULT_OUTPUT_DIR,
    with_refs: bool = True,
) -> list[dict]:
    versions = sorted(versions, key=lambda v: int(v["version"]))
    base_id = versions[0]["base_id"]
    paper_dir = Path(save_dir) / common.paper_folder_name(base_id)
    if all(common.version_is_complete(paper_dir / f"v{v['version']}") for v in versions):
        print(f"  {base_id}: all {len(versions)} versions already downloaded")
        return []

    references, ref_status = None, "skipped"
    if with_refs:
        try:
            references, ref_status = common.fetch_references(base_id)
        except Exception as exc:
            print(f"[S2] {base_id}: {exc}")
            references, ref_status = None, "failed"

    results = []
    for meta in versions:
        try:
            result = get_source(meta, save_dir, references, ref_status)
            print(f"  {meta['arxiv_id']}: {result['source_status']}")
        except Exception as exc:  # never let one version kill the crawl
            print(f"  {meta['arxiv_id']}: ERROR {type(exc).__name__}: {exc}")
            result = {"source_status": "error", "error": str(exc)}
        results.append(result)
    return results


def crawl_ids(ids: list[str], save_dir: Path, with_refs: bool) -> dict:
    missing: List[str] = []
    for arxiv_id in ids:
        print(f"[{arxiv_id}]")
        try:
            versions = extract_versions_from_oai(fetch_oai_record(arxiv_id))
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}")
            missing.append(arxiv_id)
            continue
        if not versions:
            missing.append(arxiv_id)
            continue
        crawl_paper(versions, save_dir, with_refs)
    log = {"requested": len(ids), "missing_ids": missing}
    common.save_json(log, Path(save_dir) / "crawl_log.json")
    return log


def crawl_harvest(args, save_dir: Path) -> None:
    count = 0
    for record in list_records(args.date_from, args.date_until, args.set):
        versions = parse_record(record)
        if not versions:
            continue
        print(f"[{versions[0]['base_id']}]")
        crawl_paper(versions, save_dir, not args.no_refs)
        count += 1
        if args.limit and count >= args.limit:
            break
    print(f"Done: {count} papers")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="arXiv OAI-PMH crawler")
    ap.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="output directory")
    ap.add_argument("--no-refs", action="store_true", help="skip Semantic Scholar references")
    common.add_id_arguments(ap)
    sub = ap.add_subparsers(dest="mode")

    p_h = sub.add_parser("harvest", help="bulk harvest via ListRecords")
    p_h.add_argument("--from", dest="date_from")
    p_h.add_argument("--until", dest="date_until")
    p_h.add_argument("--set", help="OAI set, e.g. cs:cs:CL")
    p_h.add_argument("--limit", type=int, default=0, help="max papers (0 = no limit)")

    args = ap.parse_args(argv)
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "harvest":
        crawl_harvest(args, save_dir)
    else:
        log = crawl_ids(common.resolve_ids(args), save_dir, not args.no_refs)
        print(f"Done: {log['requested'] - len(log['missing_ids'])}/{log['requested']} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
