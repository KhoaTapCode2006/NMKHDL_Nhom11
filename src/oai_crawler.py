"""arXiv crawler based on OAI-PMH (arXivRaw) + source download + Semantic Scholar references.

Output layout:
    sources/<base_id>/v<N>/{metadata.json, references.json, *.tex, *.bib, figures...}

Usage examples:
    # 1) Explicit IDs
    python oai_crawler.py ids 2510.10788 2510.10790

    # 2) A sequence range inside one month (replaces the old arxiv.Search loop)
    python oai_crawler.py range 2510 10788 10795

    # 3) Bulk harvest by date window (and optional set), via ListRecords + resumptionToken
    python oai_crawler.py harvest --from 2025-10-01 --until 2025-10-02 --set cs:cs:CL --limit 50
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import io
import json
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Iterator
import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OAI_ENDPOINT = "https://export.arxiv.org/oai2"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}
SRC_URL = "https://export.arxiv.org/src/"
USER_AGENT = "NMKHDL-Nhom11 friendly cute crawler from VN"

OAI_MIN_DELAY = 3.0          # arXiv asks for >= 3s between requests
S2_INTERVAL = 3.1            # delay between Semantic Scholar calls
MAX_RETRIES = 8
BACKOFF_BASE = 3.0
BACKOFF_MAX = 120.0

API_KEY = os.getenv("API_KEY")  
ALLOWED_EXTENSIONS = {
    ".tex", ".bib",      
}

_last_request_ts = 0.0


class OAIError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"OAI error [{code}]: {message}")
        self.code = code


def _throttle(min_delay: float = OAI_MIN_DELAY) -> None:
    global _last_request_ts
    wait = min_delay - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def backoff(attempt: int) -> float:
    return min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)


def oai_request(params: dict) -> ET.Element:
    """GET the OAI endpoint with throttling, Retry-After handling and exponential backoff."""
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

        if resp.status_code == 503:
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit() == True:
                delay = float(retry_after)
            else:
                delay = backoff(attempt)
            print(f"Error 503, Retry after: {retry_after or 'n/a'}; sleeping {delay:.0f}s")
            time.sleep(delay)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            delay = backoff(attempt)
            print(f"HTTP {resp.status_code}; retry in {delay:.0f}s ({attempt + 1}/{MAX_RETRIES})")
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
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    return oai_request({
        "verb": "GetRecord",
        "identifier": f"oai:arXiv.org:{base_id}",
        "metadataPrefix": "arXivRaw",
    })


def list_records(
    date_from: str | None = None,
    date_until: str | None = None,
    oai_set: str | None = None,
    metadata_prefix: str = "arXivRaw",
) -> Iterator[ET.Element]:
    """ListRecords generator following resumptionTokens. Yields <record> elements."""
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
# Phase 2: parse arXivRaw XML into per-version metadata
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
    """Return one metadata dict per version for every record in an OAI response.

    Works for GetRecord and ListRecords responses. Deleted records are skipped.
    """
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
    dates = [_to_iso(_text(v, "date")) for v in version_nodes]
    first_date = dates[0] if dates else None

    common = {
        "title": _text(raw, "title"),
        "authors": parse_authors(_text(raw, "authors")),
        "journal_ref": _text(raw, "journal-ref"),
        "primary_category": categories[0] if categories else None,
        "categories": categories,
    }

    versions: list[dict] = []
    for node, date in zip(version_nodes, dates):
        label = node.get("version", "v1")                      # e.g. "v2"
        versions.append({
            "arxiv_id": f"{base_id}{label}",
            "base_id": base_id,
            "version": label.lstrip("v"),
            "submission_date": first_date,   # v1 submission, same for all versions
            "updated_date": date,            # date of this specific version
            **common,
        })
    return versions


def group_by_paper(version_records: list[dict]) -> dict[str, list[dict]]:
    papers: dict[str, list[dict]] = {}
    for rec in version_records:
        papers.setdefault(rec["base_id"], []).append(rec)
    return papers


def download_source(versioned_id: str, retries: int = 3) -> bytes:
    """Download a source tarball; return b'' only when arXiv reports it missing."""
    request = urllib.request.Request(SRC_URL + versioned_id, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return resp.read()
        except http.client.IncompleteRead:
            print(f"[SRC] IncompleteRead {versioned_id} ({attempt + 1}/{retries})")
            if attempt + 1 < retries:
                time.sleep(backoff(attempt))
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return b""
            if err.code in (421, 429, 500, 502, 503, 504):
                retry_after = err.headers.get("Retry-After", "") if err.headers else ""
                delay = float(retry_after) if retry_after.isdigit() else backoff(attempt)
                if attempt + 1 < retries:
                    print(f"[SRC] HTTP {err.code} {versioned_id}; retry in {delay:.0f}s ({attempt + 1}/{retries})")
                    time.sleep(delay)
                else:
                    print(f"[SRC] HTTP {err.code} {versioned_id}; retries exhausted")
            else:
                raise
        except Exception as exc:  # network errors, timeouts
            print(f"[SRC] {type(exc).__name__} {versioned_id} ({attempt + 1}/{retries})")
            if attempt + 1 < retries:
                time.sleep(backoff(attempt))
    raise RuntimeError(f"Failed to download source for {versioned_id} after {retries} attempts")


def _safe_member_path(name: str) -> PurePosixPath | None:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return p


def extract_source_assets(archive_bytes: bytes, version_dir: Path) -> dict:
    """Extract tex/bib/figure files from an arXiv source blob, preserving nested paths.

    Handles tar(.gz) archives and the single gzipped-file case arXiv uses for
    one-file submissions.
    """
    out = {"tex_files": [], "bib_files": []}
    if not archive_bytes:
        return out

    def record(path: Path) -> None:
        rel = str(path.relative_to(version_dir))
        ext = path.suffix.lower()
        if ext == ".tex":
            out["tex_files"].append(rel)
        elif ext == ".bib":
            out["bib_files"].append(rel)

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                p = _safe_member_path(member.name)
                if p is None:
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                dest = version_dir.joinpath(*p.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read())
                record(dest)
        return out
    except tarfile.ReadError:
        pass

    # Not a tar: maybe a single gzipped .tex file
    try:
        data = gzip.decompress(archive_bytes)
    except (OSError, EOFError):
        return out
    if b"\\documentclass" in data or b"\\begin{document}" in data:
        dest = version_dir / "main.tex"
        dest.write_bytes(data)
        record(dest)
    return out


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_source(meta: dict, save_dir: Path, references: list | None) -> dict:
    """Download and store one version: metadata.json, references.json and source assets."""
    base_id = meta["base_id"]
    version_dir = save_dir / base_id.replace("/", "-") / f"v{meta['version']}"
    version_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = version_dir / "metadata.json"

    # Resume support: skip versions that were fully downloaded earlier
    if metadata_path.exists():
        try:
            if json.loads(metadata_path.read_text("utf-8")).get("source_status") in {"downloaded", "no_tex_or_bib"}:
                return {"version_dir": str(version_dir), "source_status": "skipped"}
        except json.JSONDecodeError:
            pass

    metadata = {k: v for k, v in meta.items() if k != "base_id"}
    metadata["source_status"] = "downloading"
    save_json(metadata, metadata_path)
    if references is not None:
        save_json(references, version_dir / "references.json")

    try:
        archive_bytes = download_source(meta["arxiv_id"])
        files = extract_source_assets(archive_bytes, version_dir)
    except Exception as exc:
        metadata["source_status"] = "download_failed"
        metadata["source_error"] = f"{type(exc).__name__}: {exc}"
        save_json(metadata, metadata_path)
        raise
    has_content = files["tex_files"] or files["bib_files"]
    metadata["source_status"] = "downloaded" if has_content else "no_tex_or_bib"
    metadata["tex_files"] = files["tex_files"]
    metadata["bib_files"] = files["bib_files"]
    save_json(metadata, metadata_path)
    return {"version_dir": str(version_dir), "source_status": metadata["source_status"], **files}


# --------------------------------------------------------------------------- #
# Phase 4: Semantic Scholar references
# --------------------------------------------------------------------------- #
def get_paper_references(arxiv_id: str) -> list[dict] | None:
    if not API_KEY:
        raise RuntimeError("API_KEY is missing. Set it in the .env file.")

    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{base_id}"
    params = {"fields": "references.paperId,references.externalIds,references.title,"
                        "references.authors,references.year"}
    headers = {"x-api-key": API_KEY}

    for attempt in range(10):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            time.sleep(S2_INTERVAL * (attempt + 2))
            continue

        if resp.status_code == 429:
            wait = S2_INTERVAL * (attempt + 2)
            print(f"[S2] 429 for {base_id}; waiting {wait:.1f}s ({attempt + 1}/10)")
            time.sleep(wait)
            continue
        if resp.status_code == 404:          # paper not indexed yet
            time.sleep(S2_INTERVAL)
            return []
        resp.raise_for_status()
        time.sleep(S2_INTERVAL)
        return resp.json().get("references") or []

    print(f"Skipping {base_id}: persistent 429")
    return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def crawl_paper(versions: list[dict], 
                save_dir: str | Path = "./sourcesOAI"
                , with_refs: bool = True) -> None:
    versions = sorted(versions, key=lambda v: int(v["version"]))
    base_id = versions[0]["base_id"]
    references = None
    if with_refs:
        try:
            references = get_paper_references(base_id)
        except Exception as exc:
            print(f"[S2] {base_id}: {exc}")
    for meta in versions:
        try:
            result = get_source(meta, save_dir, references)
            print(f"  {meta['arxiv_id']}: {result['source_status']}")
        except Exception as exc:  # never let one version kill the crawl
            print(f"  {meta['arxiv_id']}: ERROR {type(exc).__name__}: {exc}")


def crawl_ids(ids: list[str], save_dir: Path, with_refs: bool) -> None:
    for arxiv_id in ids:
        print(f"[{arxiv_id}]")
        try:
            versions = extract_versions_from_oai(fetch_oai_record(arxiv_id))
        except OAIError as exc:
            print(f"  skipped: {exc}")
            continue
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}")
            continue
        if versions:
            crawl_paper(versions, save_dir, with_refs)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="arXiv OAI-PMH crawler")
    ap.add_argument("--out", default="./sourcesOAI", help="output directory")
    ap.add_argument("--no-refs", action="store_true", help="skip Semantic Scholar references")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_ids = sub.add_parser("ids", help="crawl explicit arXiv IDs")
    p_ids.add_argument("arxiv_ids", nargs="+")

    p_rng = sub.add_parser("range", help="crawl YYMM.<start>..<end>")
    p_rng.add_argument("year_month")
    p_rng.add_argument("start", type=int)
    p_rng.add_argument("end", type=int)

    p_h = sub.add_parser("harvest", help="bulk harvest via ListRecords")
    p_h.add_argument("--from", dest="date_from")
    p_h.add_argument("--until", dest="date_until")
    p_h.add_argument("--set", help="OAI set, e.g. cs:cs:CL")
    p_h.add_argument("--limit", type=int, default=0, help="max papers (0 = no limit)")

    args = ap.parse_args()
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "ids":
        crawl_ids(args.arxiv_ids, save_dir, not args.no_refs)
    elif args.mode == "range":
        ids = [f"{args.year_month}.{n:05d}" for n in range(args.start, args.end + 1)]
        crawl_ids(ids, save_dir, not args.no_refs)
    else:
        crawl_harvest(args, save_dir)


if __name__ == "__main__":
    main()
