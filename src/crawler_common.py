"""Helpers shared by ``arxiv_crawler.py`` and ``oai_crawler.py``.

The two crawlers differ only in how they discover papers and their versions.
Everything after discovery lives here:

* downloading one versioned source archive from ``export.arxiv.org/src/``;
* extracting only the ``.tex`` and ``.bib`` members (figures never reach disk);
* querying Semantic Scholar for the reference list of a paper;
* reading the per-member ID ranges from ``members.json``.
"""

from __future__ import annotations

import gzip
import http.client
import io
import json
import logging
import os
import re
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source download
# --------------------------------------------------------------------------- #
SRC_URL = "https://export.arxiv.org/src/"
USER_AGENT = "NMKHDL-Nhom11 arXiv source crawler (university course project)"
SOURCE_REQUEST_DELAY = 3.0          # arXiv etiquette: >= 3 s between requests
SOURCE_MAX_RETRIES = 3
SOURCE_BACKOFF_BASE = 3.0
SOURCE_BACKOFF_MAX = 60.0
RETRYABLE_HTTP_CODES = {421, 429, 500, 502, 503, 504}
ALLOWED_EXTENSIONS = {".tex", ".bib"}
TEX_MARKERS = (rb"\documentclass", rb"\documentstyle", rb"\begin{document}")


@dataclass
class DownloadResult:
    """Outcome of one source download.

    ``not_found`` means arXiv answered 404 (no source for this version), which
    is a legitimate empty record. ``error`` is set only for real failures.
    ``seconds`` is the time the successful or final request took.
    """

    data: bytes = b""
    not_found: bool = False
    error: Optional[str] = None
    seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and not self.not_found


def _backoff(attempt: int, base: float, maximum: float) -> float:
    return min(base * (2 ** attempt), maximum)


def _retry_after_header(value: Optional[str]) -> float:
    """Parse a Retry-After header given either as seconds or as an HTTP date."""
    value = (value or "").strip()
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


# --------------------------------------------------------------------------- #
# arXiv request pacing
# --------------------------------------------------------------------------- #
# arXiv asks for at most one request every 3 seconds and one connection at a
# time. The 3 seconds are counted from the START of the previous request, so the
# time spent downloading and unpacking an archive counts towards the wait
# instead of being added to it. A request that takes longer than 3 seconds is
# followed immediately by the next one. Requests are never overlapped: every
# caller must finish one arXiv request before it starts the next (the
# Semantic Scholar prefetcher talks to another server and does not use this).
_arxiv_lock = threading.Lock()
_arxiv_next_start = 0.0


def wait_for_arxiv_slot(delay: float = SOURCE_REQUEST_DELAY) -> None:
    """Block until a request to arXiv may start, and reserve the next slot."""
    global _arxiv_next_start
    with _arxiv_lock:
        remaining = _arxiv_next_start - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        _arxiv_next_start = time.monotonic() + delay


def note_arxiv_request(delay: float = SOURCE_REQUEST_DELAY) -> None:
    """Record a request made elsewhere (the arxiv.py client) so the next one waits."""
    global _arxiv_next_start
    with _arxiv_lock:
        _arxiv_next_start = max(_arxiv_next_start, time.monotonic() + delay)


def reset_arxiv_pacer() -> None:
    global _arxiv_next_start
    with _arxiv_lock:
        _arxiv_next_start = 0.0


_ORIGINAL_URLOPEN = urllib.request.urlopen
_ORIGINAL_REQUESTS_GET = requests.get
_ARXIV_SESSION = requests.Session()
_ARXIV_SESSION.headers.update({"User-Agent": USER_AGENT})
_S2_SESSION = requests.Session()
_S2_SESSION.headers.update({"User-Agent": USER_AGENT})


def download_source(
    versioned_id: str,
    retries: int = SOURCE_MAX_RETRIES,
    delay: float = SOURCE_REQUEST_DELAY,
) -> DownloadResult:
    """Download the source archive of one version, e.g. ``2510.10788v2``.

    Uses a persistent ``requests.Session`` (HTTP Keep-Alive) so successive
    requests reuse the same single TCP/TLS connection to ``export.arxiv.org``
    without repeating handshakes. Falls back to ``urllib.request.urlopen`` if a
    caller/test monkeypatches it.
    """
    url = SRC_URL + versioned_id
    headers = {"User-Agent": USER_AGENT}
    use_urllib = urllib.request.urlopen is not _ORIGINAL_URLOPEN
    error: Optional[str] = None
    for attempt in range(retries):
        retry_after = 0.0
        wait_for_arxiv_slot(delay)
        started = time.monotonic()
        try:
            if use_urllib:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                return DownloadResult(data=data, seconds=time.monotonic() - started)

            response = _ARXIV_SESSION.get(url, headers=headers, timeout=120)
            elapsed = time.monotonic() - started
            if response.status_code == 404:
                return DownloadResult(not_found=True, seconds=elapsed)
            if response.status_code >= 400:
                error = f"HTTP {response.status_code}: {response.reason}"
                if response.status_code not in RETRYABLE_HTTP_CODES:
                    return DownloadResult(error=error, seconds=elapsed)
                retry_after = _retry_after_header(response.headers.get("Retry-After"))
            else:
                return DownloadResult(data=response.content, seconds=elapsed)
        except http.client.IncompleteRead as exc:
            error = f"IncompleteRead: {exc}"
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            if exc.code == 404:
                return DownloadResult(not_found=True, seconds=elapsed)
            error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code not in RETRYABLE_HTTP_CODES:
                return DownloadResult(error=error, seconds=elapsed)
            headers_err = exc.headers or {}
            retry_after = _retry_after_header(headers_err.get("Retry-After"))
        except Exception as exc:  # timeouts, connection resets, DNS errors
            error = f"{type(exc).__name__}: {exc}"

        if attempt + 1 < retries:
            wait = max(
                _backoff(attempt, SOURCE_BACKOFF_BASE, SOURCE_BACKOFF_MAX),
                retry_after,
            )
            print(
                f"[SRC] {versioned_id}: {error}; retry in {wait:.0f}s "
                f"({attempt + 1}/{retries})"
            )
            time.sleep(wait)
    return DownloadResult(error=error or "Download failed")


def _safe_member_path(name: str) -> Optional[PurePosixPath]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return path


def extract_source_archive(archive_bytes: bytes, version_dir: Path) -> dict:
    """Write only the ``.tex``/``.bib`` members of an arXiv source blob.

    Handles tar / tar.gz archives, the single gzipped file that arXiv serves for
    one-file submissions, and PDF-only versions. Returns a dict with:

    ``tex_files``, ``bib_files``: paths relative to ``version_dir``;
    ``total_uncompressed_bytes``: size of every file in the archive (before);
    ``filtered_uncompressed_bytes``: size of the kept files (after);
    ``source_format``: ``tar``, ``single_tex``, ``pdf``, ``non_tex`` or ``empty``;
    ``error``: set only when the bytes cannot be read at all.
    """
    version_dir = Path(version_dir)
    out = {
        "tex_files": [],
        "bib_files": [],
        "total_uncompressed_bytes": 0,
        "filtered_uncompressed_bytes": 0,
        "source_format": "empty",
        "error": None,
    }
    if not archive_bytes:
        out["error"] = "Source endpoint returned an empty response"
        return out
    if archive_bytes.lstrip().startswith(b"%PDF-"):
        out["source_format"] = "pdf"
        return out

    def record(path: Path) -> None:
        rel = str(path.relative_to(version_dir))
        if path.suffix.lower() == ".tex":
            out["tex_files"].append(rel)
        else:
            out["bib_files"].append(rel)

    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*")
    except (tarfile.ReadError, EOFError, zlib.error, OSError):
        archive = None

    if archive is not None:
        out["source_format"] = "tar"
        with archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                out["total_uncompressed_bytes"] += member.size
                member_path = _safe_member_path(member.name)
                if member_path is None:
                    continue
                source_file = archive.extractfile(member)
                if source_file is None:
                    continue
                content = source_file.read()
                destination = version_dir.joinpath(*member_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                out["filtered_uncompressed_bytes"] += len(content)
                record(destination)
        return out

    # Not a tar archive: arXiv serves single-file submissions as plain gzip.
    try:
        data = gzip.decompress(archive_bytes)
    except (OSError, EOFError, zlib.error) as exc:
        out["error"] = f"Invalid source archive: {exc}"
        return out

    out["total_uncompressed_bytes"] = len(data)
    if data.lstrip().startswith(b"%PDF-"):
        out["source_format"] = "pdf"
    elif any(marker in data for marker in TEX_MARKERS):
        out["source_format"] = "single_tex"
        destination = version_dir / "main.tex"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        out["filtered_uncompressed_bytes"] = len(data)
        record(destination)
    else:
        # A readable file that is not TeX (e.g. HTML or a Word document):
        # keep the version as an empty record instead of a download failure.
        out["source_format"] = "non_tex"
    return out


def source_status(download: DownloadResult, files: Optional[dict]) -> Tuple[str, Optional[str]]:
    """Map a download + extraction result to ``(source_status, error)``."""
    if download.not_found:
        return "no_tex_or_bib", None
    if not download.succeeded:
        return "download_failed", download.error
    if files is None:
        return "download_failed", "Extraction did not run"
    if files["tex_files"] or files["bib_files"]:
        return "downloaded", None
    if files["error"]:
        return "download_failed", files["error"]
    return "no_tex_or_bib", None


def save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


COMPLETED_STATUSES = {"downloaded", "no_tex_or_bib"}


def version_is_complete(version_dir: Path) -> bool:
    """True when an earlier run fully processed this version (resume support)."""
    metadata = read_json(Path(version_dir) / "metadata.json")
    return isinstance(metadata, dict) and metadata.get("source_status") in COMPLETED_STATUSES


def paper_folder_name(base_id: str) -> str:
    """Folder name of a publication; old-style IDs ``hep-th/9901001`` lose the slash."""
    return base_id.replace("/", "-")


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def split_version(arxiv_id: str) -> Tuple[str, int]:
    """``2510.10788v3`` -> (``2510.10788``, 3); no suffix means version 1."""
    match = re.match(r"^(.*?)v(\d+)$", arxiv_id.strip())
    if match:
        return match.group(1), int(match.group(2))
    return arxiv_id.strip(), 1


def chunk_list(data: List, chunk_size: int = 100) -> Iterator[List]:
    for start in range(0, len(data), chunk_size):
        yield data[start:start + chunk_size]


# --------------------------------------------------------------------------- #
# Assigned ID ranges
# --------------------------------------------------------------------------- #
def ids_from_range(year_month: str, start: int, end: int) -> List[str]:
    return [f"{year_month}.{seq:05d}" for seq in range(int(start), int(end) + 1)]


def load_member_ids(members_file: Path, member: Optional[str] = None) -> List[str]:
    """Read ``members.json`` and return the IDs of one member, or of everyone.

    Expected format, one entry per member::

        {"khoa": {"year_month": "2506", "start": 10788, "end": 15787},
         "tan":  {"segments": [["2407", 17839, 21794], ["2408", 1, 1044]]},
         "x":    {"ids": ["2510.10788", "2510.10790"]}}

    ``segments`` covers a range that crosses a month boundary (the end of the
    first month must be written explicitly).
    """
    return [pid for pid, _name in member_id_pairs(members_file, member)]


def member_id_pairs(members_file: Path, member: Optional[str] = None) -> List[Tuple[str, str]]:
    """``(arxiv_id, member)`` for every assigned ID, in order, without duplicates."""
    config = json.loads(Path(members_file).read_text(encoding="utf-8"))
    names = [member] if member else list(config)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for name in names:
        if name not in config:
            raise KeyError(f"Member '{name}' not found in {members_file}")
        entry = config[name]
        if "ids" in entry:
            ids = [strip_version(i) for i in entry["ids"]]
        elif "segments" in entry:
            ids = [
                pid
                for year_month, start, end in entry["segments"]
                for pid in ids_from_range(str(year_month), start, end)
            ]
        else:
            ids = ids_from_range(entry["year_month"], entry["start"], entry["end"])
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                pairs.append((pid, name))
    return pairs


def member_of(members_file: Path) -> Dict[str, str]:
    """``{arxiv_id: member}`` for every assigned ID."""
    return dict(member_id_pairs(members_file))


def add_id_arguments(parser) -> None:
    """Register the shared options that select which arXiv IDs to crawl."""
    group = parser.add_argument_group("assigned IDs (pick one way)")
    group.add_argument("--ids", nargs="+", help="Explicit arXiv IDs")
    group.add_argument("--year-month", help="YYMM part of a sequence range, e.g. 2510")
    group.add_argument("--start", type=int, help="First sequence number of the range")
    group.add_argument("--end", type=int, help="Last sequence number of the range")
    group.add_argument("--member", help="Member name in the members file")
    group.add_argument(
        "--members-file",
        default="members.json",
        help="JSON file with the ID range of each member (default: members.json)",
    )


def resolve_ids(args) -> List[str]:
    """Turn the parsed ``add_id_arguments`` options into a list of base IDs."""
    if getattr(args, "ids", None):
        return [strip_version(i) for i in args.ids]
    if getattr(args, "year_month", None) and args.start is not None and args.end is not None:
        return ids_from_range(args.year_month, args.start, args.end)
    members_file = Path(getattr(args, "members_file", "members.json"))
    if getattr(args, "member", None) or members_file.is_file():
        return load_member_ids(members_file, getattr(args, "member", None))
    raise SystemExit(
        "No IDs selected: pass --ids, --year-month/--start/--end, or --member "
        "with a members.json file."
    )


# --------------------------------------------------------------------------- #
# Semantic Scholar references
# --------------------------------------------------------------------------- #
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}"
S2_REFERENCE_FIELDS = (
    "paperId",
    "externalIds",
    "title",
    "authors",
    "year",
    "publicationDate",
    "venue",
)
S2_FIELDS = ",".join(f"references.{name}" for name in S2_REFERENCE_FIELDS)

API_KEY = os.getenv("API_KEY") or os.getenv("S2_API_KEY")
# A key guarantees 1 request/s; the shared unauthenticated pool is far tighter.
REFERENCE_REQUEST_INTERVAL = 1.25
UNAUTHENTICATED_REQUEST_INTERVAL = 3.5
# Measured with a valid key at 0.8 requests/s: about two thirds of the requests
# were answered 429, and slower pacing (2.5, 3.5, 5 s) did not help, so the
# server is throttling at random under load rather than punishing our rate. A
# short retry is as likely to succeed as a long wait, so the delay grows gently
# (1.25 s, 1.9 s, 2.8 s, ... up to 30 s) and many attempts are allowed: with
# 25 attempts a paper is lost only if every one of them is rejected.
REFERENCE_BACKOFF_BASE = 1.25
REFERENCE_BACKOFF_GROWTH = 1.5
REFERENCE_BACKOFF_MAX = 30.0
REFERENCE_MAX_RETRIES = 25
_reference_next_request_at = 0.0
_reference_request_lock = threading.Lock()


def _reference_interval() -> float:
    return REFERENCE_REQUEST_INTERVAL if API_KEY else UNAUTHENTICATED_REQUEST_INTERVAL


def _reference_backoff(attempt: int) -> float:
    """Delay before retry number ``attempt + 1`` of a Semantic Scholar request."""
    return min(REFERENCE_BACKOFF_BASE * (REFERENCE_BACKOFF_GROWTH ** attempt), REFERENCE_BACKOFF_MAX)


def _wait_for_reference_request(minimum_delay: float = 0.0) -> None:
    """Wait for the next request slot using a monotonic deadline."""
    global _reference_next_request_at
    with _reference_request_lock:
        deadline = max(
            _reference_next_request_at,
            time.monotonic() + max(0.0, minimum_delay),
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(remaining)
        _reference_next_request_at = time.monotonic() + _reference_interval()


def _retry_after_seconds(response: requests.Response) -> float:
    return _retry_after_header(response.headers.get("Retry-After", ""))


def fetch_references(arxiv_id: str) -> Tuple[Optional[list], str]:
    """Return ``(references, status)`` for one paper.

    ``status`` is ``success``, ``not_indexed`` (HTTP 404: Semantic Scholar does
    not know the paper yet, references are ``[]``) or ``failed`` (references
    are ``None``). The version suffix is ignored: S2 does not separate versions.
    """
    base_id = strip_version(arxiv_id)
    url = S2_URL.format(base_id)
    params = {"fields": S2_FIELDS}
    headers = {"User-Agent": USER_AGENT}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    retry_delay = 0.0

    http_get = requests.get if requests.get is not _ORIGINAL_REQUESTS_GET else _S2_SESSION.get
    for attempt in range(REFERENCE_MAX_RETRIES):
        _wait_for_reference_request(retry_delay)
        try:
            response = http_get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            if attempt + 1 == REFERENCE_MAX_RETRIES:
                print(f"[S2] {base_id}: giving up after {type(exc).__name__}")
                return None, "failed"
            retry_delay = _reference_backoff(attempt)
            print(
                f"[S2] {base_id}: {type(exc).__name__}; retry in {retry_delay:.1f}s "
                f"({attempt + 1}/{REFERENCE_MAX_RETRIES})"
            )
            continue

        status_code = response.status_code
        if status_code == 404:
            return [], "not_indexed"
        if status_code == 429 or status_code >= 500:
            if attempt + 1 == REFERENCE_MAX_RETRIES:
                print(f"[S2] {base_id}: giving up after repeated HTTP {status_code}")
                return None, "failed"
            retry_delay = max(
                _reference_backoff(attempt),
                _retry_after_seconds(response),
            )
            print(
                f"[S2] {base_id}: HTTP {status_code}; retry in {retry_delay:.1f}s "
                f"({attempt + 1}/{REFERENCE_MAX_RETRIES})"
            )
            continue
        if status_code >= 400:
            print(f"[S2] {base_id}: HTTP {status_code}, not retried")
            return None, "failed"
        try:
            payload = response.json()
        except ValueError:
            return None, "failed"
        return payload.get("references") or [], "success"

    return None, "failed"


def get_paper_references(arxiv_id: str) -> Optional[list]:
    """Reference list of a paper, ``[]`` when not indexed, ``None`` on failure."""
    references, _status = fetch_references(arxiv_id)
    return references


def reference_summary(references: Optional[Iterable[dict]]) -> dict:
    """Counts stored in metadata.json so statistics need not reload the list."""
    if references is None:
        return {"references_count": 0, "arxiv_references_count": 0}
    references = list(references)
    with_arxiv = sum(
        1
        for ref in references
        if isinstance(ref, dict)
        and isinstance(ref.get("externalIds"), dict)
        and ref["externalIds"].get("ArXiv")
    )
    return {"references_count": len(references), "arxiv_references_count": with_arxiv}


class ReferencePrefetcher:
    """Fetch the Semantic Scholar references of upcoming papers in a background thread.

    Semantic Scholar is another server with its own rate limit, so asking it for
    the next papers while arXiv sources are being downloaded costs no arXiv
    request and hides the reference time behind the 3-second arXiv pacing.
    ``lookahead`` bounds how many results may wait unused, so the thread never
    runs far ahead of the crawl.

    Use ``has(id)`` to know whether an ID was queued, and ``get(id)`` to take its
    ``(references, status)``; ``get`` blocks until that result is ready.
    """

    def __init__(
        self,
        ids: Iterable[str],
        fetch=None,
        lookahead: int = 25,
    ) -> None:
        self._ids = list(dict.fromkeys(ids))
        self._queued = set(self._ids)
        self._fetch = fetch or fetch_references
        self._results: Dict[str, Tuple[Optional[list], str]] = {}
        self._condition = threading.Condition()
        self._slots = threading.Semaphore(max(1, lookahead))
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="reference-prefetcher", daemon=True
        )

    def start(self) -> "ReferencePrefetcher":
        if self._ids:
            self._thread.start()
        return self

    def _run(self) -> None:
        for paper_id in self._ids:
            self._slots.acquire()
            if self._stopped.is_set():
                return
            try:
                result = self._fetch(paper_id)
            except Exception as exc:  # one paper must not stop the others
                print(f"[S2] {paper_id}: {type(exc).__name__}: {exc}")
                result = (None, "failed")
            with self._condition:
                self._results[paper_id] = result
                self._condition.notify_all()

    def has(self, paper_id: str) -> bool:
        return paper_id in self._queued

    def try_get(self, paper_id: str) -> Optional[Tuple[Optional[list], str]]:
        """Return ``(references, status)`` immediately if ready, else ``None``."""
        with self._condition:
            if paper_id not in self._results:
                return None
            result = self._results.pop(paper_id)
        self._slots.release()
        return result

    def get(self, paper_id: str) -> Tuple[Optional[list], str]:
        with self._condition:
            while paper_id not in self._results:
                if not self._thread.is_alive():
                    return None, "failed"  # the thread ended without this result
                self._condition.wait(timeout=1.0)
            result = self._results.pop(paper_id)
        self._slots.release()
        return result

    def stop(self) -> None:
        self._stopped.set()
        self._slots.release()  # wake a thread waiting for a free slot
