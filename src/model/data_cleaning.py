"""Milestone 3 Phase 1: normalize BibTeX and API reference tables."""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd


# New scheme (April 2007 on): YYMM.NNNN until 1412, YYMM.NNNNN from 1501.
# The look-behind rejects numbers inside other identifiers, e.g. the DOI
# "10.1016/j.cell.2020.01234" must not yield "2020.01234".
ARXIV_NEW_RE = re.compile(
    r"(?<![\w.])(?P<id>(?P<yymm>\d{4})\.(?P<seq>\d{4,5}))(?:v\d+)?(?![\w.]*\d)"
)
# An explicit "arXiv" prefix is trusted even after a dot, as in the arXiv DOI
# "10.48550/arXiv.2301.12345" or "arXiv:2301.12345".
ARXIV_PREFIXED_RE = re.compile(
    r"arxiv(?:\.org/(?:abs|pdf)/|[.:/\s]\s*)(?P<id>\d{4}\.\d{4,5})(?:v\d+)?(?!\d)",
    re.IGNORECASE,
)
# Old scheme (before April 2007): archive[.SC]/YYMMNNN, archives listed explicitly.
OLD_ARCHIVES = (
    "acc-phys|adap-org|alg-geom|ao-sci|astro-ph|atom-ph|bayes-an|chao-dyn|chem-ph|"
    "cmp-lg|comp-gas|cond-mat|cs|dg-ga|funct-an|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|"
    "math-ph|math|mtrl-th|nlin|nucl-ex|nucl-th|patt-sol|physics|plasm-ph|q-alg|"
    "q-bio|q-fin|quant-ph|solv-int|stat|supr-con"
)
ARXIV_OLD_RE = re.compile(
    rf"(?<![\w-])(?P<id>(?:{OLD_ARCHIVES})(?:\.[A-Z]{{2}})?/(?P<yymm>\d{{4}})\d{{3}})"
    r"(?:v\d+)?(?!\d)",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _valid_new_scheme(yymm: str, seq: str) -> bool:
    """Month 01-12, not before 0704, 4-digit sequence until 1412 and 5 digits after."""
    month = int(yymm[2:])
    if not 1 <= month <= 12 or yymm < "0704":
        return False
    return len(seq) == (4 if yymm < "1501" else 5)


def _valid_old_scheme(yymm: str) -> bool:
    """Old identifiers run from 9108 to 0703 with a real month."""
    year, month = int(yymm[:2]), int(yymm[2:])
    if not 1 <= month <= 12:
        return False
    return year >= 91 or yymm <= "0703"


def normalize_arxiv_id(value: object) -> Optional[str]:
    """Return a base, versionless arXiv ID or None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://(?:arxiv\.org/(?:abs|pdf)/)?", "", text, flags=re.I)
    text = re.sub(r"^(?:arxiv:|arxiv/)", "", text, flags=re.I).strip()
    text = re.sub(r"v\d+$", "", text, flags=re.I)
    match = ARXIV_NEW_RE.fullmatch(text)
    if match and _valid_new_scheme(match.group("yymm"), match.group("seq")):
        return text.lower()
    match = ARXIV_OLD_RE.fullmatch(text)
    if match and _valid_old_scheme(match.group("yymm")):
        return text.lower()
    return None


def extract_arxiv_id(*values: object) -> Optional[str]:
    """Extract the first valid arXiv ID from text fields."""
    for value in values:
        if value is None or pd.isna(value):
            continue
        direct = normalize_arxiv_id(value)
        if direct:
            return direct
        text = str(value)
        for pattern in (ARXIV_PREFIXED_RE, ARXIV_NEW_RE, ARXIV_OLD_RE):
            for match in pattern.finditer(text):
                candidate = normalize_arxiv_id(match.group("id"))
                if candidate:
                    return candidate
    return None


def clean_latex(text: object) -> str:
    """Convert common LaTeX formatting to plain text."""
    if text is None or pd.isna(text):
        return ""
    value = str(text)
    value = re.sub(r"\\(?:emph|textit|textbf|textrm|textsc)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\$\{([^{}]*)\}\$", r"\1", value)
    value = re.sub(r"\\url\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:[\"'`^~=uvHcdb])\s*\{?\s*([A-Za-z])\s*\}?", r"\1", value)
    value = value.replace(r"\&", "&")
    value = value.replace(r"\_", "_")
    value = value.replace(r"\%", "%")
    value = value.replace(r"\#", "#")
    value = value.replace("~", " ")
    value = re.sub(r"\\[A-Za-z]+\*?(?:\s*\[[^\]]*\])?\s*(?:\{[^{}]*\})?", " ", value)
    value = value.replace("{", "").replace("}", "")
    return value


def clean_title(value: object) -> str:
    """Create a stable lowercase, accent-free title key."""
    text = clean_latex(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-")
    text = text.replace("\u2014", "-").replace("\u2015", "-")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_year(value: object) -> Optional[int]:
    """Extract the first four-digit publication year."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(value))
    if not match:
        return None
    return int(match.group(0))


def _split_author_string(value: str) -> List[str]:
    """Split BibTeX and plain-text author strings."""
    text = clean_latex(value).strip()
    if not text:
        return []

    groups = re.split(r"\s+\band\b\s+|;", text, flags=re.IGNORECASE)
    names: List[str] = []
    for group in groups:
        group = group.strip(" ,.")
        if not group:
            continue
        # "A. Vaswani, N. Shazeer" is comma-separated initials.
        comma_parts = re.split(
            r",\s*(?=(?:[A-Z]\.\s*)+[A-Z][\w'’\-]*)",
            group,
        )
        names.extend(part.strip(" ,.") for part in comma_parts if part.strip(" ,"))
    return names


def clean_author_name(value: object) -> Optional[str]:
    """Return a normalized <first_initial>_<last_name> author key."""
    if value is None or pd.isna(value):
        return None
    text = clean_latex(value).strip(" \t\r\n,.")
    if not text:
        return None
    if text.lower().replace(".", "").strip() in {"et al", "others"}:
        return None

    if "," in text:
        last, first = (part.strip() for part in text.split(",", 1))
    else:
        parts = re.findall(r"[^\s]+", text)
        if not parts:
            return None
        first = parts[0]
        last = parts[-1]

    first = unicodedata.normalize("NFKD", clean_latex(first))
    last = unicodedata.normalize("NFKD", clean_latex(last))
    first = "".join(char for char in first if not unicodedata.combining(char)).lower()
    last = "".join(char for char in last if not unicodedata.combining(char)).lower()
    first = re.sub(r"[^a-z]", "", first)
    last = re.sub(r"[^a-z]", "", last)
    if not last:
        return None
    if not first:
        return last
    return f"{first[0]}_{last}"


def clean_authors(value: object) -> List[str]:
    """Normalize a list or string of authors."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

    if isinstance(value, (list, tuple, set)):
        raw_names = list(value)
    elif isinstance(value, pd.Series):
        raw_names = value.dropna().tolist()
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                raw_names = parsed
            else:
                raw_names = _split_author_string(text)
        except (SyntaxError, ValueError):
            parsed = None
        try:
            if parsed is None:
                parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_names = parsed
            elif parsed is None:
                raw_names = _split_author_string(text)
            else:
                raw_names = _split_author_string(text)
        except json.JSONDecodeError:
            if parsed is None:
                raw_names = _split_author_string(text)
            else:
                raw_names = parsed if isinstance(parsed, list) else _split_author_string(text)

    cleaned: List[str] = []
    index = 0
    while index < len(raw_names):
        raw_name = raw_names[index]
        if isinstance(raw_name, Mapping):
            raw_name = (
                raw_name.get("name")
                or raw_name.get("author")
                or raw_name.get("full_name")
            )
        elif isinstance(raw_name, str):
            raw_name = raw_name.strip()

        # Real parser output contains alternating "Family" and "Initial" entries.
        if (
            isinstance(raw_name, str)
            and "," not in raw_name
            and index + 1 < len(raw_names)
            and isinstance(raw_names[index + 1], str)
            and re.fullmatch(
                r"(?:[A-Z]\.\s*)+[A-Z]?(?:[A-Z]\.)?",
                raw_names[index + 1].strip(),
            )
        ):
            raw_name = f"{raw_name}, {raw_names[index + 1].strip()}"
            index += 1

        author = clean_author_name(raw_name)
        if author and author not in cleaned:
            cleaned.append(author)
        index += 1
    return cleaned


def find_column(
    columns: Iterable[str],
    aliases: Sequence[str],
) -> Optional[str]:
    """Find a column by exact name or a normalized alias."""
    available = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in available:
            return available[key]
    return None


# Identifier columns must stay text: read as numbers, "2407.17850" would become
# 2407.1785 and no longer match the publication or reference it names.
ID_COLUMNS = (
    "arxiv_id", "citing_arxiv_id", "ref_arxiv_id", "groundtruth_arxiv_id",
    "candidate_ref_id", "extracted_arxiv_id", "suggested_arxiv_id",
    "prefilled_arxiv_id", "citation_key", "eprint", "ref_s2_id",
    "cand_1", "cand_2", "cand_3", "cand_4", "cand_5",
)


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, JSON/JSONL, or Parquet according to file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype={column: str for column in ID_COLUMNS})
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True, dtype=False)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("bibtex", "references", "entries", "data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.json_normalize(data)
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise RuntimeError(
                f"Cannot read {path}: install a Parquet engine (pyarrow or fastparquet)"
            ) from error
    raise ValueError(f"Unsupported input format: {path}")


def print_schema_report(label: str, frame: pd.DataFrame) -> None:
    """Print schema, sample rows, and core-column null rates."""
    print(f"\n[{label}] columns:")
    for column, dtype in frame.dtypes.items():
        print(f"  - {column}: {dtype}")
    print(f"[{label}] sample (first 5 rows):")
    print(frame.head(5).to_string(index=False))

    for candidates in (
        ("title", "title"),
        ("author", "author", "authors"),
        ("year", "year", "publicationYear"),
    ):
        found = find_column(frame.columns, candidates)
        if found:
            null_count = int(frame[found].isna().sum())
            print(
                f"[{label}] null {found}: {null_count}/{len(frame)} "
                f"({null_count / len(frame) * 100:.2f}%)"
                if len(frame)
                else f"[{label}] null {found}: 0/0"
            )


def clean_bibtex_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized BibTeX columns and preserve every original column."""
    result = frame.copy()
    title_column = find_column(frame.columns, ("title", "name"))
    author_column = find_column(frame.columns, ("authors", "author"))
    year_column = find_column(frame.columns, ("year", "date", "publicationYear"))
    result["clean_title"] = (
        result[title_column].map(clean_title) if title_column else ""
    )
    result["clean_authors_list"] = (
        result[author_column].map(clean_authors) if author_column else [[]] * len(result)
    )
    result["clean_year"] = (
        pd.array(
            result[year_column].map(clean_year),
            dtype="Int64",
        )
        if year_column
        else pd.array([pd.NA] * len(result), dtype="Int64")
    )

    text_columns = [
        column
        for column in ("eprint", "note", "url", "journal", "booktitle", "doi", "title")
        if column in result.columns
    ]

    def extract(row: pd.Series) -> Optional[str]:
        values = [row[column] for column in text_columns]
        return extract_arxiv_id(*values)

    result["extracted_arxiv_id"] = result.apply(extract, axis=1)
    return result


def collect_bibtex_tables(source_dir: Path) -> pd.DataFrame:
    """Aggregate every publication-level BibTeX JSON/CSV table."""
    source_dir = Path(source_dir)
    frames: List[pd.DataFrame] = []

    json_paths = sorted(source_dir.rglob("bibtex.json"))
    json_parents = {path.parent for path in json_paths}

    for path in json_paths:
        try:
            frames.append(read_table(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[bibtex] skipped {path}: {error}")

    for path in sorted(source_dir.rglob("bibtex.csv")):
        if path.parent in json_parents:
            continue
        try:
            frames.append(read_table(path))
        except (OSError, ValueError) as error:
            print(f"[bibtex] skipped {path}: {error}")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    subset = [
        column
        for column in ("arxiv_id", "citation_key")
        if column in combined.columns
    ]
    if len(subset) == 2:
        combined = combined.drop_duplicates(subset=subset, keep="first")
    return combined


REFERENCE_COLUMNS = [
    "citing_arxiv_id",
    "ref_arxiv_id",
    "ref_s2_id",
    "title",
    "authors",
    "year",
    "publication_date",
    "venue",
]


def _one_references_file_per_paper(source_dir: Path) -> List[Path]:
    """The crawler copies one reference list into every version: read it once.

    Keeps the lowest version per paper folder (``<paper>/v<N>/references.json``);
    files outside that layout are all kept.
    """
    chosen: dict = {}
    others: List[Path] = []
    for path in sorted(Path(source_dir).rglob("references.json")):
        version_dir = path.parent
        match = re.fullmatch(r"v(\d+)", version_dir.name)
        if not match:
            others.append(path)
            continue
        paper_dir = version_dir.parent
        version = int(match.group(1))
        if paper_dir not in chosen or version < chosen[paper_dir][0]:
            chosen[paper_dir] = (version, path)
    return [path for _version, path in sorted(chosen.values(), key=lambda x: str(x[1]))] + others


def flatten_references_json(
    source_dir: Path,
) -> pd.DataFrame:
    """Flatten crawler references.json files into one row per arXiv-keyed reference.

    Keeps the Semantic Scholar ``paperId`` (``ref_s2_id``), the publication date
    and the venue, which the published ``references`` table needs.
    """
    rows: List[dict] = []
    for references_path in _one_references_file_per_paper(Path(source_dir)):
        try:
            citing_id = None
            metadata_path = references_path.with_name("metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                citing_id = normalize_arxiv_id(
                    metadata.get("arxiv_id")
                    or metadata_path.parent.parent.name
                )
            if not citing_id:
                citing_id = normalize_arxiv_id(
                    references_path.parent.parent.name
                )
            references = json.loads(references_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            external = reference.get("externalIds") or {}
            ref_id = extract_arxiv_id(
                external.get("ArXiv")
                if isinstance(external, dict)
                else None
            )
            if not ref_id:
                continue
            rows.append({
                "citing_arxiv_id": citing_id,
                "ref_arxiv_id": ref_id,
                "ref_s2_id": reference.get("paperId"),
                "title": reference.get("title"),
                "authors": [
                    author.get("name") if isinstance(author, dict) else author
                    for author in (reference.get("authors") or [])
                    if author
                ],
                "year": reference.get("year"),
                "publication_date": reference.get("publicationDate"),
                "venue": reference.get("venue") or None,
            })
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def build_references_table(references: pd.DataFrame) -> pd.DataFrame:
    """Shape reference rows into the published ``references`` table.

    Columns: arxiv_id (citing paper), ref_arxiv_id, ref_title, ref_authors,
    ref_submission_date (publication date, else the year), ref_s2_id.
    """
    frame = references.copy()
    if "citing_arxiv_id" not in frame.columns:
        frame = clean_references_table(frame)
    frame = frame[frame["ref_arxiv_id"].notna() & frame["citing_arxiv_id"].notna()]

    def submission_date(row: pd.Series) -> Optional[str]:
        date = row.get("publication_date")
        if isinstance(date, str) and date.strip():
            return date.strip()
        year = clean_year(row.get("year"))
        return str(year) if year else None

    def author_list(value: object) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(name) for name in value if name]
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return []
        return [str(value)]

    table = pd.DataFrame({
        "arxiv_id": frame["citing_arxiv_id"].astype(str),
        "ref_arxiv_id": frame["ref_arxiv_id"].astype(str),
        "ref_title": frame.get("title", pd.Series([None] * len(frame), index=frame.index)),
        "ref_authors": frame.get("authors", pd.Series([[]] * len(frame), index=frame.index)).map(author_list),
        "ref_submission_date": frame.apply(submission_date, axis=1) if len(frame) else [],
        "ref_s2_id": frame.get("ref_s2_id", pd.Series([None] * len(frame), index=frame.index)),
    })
    return table.drop_duplicates(subset=["arxiv_id", "ref_arxiv_id"]).reset_index(drop=True)


def clean_references_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid arXiv references and add normalized columns."""
    result = frame.copy()

    citing_column = find_column(
        frame.columns,
        ("citing_arxiv_id", "citing_paper_arxiv_id", "source_arxiv_id", "arxiv_id"),
    )
    ref_column = find_column(
        frame.columns,
        (
            "ref_arxiv_id",
            "cited_arxiv_id",
            "reference_arxiv_id",
            "externalIds.ArXiv",
            "arxiv_id",
        ),
    )
    title_column = find_column(frame.columns, ("title", "name"))
    author_column = find_column(frame.columns, ("authors", "author"))
    year_column = find_column(frame.columns, ("year", "publicationYear"))

    if citing_column:
        result["citing_arxiv_id"] = result[citing_column].map(normalize_arxiv_id)
    else:
        result["citing_arxiv_id"] = pd.NA

    if ref_column:
        result["ref_arxiv_id"] = result[ref_column].map(extract_arxiv_id)
    elif "externalIds" in result.columns:
        result["ref_arxiv_id"] = result["externalIds"].map(
            lambda value: extract_arxiv_id(
                value.get("ArXiv") if isinstance(value, Mapping) else None
            )
        )
    else:
        result["ref_arxiv_id"] = pd.NA

    result = result[
        result["ref_arxiv_id"].notna()
        & result["ref_arxiv_id"].astype(str).str.strip().ne("")
    ].copy()
    result["ref_arxiv_id"] = result["ref_arxiv_id"].map(normalize_arxiv_id)

    result["clean_title"] = (
        result[title_column].map(clean_title) if title_column else ""
    )
    result["clean_authors_list"] = (
        result[author_column].map(clean_authors) if author_column else [[]] * len(result)
    )
    result["clean_year"] = pd.array(
        result[year_column].map(clean_year) if year_column else [pd.NA] * len(result),
        dtype="Int64",
    )

    pair_columns = [column for column in ("citing_arxiv_id", "ref_arxiv_id") if column in result]
    if pair_columns:
        result = result.drop_duplicates(subset=pair_columns, keep="first")
    return result.reset_index(drop=True)


def discover_inputs(
    project_root: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Discover known Milestone 1/2 outputs without hard-coding a row schema."""
    bibtex_candidates = [
        project_root / "data" / "bibtex",
        project_root / "src" / "sourcesArxiv",
        project_root / "sourcesArxiv",
    ]
    reference_candidates = [
        project_root / "data" / "references",
        project_root / "src" / "sourcesArxiv",
        project_root / "sourcesArxiv",
    ]

    bibtex_table = None
    for directory in bibtex_candidates:
        if not directory.is_dir():
            continue
        matches = sorted(
            list(directory.rglob("bibtex.json"))
            + list(directory.rglob("bibtex.csv"))
            + list(directory.rglob("bibtex.parquet"))
        )
        if matches:
            bibtex_table = directory
            break

    references_table = None
    for directory in reference_candidates:
        if not directory.is_dir():
            continue
        direct_matches = sorted(
            list(directory.rglob("references.csv"))
            + list(directory.rglob("references.parquet"))
            + list(directory.rglob("references.json"))
        )
        if direct_matches:
            references_table = directory
            break
    return bibtex_table, references_table


def write_table(frame: pd.DataFrame, path: Path, file_format: str) -> None:
    """Write cleaned output as CSV or Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif file_format == "parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Output format must be 'csv' or 'parquet'")


def print_cleaning_summary(label: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Print accepted and removed row counts."""
    removed = len(before) - len(after)
    print(
        f"[{label}] rows: {len(before)} -> {len(after)} "
        f"(removed {removed}, {removed / len(before) * 100:.2f}%)"
        if len(before)
        else f"[{label}] rows: 0 -> 0 (removed 0)"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Clean Milestone 2 BibTeX and Milestone 1 references."
    )
    parser.add_argument("--bibtex", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "src" / "data" / "cleaned",
    )
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument(
        "--references-source-dir",
        type=Path,
        default=None,
        help="Directory containing crawler references.json files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    discovered_bibtex, discovered_references = discover_inputs(project_root)
    bibtex_path = args.bibtex or discovered_bibtex
    references_path = args.references or discovered_references

    if bibtex_path:
        bibtex_path = bibtex_path.resolve()
        if bibtex_path.is_dir():
            bibtex = collect_bibtex_tables(bibtex_path)
        else:
            bibtex = read_table(bibtex_path)
        if bibtex.empty:
            print("[bibtex] no table rows found")
        else:
            print_schema_report("bibtex", bibtex)
            bibtex_cleaned = clean_bibtex_table(bibtex)
            print_cleaning_summary("bibtex_cleaned", bibtex, bibtex_cleaned)
            write_table(
                bibtex_cleaned,
                args.output_dir / f"bibtex_cleaned.{args.format}",
                args.format,
            )
    else:
        print("[bibtex] input not found; use --bibtex PATH")

    if args.references_source_dir:
        references = flatten_references_json(args.references_source_dir.resolve())
    elif references_path:
        references_path = references_path.resolve()
        if references_path.is_dir():
            references = flatten_references_json(references_path)
        elif references_path.name == "references.json":
            references = flatten_references_json(references_path.parents[2])
        else:
            references = read_table(references_path)
    else:
        references = pd.DataFrame()

    if len(references):
        print_schema_report("references", references)
        references_cleaned = clean_references_table(references)
        print_cleaning_summary("references_cleaned", references, references_cleaned)
        write_table(
            references_cleaned,
            args.output_dir / f"references_cleaned.{args.format}",
            args.format,
        )
        # Published `references` table (arXiv-keyed references only)
        write_table(
            build_references_table(references_cleaned),
            args.output_dir / f"references_table.{args.format}",
            args.format,
        )
    else:
        print("[references] input not found; use --references PATH or --references-source-dir PATH")

    print(f"[output] {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
