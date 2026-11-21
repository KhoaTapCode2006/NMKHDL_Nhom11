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


ARXIV_NEW_RE = re.compile(r"\b(?P<id>\d{4}\.\d{4,5})(?:v\d+)?\b")
ARXIV_OLD_RE = re.compile(
    r"\b(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def normalize_arxiv_id(value: object) -> Optional[str]:
    """Return a base, versionless arXiv ID or None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://(?:arxiv\.org/(?:abs|pdf)/)?", "", text, flags=re.I)
    text = re.sub(r"^(?:arxiv:|arxiv/)", "", text, flags=re.I).strip()
    text = re.sub(r"v\d+$", "", text, flags=re.I)
    if ARXIV_NEW_RE.fullmatch(text):
        return text.lower()
    if ARXIV_OLD_RE.fullmatch(text):
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
        match = ARXIV_NEW_RE.search(text)
        if match:
            return normalize_arxiv_id(match.group("id"))
        match = ARXIV_OLD_RE.search(text)
        if match:
            return normalize_arxiv_id(match.group("id"))
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


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, JSON/JSONL, or Parquet according to file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
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


def flatten_references_json(
    source_dir: Path,
) -> pd.DataFrame:
    """Flatten crawler references.json files for Semantic Scholar API rows."""
    rows: List[dict] = []
    for references_path in sorted(source_dir.rglob("references.json")):
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
                "title": reference.get("title"),
                "authors": [
                    author.get("name") if isinstance(author, dict) else author
                    for author in (reference.get("authors") or [])
                    if author
                ],
                "year": reference.get("year"),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "citing_arxiv_id",
            "ref_arxiv_id",
            "title",
            "authors",
            "year",
        ],
    )


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
        default=project_root / "data" / "cleaned",
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
    else:
        print("[references] input not found; use --references PATH or --references-source-dir PATH")

    print(f"[output] {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
