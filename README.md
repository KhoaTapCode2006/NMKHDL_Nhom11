# NMKHDL_Nhom11

Course project on arXiv: crawling full-text LaTeX sources (Milestone 1), parsing
them into a deduplicated hierarchy with unified BibTeX entries (Milestone 2),
and matching extracted citations to arXiv reference metadata (Milestone 3).

## 1. Requirements

- Python 3.10 or newer (the benchmark testbed is a CPU-only Google Colab instance)
- Packages:

```bash
pip install -r src/requirements.txt
```

## 2. Configuration

**Semantic Scholar key (optional).** Create a `.env` file at the repository root:

```text
API_KEY=<your Semantic Scholar API key>
```

Without a key the crawler still works, using the shared unauthenticated pool with
a slower pace (one request every 3.5 s instead of every 1.25 s).

**Assigned ID ranges.** Each member crawls their own 5,000-paper range defined in
`members.json` (template in `members.example.json`):

```json
{
  "tan":   {"segments": [["2407", 17839, 21794], ["2408", 1, 1044]]},
  "hoang": {"year_month": "2411", "start": 13799, "end": 18798},
  "kiet":  {"segments": [["2505", 20666, 24878], ["2506", 1, 787]]},
  "khoa":  {"year_month": "2506", "start": 10788, "end": 15787}
}
```

Every script that takes IDs accepts `--member <name> --members-file <path>`, or
explicit `--ids ...`, or `--year-month YYMM --start N --end M`.

## 3. End-to-End Orchestrator & Hugging Face Dataset Export

`src/milestone_pipeline.py` runs Milestones 1–3 (or any selected subset of stages)
and exports the six sharded Parquet tables (`papers`, `references`, `elements`,
`hierarchy`, `bibtex`, `predictions`) together with the YAML Dataset Card
(`README.md`) ready for Hugging Face Hub:

```bash
# Run Milestones 1 + 2 + export 5 tables for one member (before manual labels are ready)
python -m src.milestone_pipeline \
  --member tan --members-file members.json \
  --source-dir src/sourcesArxiv \
  --dataset-dir data/huggingface_dataset \
  --skip-match

# Export all 6 sharded Parquet tables + README.md from already processed directories
python -m src.milestone_pipeline \
  --export-only \
  --source-dir src/sourcesArxiv sourcesManual20 \
  --matching-output-dir data/matching \
  --dataset-dir data/huggingface_dataset \
  --shard-rows 200000

# Full end-to-end run (crawl -> parse -> match -> export)
python -m src.milestone_pipeline \
  --year-month 2407 --start 17839 --end 17858 \
  --manual-labels data/manual_labeling/manual_labels.csv \
  --members-file members.json --num-members 4 \
  --source-dir data/pipeline/sources \
  --work-dir data/pipeline/work \
  --dataset-dir data/huggingface_dataset
```

Each milestone exports its sharded Parquet tables immediately upon stage
completion (`papers` and arXiv-keyed `references` after Milestone 1; `elements`,
`hierarchy`, and `bibtex` after Milestone 2; `predictions` and the dataset card
after Milestone 3), following the Hugging Face Hub config-per-table layout:

```text
data/huggingface_dataset/
├── README.md
└── data/
    ├── papers/part-00000.parquet
    ├── references/part-00000.parquet
    ├── elements/part-00000.parquet ...
    ├── hierarchy/part-00000.parquet ...
    ├── bibtex/part-00000.parquet
    └── predictions/part-00000.parquet
```

## 4. Milestone 1: Crawl

```bash
# arXiv API discovery (recommended for assigned ID ranges)
python src/arxiv_crawler.py --member tan --members-file members.json --out src/sourcesArxiv

# Same range via OAI-PMH discovery
python src/oai_crawler.py --member tan --members-file members.json --out src/sourcesOAI

# OAI-PMH date-window harvest (repository-scale demonstration)
python src/oai_crawler.py harvest --from 2025-10-01 --until 2025-10-02 --set cs:cs:CL --limit 50
```

What the arXiv crawler does:

1. resolves the IDs with batched `id_list` API calls (100 IDs/call) and reads the latest version `vN`;
2. asks the API once more for the intermediate versioned IDs `v2..vN-1` (batched) to get the date of every version;
3. downloads every version from `https://export.arxiv.org/src/<id>vK` (3 s pacing counted from request start, descriptive User-Agent) and extracts only `.tex` / `.bib` files so figures never reach disk;
4. fetches the references once per paper from Semantic Scholar (`paperId, externalIds, title, authors, year, publicationDate, venue`) in a background prefetcher thread.

Output layout:

```text
src/sourcesArxiv/
├── crawl_log.json                 # requested/resolved IDs, missing IDs, statuses
└── 2407.17839/
    ├── v1/
    │   ├── metadata.json          # paper metadata + byte sizes + status fields
    │   ├── references.json        # every Semantic Scholar reference
    │   └── *.tex, *.bib           # original relative paths inside the archive
    └── v2/ ...
```

Interrupted runs resume automatically: versions whose `metadata.json` already
records `downloaded` or `no_tex_or_bib` are skipped, and missing Semantic Scholar
references are backfilled on re-run.

### Crawler benchmark

Measure each crawler in its own process so the second one does not inherit the
memory footprint of the first:

```bash
cd src
python benchmark.py --crawler arxiv --clean --member tan --members-file ../members.json
python benchmark.py --crawler oai   --clean --member tan --members-file ../members.json
```

Outputs are written to `data/benchmarks/benchmark_results.csv` and
`data/benchmarks/benchmark_timeseries.csv`.

## 5. Milestone 2: Parse, Normalize & Deduplicate

```bash
python -m src.parser.pipeline --source-dir src/sourcesArxiv
```

Stages per publication: multi-file gathering → hierarchy construction →
LaTeX cleanup and math normalization → reference extraction and deduplication →
citation key renaming across `\cite{...}` commands → full-text deduplication across versions.

Options of `src.parser.pipeline`:

| Option | Effect |
| --- | --- |
| `--stage {all,normalize,dedup}` | Run the full pipeline (`all`, default) or re-run only normalization or deduplication |
| `--limit N` | Process only the first N publications |
| `--clean-tex` | Delete original `.tex` files afterwards (off by default so Milestone 1 sources stay reusable) |
| `--trace-memory` | Also measure peak heap with `tracemalloc` (slows execution; do not enable during wall-time benchmarking) |
| `--report-md`, `--stats-json` | Paths for benchmark summary files (default under `data/benchmarks/`) |
| `--dedup-summary-json` | Optional path to write detailed deduplication statistics JSON |

Version statuses in `elements/compilation_manifest.json`:

- `success`: every included `.tex` file on the compilation path was found;
- `partial`: root and document body were parsed, with missing auxiliary `\input` recorded as a warning;
- `empty_version`: version has no TeX sources (PDF-only), preserved as an explicit empty record;
- `failed`: no root file or no document body could be parsed.

Output per publication:

```text
2407.17839/
├── elements/
│   ├── elements.json / elements.parquet      # `elements` table, deduplicated across versions
│   ├── hierarchy.json / hierarchy.parquet    # `hierarchy` table, one row per node-parent edge per version
│   ├── hierarchy_tree_vN.json                # nested tree of each version
│   ├── full_inlined_vN.tex                   # compilation path inlined into one file
│   ├── compilation_manifest.json             # root, included files, orphans, status
│   ├── raw_references.json                   # bibliography blocks set aside for extraction
│   └── dedup_stats.json                      # counts before/after deduplication
└── bibtex/
    ├── bibtex.json / bibtex.csv / bibtex.parquet   # unified `bibtex` table
    ├── bibtex.bib                                   # canonical BibTeX records
    └── citation_key_mapping.json                    # per-version old key -> canonical key
```

Stage-only re-runs on existing parsed outputs:

```bash
python -m src.parser.pipeline --source-dir src/sourcesArxiv --stage normalize
python -m src.parser.pipeline --source-dir src/sourcesArxiv --stage dedup --dedup-summary-json data/benchmarks/dedup_summary.json
```

## 6. Milestone 3: Reference Matching Pipeline

### End-to-end matching

```bash
python -m src.model.run_matching \
  --source-dir src/sourcesArxiv sourcesManual20 \
  --manual-labels data/manual_labeling/manual_labels.csv \
  --members-file members.json --num-members 4 \
  --output-dir data/matching
```

Steps: text cleaning (`data_cleaning`) → automatic labeling on a random pool of
at least $\max(\lceil 10\% \text{ of non-manual publications} \rceil, 2k)$ balanced
across members (`auto_labeler`), plus precision/recall validation on the manual pool →
partition split ($k$ manual + $k$ auto publications in `test` and in `valid`,
remaining in `train`) → pairwise candidate matrix with 24 engineered features
(`feature_engineering`) → ranking models selected on manual `valid` and scored with
MRR@5 on `test` (`modeling`) → `predictions` table.

### Manual labeling workflow

The manual ground truth covers $5k = 20$ publications ($k = 4$ members, 5 per
assigned range), with every BibTeX entry of each selected publication labeled
($\ge 20$ entries per publication, $\ge 400$ entries total):

```bash
python -m src.model.manual_labeling crawl-team20 --out sourcesManual20 --papers-per-member 5
python -m src.model.manual_labeling prepare --source-dir sourcesManual20 --output-dir data/manual_labeling
# Inspect and confirm every row in data/manual_labeling/manual_review_ui.html (mark reviewed = yes), then:
python -m src.model.manual_labeling finalize \
  --review-csv data/manual_labeling/manual_review_sheet.csv \
  --references-csv data/manual_labeling/references_cleaned.csv \
  --output-dir data/manual_labeling \
  --min-publications 20 --min-entries-per-pub 20 --min-total-entries 400
```

### Label Documentation

Internal label files (`data/manual_labeling/manual_labels.csv` and merged `data/matching/labels.csv`):

| Column | Type | Description |
| --- | --- | --- |
| `arxiv_id` | `string` | Base arXiv ID of the citing publication (versionless, e.g. `2407.17850`) |
| `citation_key` | `string` | Canonical BibTeX `citation_key` from Milestone 2 (`bibtex` table) |
| `groundtruth_arxiv_id` | `string` (nullable) | Matched `ref_arxiv_id` within the same publication, or empty when no crawled arXiv reference corresponds to the entry |
| `is_positive` | `int` (`0`/`1`) | `1` when `groundtruth_arxiv_id` is non-empty, `0` for an all-negative row (manual file) |
| `label_source` | `string` | `"manual"` (human-verified ground truth) or `"auto"` (high-precision rule/heuristic matcher) |

In the $m \times n$ candidate matrix (`candidate_matrix.parquet`), each labeled BibTeX
entry generates $n$ candidate pair rows `(arxiv_id, citation_key, candidate_ref_id)`
against the $n$ arXiv-keyed crawled references of that publication: `label = 1` for
the single matching `groundtruth_arxiv_id` (when present) and `label = 0` for all
other candidates; unlabeled entries generate no training/evaluation label rows.

### Feature Documentation (24 Engineered Features)

`FeatureExtractor` (`src/model/feature_engineering.py`) frames reference matching as
within-publication candidate ranking over `(BibTeX entry, crawled arXiv reference)`
pairs and extracts **24 interpretable features** across **6 groups**:

| Group | Feature Name | Range / Type | Underlying Idea & Justification |
| --- | --- | --- | --- |
| **1. Title Similarity (8)** | `sim_title_exact` | `{0, 1}` | Exact equality after lowercasing, LaTeX stripping, and punctuation removal; captures unchanged titles between preprint and bibliography. |
| | `sim_title_ratio` | `[0, 1]` | Normalized edit-distance similarity (`fuzz.ratio / 100`); tolerates minor typos, hyphenation, or LaTeX accent artifacts. |
| | `sim_title_token_sort` | `[0, 1]` | Edit similarity after sorting words alphabetically; handles reordered clauses or subtitle swaps. |
| | `sim_title_token_set` | `[0, 1]` | Set-based fuzzy similarity (`fuzz.token_set_ratio / 100`); robust when a journal version appends or drops a subtitle relative to the arXiv preprint. |
| | `sim_title_jaccard_words` | `[0, 1]` | Word-level Jaccard overlap $|W_{\text{bib}} \cap W_{\text{ref}}| / |W_{\text{bib}} \cup W_{\text{ref}}|$; measures shared vocabulary independent of word order. |
| | `sim_title_char_3gram` | `[0, 1]` | Character 3-gram Jaccard overlap; resilient to OCR/bibitem concatenation errors, compound words, and spelling variants. |
| | `feat_title_length_diff` | `[0, ∞)` | Absolute character-length difference; penalizes short generic titles matching long unrelated titles via token containment. |
| | `feat_title_word_count_diff` | `[0, ∞)` | Absolute difference in unique word counts; complements `sim_title_token_set` to prevent false positives on partial title prefixes. |
| **2. Author Overlap (5)** | `sim_author_exact` | `{0, 1}` | Exact match of normalized `firstinitial_lastname` author lists in order. |
| | `sim_author_jaccard` | `[0, 1]` | Jaccard similarity over normalized author tokens; handles truncated `et al.` lists or minor author-list edits across versions. |
| | `feat_author_first_match` | `{0, 1}` | Exact surname match of the first author; critical signal when `\bibitem` entries omit titles (physics/math style `Author, Journal (Year)`) or abbreviate co-authors. |
| | `feat_author_count_diff` | `[0, ∞)` | Absolute difference in author counts; distinguishes solo papers from large collaborations sharing a lead author. |
| | `feat_author_missing` | `{0, 1}` | Indicator (`1` when either side lacks parsed authors); allows tree/linear models to down-weight zero author scores caused by missing fields rather than true mismatch. |
| **3. Publication Year (3)** | `feat_year_exact` | `{0, 1}` | Exact publication year match (`diff == 0`). |
| | `feat_year_diff` | `[0, 20]` | Clipped absolute year gap $\min(|y_{\text{bib}} - y_{\text{ref}}|, 20)$ (defaults to `10` when missing); captures typical 1–2 year lag between arXiv v1 submission and conference/journal publication while penalizing distant years. |
| | `feat_year_missing` | `{0, 1}` | Indicator (`1` when either year is missing); separates missing metadata from genuine year disagreement. |
| **4. Explicit Identifiers (2)** | `feat_id_exact_match` | `{0, 1}` | `1` when an arXiv ID extracted from BibTeX (`eprint`, `url`, `journal`, or `note`, supporting both post-2007 `YYMM.NNNNN` and pre-2007 `archive/YYMMNNN`) matches `ref_arxiv_id`; near-deterministic positive signal. |
| | `feat_has_bib_arxiv_id` | `{0, 1}` | `1` when the BibTeX entry explicitly states an arXiv ID; combined with `feat_id_exact_match == 0`, strongly penalizes candidates with a *different* arXiv ID. |
| **5. Milestone 2 Hierarchy Context (3)** | `feat_citation_depth` | `[0, ∞)` | Minimum tree depth in the Milestone 2 `hierarchy` table of elements citing `\cite{key}` (`root=0`, `section=1`, `subsection=2`, ...); reflects structural placement of the citation. |
| | `feat_citation_frequency` | `[0, ∞)` | Number of distinct deduplicated `elements` citing `key`; core methodological papers cited repeatedly are far more likely to be CS/ML preprints indexed with arXiv IDs. |
| | `feat_is_in_intro_or_related` | `{0, 1}` | `1` if cited under an *Introduction / Related Work / Background* heading in the Milestone 2 hierarchy tree. |
| **6. Candidate-Group Relative (3)** | `feat_title_sim_rank` | `[1, n]` | Within-entry rank of this candidate sorted by `sim_title_ratio` descending (`1` = best title match among the $n$ candidates of the entry); injects listwise competition into pointwise classifiers. |
| | `feat_title_sim_diff_to_max` | `[-1, 0]` | Margin `sim_title_ratio - max_j(sim_title_ratio)` within the same BibTeX entry; separates clear winners (`0.0` with runner-up `-0.5`) from ambiguous ties. |
| | `feat_candidate_count` | `[1, ∞)` | Total number of arXiv-keyed candidate references $n$ in the publication; calibrates match probability against candidate-pool size. |

### Modeling & Evaluation Summary

- **Models (`src/model/modeling.py`):** Title-similarity baseline, Logistic Regression, Histogram Gradient Boosting, and Random Forest (`scikit-learn` only — no pretrained LLMs or external embeddings).
- **Post-processing:** Direct arXiv-ID match override + Hungarian one-to-one assignment within each publication (retained only when improving validation MRR@5).
- **Model selection:** Chosen strictly on the **manual** subset of the `valid` partition to avoid optimistic bias from heuristic `auto` labels.
- **Evaluation metric:** $\text{MRR@5} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$, reported both overall (`mrr@5`, where entries with no arXiv match score `0`) and on matchable entries (`mrr@5_matchable`), broken down by `all`, `manual`, and `auto`.

## 7. Tests

```bash
pytest -q
```

## 8. Crawler Comparison

| Aspect | `arxiv_crawler.py` | `oai_crawler.py` |
| --- | --- | --- |
| Discovery | `arxiv.Search(id_list=...)`, 100 IDs per call | `GetRecord` per ID; `ListRecords` for harvests |
| Version dates | one extra batched call on intermediate versioned IDs | included directly in `arXivRaw` |
| Discovery requests for 50 papers | 1–2 | 50 (3 s pacing each) |
| Download, extraction, S2 references | shared `crawler_common.py` | shared `crawler_common.py` |
| Best for | assigned ID lists or ranges | date-window or category harvests |
