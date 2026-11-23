# NMKHDL_Nhom11

Course project on arXiv: crawling full-text LaTeX sources (Milestone 1), parsing
them into a deduplicated hierarchy with unified BibTeX entries (Milestone 2),
and matching references (Milestone 3).

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

**Assigned ID ranges.** Each member crawls their own range. Copy
`members.example.json` to `members.json` and fill in the real ranges:

```json
{
  "tan": {"year_month": "2510", "start": 10788, "end": 10807},
  "hoang": {"year_month": "2510", "start": 10808, "end": 10827},
  "kiet": {"ids": ["2510.10828", "2510.10829"]}
}
```

Every script that takes IDs accepts `--member <name> --members-file <path>`, or
explicit `--ids ...`, or `--year-month YYMM --start N --end M`.

## 3. Milestone 1: crawl

Run the commands from `src/`.

```bash
cd src
# arXiv API discovery (recommended)
python arxiv_crawler.py --member tan --members-file ../members.json --out sourcesArxiv
# same range, OAI-PMH discovery
python oai_crawler.py --member tan --members-file ../members.json --out sourcesOAI
# OAI-PMH date-window harvest (repository-scale example, not needed for grading)
python oai_crawler.py harvest --from 2025-10-01 --until 2025-10-02 --set cs:cs:CL --limit 50
```

What the arXiv crawler does:

1. resolves the IDs with batched `id_list` API calls and reads the latest version `vN`;
2. asks the API once more for the versioned IDs `v2..vN-1` (batched) to get the date of every version;
3. downloads every version from `https://export.arxiv.org/src/<id>vK` (3 s pacing,
   descriptive User-Agent) and extracts only `.tex` / `.bib` files, so figures never
   reach the disk;
4. fetches the references once per paper from Semantic Scholar
   (`paperId, externalIds, title, authors, year, publicationDate, venue`).

Output layout:

```text
sourcesArxiv/
├── crawl_log.json                 # requested/resolved IDs, missing IDs, statuses
└── 2510.10788/
    ├── v1/
    │   ├── metadata.json          # see below
    │   ├── references.json        # every Semantic Scholar reference (with or without arXiv ID)
    │   └── *.tex, *.bib           # original paths inside the archive are kept
    └── v2/ ...
```

`metadata.json` holds the columns of the `papers` table plus crawl details:
`arxiv_id` (versioned), `base_id`, `version`, `title`, `authors`, `submission_date`
(v1), `version_date`, `revised_dates` (v2..vN), `venue` (journal reference),
`source_status` (`downloaded`, `no_tex_or_bib` for PDF-only/non-TeX versions,
`download_failed`), `source_format`, the byte sizes before/after removing figures,
`references_status` (`success`, `not_indexed`, `failed`), `references_count` and
`arxiv_references_count`.

Interrupted runs resume: versions whose `metadata.json` says `downloaded` or
`no_tex_or_bib` are skipped.

### Crawler benchmark

Measure each crawler in its own process so the second one does not inherit the
memory of the first:

```bash
python benchmark.py --crawler arxiv --clean --member tan --members-file ../members.json
python benchmark.py --crawler oai   --clean --member tan --members-file ../members.json
```

Results go to `benchmark_results.csv` (one row per crawler) and
`benchmark_timeseries.csv` (RAM sampled every 0.5 s plus the time of every paper).
Commit these files exactly as the script writes them, and record `du -sh <output>`
next to them.

## 4. Milestone 2: parse

```bash
python -m src.parser.pipeline --source-dir src/sourcesArxiv
```

Stages, per publication: multi-file gathering → hierarchy construction →
cleanup and math normalization → reference extraction and deduplication →
citation renaming → full-text deduplication across versions.

Useful options:

| Option | Effect |
| --- | --- |
| `--limit N` | process only the first N publications |
| `--clean-tex` | delete the original `.tex` files afterwards (off by default: the Milestone 1 corpus stays usable for re-runs) |
| `--trace-memory` | also measure peak heap with `tracemalloc` (slows the run; do not use for timing) |
| `--report-md`, `--stats-json` | where to write the run summary |

Version statuses in `elements/compilation_manifest.json`:

- `success`: every included file was found;
- `partial`: the root and the document body were found but an included `.tex`
  file is missing; the version is parsed and the warning is recorded;
- `empty_version`: the version has no TeX sources (PDF-only), kept as an empty record;
- `failed`: no root file or no document body.

`\input` targets that are never TeX sources (`.tikz`, `.pgf`, `.bbl`, data files)
are skipped without affecting the status.

Output per publication:

```text
2510.10788/
├── elements/
│   ├── elements.json / elements.parquet      # `elements` table, deduplicated across versions
│   ├── hierarchy.json / hierarchy.parquet    # `hierarchy` table, one row per edge and version
│   ├── hierarchy_tree_vN.json                # nested tree of each version
│   ├── full_inlined_vN.tex                   # compilation path inlined into one file
│   ├── compilation_manifest.json             # root, included files, orphans, status
│   ├── raw_references.json                   # bibliography blocks set aside by the parser
│   └── dedup_stats.json                      # counts before/after deduplication
└── bibtex/
    ├── bibtex.json / bibtex.csv / bibtex.parquet   # `bibtex` table
    ├── bibtex.bib                                   # the same entries as BibTeX records
    └── citation_key_mapping.json                    # per-version old key -> canonical key
```

BibTeX notes: `.bib` files are chosen from `\bibliography{...}` when present, and
`.bib` entries that the version never cites are left out (counted as
`uncited_bib_entries_dropped`). `\bibitem` entries are converted with regular
expressions, including the `\bibinfo{field}{value}` form; text that cannot be
split into fields is kept in the BibTeX `note` rather than guessed as a title.

Stage-only runners, for re-running one step on existing outputs:

```bash
python src/run_normalizer.py --source-dir src/sourcesArxiv
python src/run_dedup.py --source-dir src/sourcesArxiv --summary-json dedup_summary.json
```

## 5. Milestone 3: reference matching

### End-to-end pipeline

```bash
python -m src.model.run_matching \
  --source-dir src/sourcesArxiv \
  --manual-labels data/manual_labeling/manual_labels.csv \
  --members-file members.json --num-members 4 \
  --output-dir data/matching
```

Steps: cleaning (`data_cleaning`) → automatic labels on a random pool of at
least max(ceil(10%) of the non-manual publications, 2k), balanced across members
(`auto_labeler`) → split with k manual + k auto publications in test and in valid
→ candidate matrix with citation-context features from `elements/`
(`feature_engineering`) → models, selection on valid, evaluation on test
(`modeling`) → `predictions` table.

Labels (internal format, `data/matching/labels.csv`): one row per labeled BibTeX
entry with `arxiv_id`, `citation_key`, `groundtruth_arxiv_id` (empty when the
entry has no true reference among the crawled references) and `label_source`
(`manual` or `auto`).

Modeling: ranking of the candidate references of each BibTeX entry with
pointwise scorers: a title-similarity baseline, logistic regression, histogram
gradient boosting and random forest (scikit-learn, no pretrained model).
Post-processing (arXiv-ID override, one-to-one assignment) is kept only when it
improves validation MRR. Selection uses the **manual** part of the validation
set, since automatic labels come from the same similarities as the features.

Metrics: `mrr@5` counts every labeled entry (entries without a true reference
score 0, the literal reading of the statement); `mrr@5_matchable` counts only
entries with a true reference. Both are reported for all, manual and auto entries.

Outputs in `--output-dir`: `labels.csv`, `split.json`, `candidate_matrix.parquet`,
`model_comparison.csv`, `feature_importance.csv` (permutation importance on
test), `ablation.csv` (feature groups on valid), `metrics.json` (including the
precision/recall of the automatic labels on the manual pool),
`references_table.parquet` and `predictions.parquet`
(`arxiv_id, partition, citation_key, groundtruth_arxiv_id, cand_1..cand_5`).

### Cleaning only

```bash
python -m src.model.data_cleaning
```

If the inputs are not under the default folders, pass them explicitly:

```bash
python -m src.model.data_cleaning \
  --bibtex "path/to/bibtex.csv" \
  --references "path/to/references.json" \
  --references-source-dir "src/sourcesArxiv" \
  --output-dir "data/cleaned" \
  --format parquet
```

### Manual labeling

The manual ground truth covers 20 publications, 5 from each assigned ID range,
with every BibTeX entry of each publication labeled (at least 20 entries per
publication, at least 400 in total):

```bash
python -m src.model.manual_labeling crawl-team20 --out sourcesManual20
python -m src.model.manual_labeling prepare --source-dir sourcesManual20 --output-dir data/manual_labeling
# review every row in manual_review_ui.html (or the CSV), then:
python -m src.model.manual_labeling finalize \
  --review-csv data/manual_labeling/manual_review_sheet.csv \
  --references-csv data/manual_labeling/references_cleaned.csv \
  --output-dir data/manual_labeling \
  --min-publications 20 --min-entries-per-pub 20 --min-total-entries 400
```

`prepare` ranks the candidate references of every entry and pre-fills a
suggestion; a person then checks each row, sets `groundtruth_arxiv_id` (empty
when no crawled reference matches) and marks the row `reviewed = yes`.
`finalize` refuses rows that were not reviewed and reports how many suggestions
the reviewer changed (`manual_labeling_stats.json`).

Label schema under `data/manual_labeling/`:
- `manual_labels.csv` / `manual_labels.json`: one row per labeled BibTeX entry
  with `arxiv_id`, `citation_key`, `groundtruth_arxiv_id` (matched
  `ref_arxiv_id`, or empty when unmatched), `is_positive` (1 or 0) and
  `label_source` (`manual`). This file is the `--manual-labels` input of
  `run_matching`.
- `manual_candidate_pairs.csv`: the within-publication m × n binary candidate
  matrix with `arxiv_id`, `citation_key`, `ref_arxiv_id`, `label`, `label_source`.

## 6. Tests

```bash
pytest -q
```

## 7. Crawler comparison

| Aspect | `arxiv_crawler.py` | `oai_crawler.py` |
| --- | --- | --- |
| Discovery | `arxiv.Search(id_list=...)`, 100 IDs per call | `GetRecord` per ID; `ListRecords` for harvests |
| Version dates | one extra batched call on versioned IDs | included in `arXivRaw` |
| Requests for 50 papers | 1–2 | 50 (3 s pacing each) |
| Download, extraction, references | shared `crawler_common.py` | shared `crawler_common.py` |
| Best for | a known list or range of IDs | date or category harvests |
