# Unified parser pipeline tasks

## 1. Define and preserve pipeline contracts
- [x] Keep `PublicationDeduplicator` in `src/parser/dedup.py` as the sole
  reference/citation/full-text deduplication implementation.
- [x] Preserve per-version inputs expected by dedup: pre-dedup elements,
  hierarchy edges, and `raw_references.json`.
- [x] Distinguish successful, empty, and failed versions throughout manifests
  and benchmark results.

## 2. Consolidate orchestration in `src/parser/pipeline.py`
- [x] Implement the ordered stages: gatherer → hierarchy builder →
  hierarchy normalizer → publication deduplication.
- [x] Persist inlined TeX, compilation manifests, per-version trees, and final
  JSON/CSV/Parquet artifacts supported by existing components.
- [x] Run source cleanup only after all required outputs are safely persisted.
- [x] Retain reusable pipeline methods for stage-only runner commands.

## 3. Adapt compatible runners and documentation
- [x] Route `run_parser.py` through the complete pipeline.
- [x] Keep `run_hierarchy.py`, `run_normalizer.py`, and `run_dedup.py` as
  backward-compatible wrappers around pipeline stages.
- [x] Document the canonical end-to-end command and stage-only commands.

## 4. Test the integrated workflow
- [x] Verify transitive includes, orphan exclusion, and sentence/list hierarchy.
- [x] Verify BibTeX deduplication, citation rewriting, and final normalized trees.
- [x] Verify empty/failed-version outputs and manifest states.
- [x] Run focused tests and the full suite; distinguish existing failures from
  regressions.

## Validation notes

- The focused integration/regression suite passes: 26 tests.
- All four runner `--help` commands work.
- Full suite: 50 tests, 43 pass, 6 errors, 1 failure. The remaining cases expect
  missing legacy `HierarchyBuilder` reference APIs, legacy per-version
  `references_vN.bib` files not emitted by the selected `dedup.py` schema, or
  permissive handling of dangling hierarchy edges. These behaviors are not used
  by the unified pipeline.
- The existing `run_dedup.py --summary-json` option is still advertised but its
  summary-writing block is absent in the current worktree; it was left unchanged
  rather than restoring unrelated output behavior.
