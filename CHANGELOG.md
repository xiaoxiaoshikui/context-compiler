# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-08-18

### Added

- GitHub Actions CI (Python 3.10-3.13 matrix).
- `benchmarks/`: a 15-task benchmark harness across five task shapes
  (constraint, config-lookup, code-behavior, decision-record,
  dependency-graph), comparing the compiler against naive `full`/`random`
  baselines and an oracle token-cost reference, with reproducible
  before/after numbers for every change below. See `benchmarks/README.md`.
- `benchmarks/llm_judge_eval.py`: an opt-in LLM-judge evaluator
  (`--evaluator llm_judge --yes`) using the official `anthropic` SDK
  behind a new `llm_judge` extra. Never runs as part of the default sweep.
- `benchmarks/learn_weights.py`: fits `LEARNED_WEIGHTS_V1`
  (`context_compiler.scoring`), an alternative scoring-weights preset
  learned from deletion-test labels. Not the default; opt in via
  `ContextCompiler(store, weights=LEARNED_WEIGHTS_V1)`.
- `context_compiler.retrieval`: a pluggable `Retriever` protocol.
  `TfidfRetriever` (new default) scores every candidate instead of
  hard-filtering by exact token match first; `FTSRetriever` wraps the
  original SQLite FTS5 `candidate_search` for comparison/opt-in
  (`benchmarks/run.py --retriever fts`).
- `context_compiler.graph`: resolves import strings into real
  forward/reverse dependency edges between stored items, fed back into
  scoring (`ContextCompiler(..., use_graph=True)`, the default;
  `--graph off` for comparison).
- `ctxc --version`.
- `CONTRIBUTING.md`.

### Changed

- Dropped "MVP" framing from docs and package metadata — the project is
  no longer described as a stripped-down proof of concept.
- `CompilerConfig.constraint_min_level` default changed `L1` -> `L2`: `L1`
  only ever renders a fixed lead line, which could miss a constraint
  doc's actual invariant if it wasn't the file's first line.
- `ingest.py`'s Python dependency extraction now captures relative-import
  level (previously discarded, so `from . import x` was silently dropped
  entirely).
- CLI errors that previously surfaced as raw Python tracebacks (a missing
  `ingest` path, `compile --budget 0`, a missing optional
  tokenizer/MCP dependency) now print a one-line `error: ...` message
  with a clean, deliberate nonzero exit code.

### Fixed

- **`ContextCompiler.compile()` was not deterministic.** The identical
  repository, re-ingested into a fresh store, could occasionally flip a
  budget-boundary result between runs. Two compounding causes, both in
  `store.py`:
  - `ContextStore.list()`'s `ORDER BY` had no tiebreak for items sharing
    the same `updated_at` (the common case for a freshly-ingested repo),
    and SQLite does not guarantee stable row order for ties.
  - Randomly-generated item ids (embedded in every rendered `[CTX:...]`
    header) tokenized as one or two tokens depending on whether the
    random hex happened to start with a digit, shifting reported token
    counts by +-1 at random — occasionally enough to flip which side of a
    tight budget an item landed on.

  See `benchmarks/README.md` "Phase 4a" for how this was found (while
  verifying the dependency-graph feature above) and confirmed fixed
  (`diff` on two independent full-suite benchmark runs is now empty).

## [0.1.0] - 2026-08-17

Initial release. SQLite-backed lossless `ContextStore`; five-resolution
renderer (`L0`-`L4`); task-aware scoring (relevance, importance, omission
risk, recency, dependency); a greedy marginal-utility-per-token budget
allocator; pinned/constraint minimum representations; reversible
`expand(CTX_ID)`; CLI; a zero-dependency local JSON HTTP API; an optional
MCP v2 server; budget-sweep, deletion-test, and greedy-minimalization
experiment primitives; an optional `tiktoken` counter.
