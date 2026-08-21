# Verification

Validated in the build environment on 2026-08-17:

- `python -m compileall` on the source package: PASS
- `python -m unittest discover -s tests -v`: 5/5 PASS
- editable install with local build tooling / no dependencies: PASS
- CLI repository ingestion: PASS
- CLI hard-budget compilation: PASS (`385 / 650` heuristic tokens on demo before final constraint-classification tweak)
- experiment demo: PASS; toy evaluator produced a visible budget frontier and `B95=180`
- HTTP `/health` and `/compile`: PASS
- MCP adapter: source/API shape checked against current official MCP Python SDK v2 docs; runtime dependency is not installed in the sandbox, so end-to-end MCP transport was not executed here

The bundled demo/evaluator is synthetic and is only a plumbing test. It is not evidence of model-quality gains.

## 2026-08-17 (session 2): baseline benchmark harness

- Initialized git; first commit is the initial snapshot as delivered.
- Added `src/context_compiler/baselines.py` (`FullContextBaseline`,
  `RandomContextBaseline`) and `benchmarks/` (5 synthetic tasks, `run.py`
  budget-sweep driver, generated `REPORT.md`/`results/results.json`).
- `python -m unittest discover -s tests -v`: 7/7 PASS (5 previous + 2 new
  baseline tests).
- `python benchmarks/run.py`: PASS; produced a real (not just plumbing)
  finding — see `benchmarks/README.md`, "A real finding this harness
  already surfaced" — which led to changing
  `CompilerConfig.constraint_min_level` from `L1` to `L2` in
  `compiler.py`. Existing tests still pass after the change.
- As with the original demo evaluator, the benchmark's keyword-conjunction
  evaluator is a plumbing/proxy check, not evidence of real model-quality
  gains. See `benchmarks/README.md` for the full list of caveats.

## 2026-08-18 (session 3, "full development" Phase 1): CI, benchmark diversification, LLM judge

- Added `.github/workflows/tests.yml` (Python 3.10-3.13 matrix); pushing
  it required requesting the `workflow` OAuth scope for the `gh` CLI
  token (interactive device-flow authorization, completed by the user).
  First run: **all 4 matrix jobs green**
  (https://github.com/xiaoxiaoshikui/context-compiler/actions/runs/32111732215).
- Scaled `benchmarks/` from 5 tasks (one shape) to 14 tasks across four
  shapes (constraint/config-lookup/code-behavior/decision-record) — see
  `benchmarks/README.md`.
- Re-running the sweep on the expanded set surfaced a second real,
  currently open finding: on `cache_ttl`, `ours` never reaches 100%
  success at any swept budget up to 3000 tokens (SQLite FTS5 zero-recall
  on "cached" vs "cache.py"; the item never enters the candidate pool
  regardless of budget) — a direct, reproducible case of README
  limitation #1. Documented, not patched (Phase 3 scope).
- Added `benchmarks/llm_judge_eval.py` (official `anthropic` SDK, new
  optional `llm_judge` extra, `claude-haiku-4-5`), opt-in only
  (`--evaluator llm_judge --yes`), never part of the default sweep.
  Verified via mocked unit tests only; **no real API calls were made**.
- `python -m unittest discover -s tests -v`: 12/12 PASS (7 previous + 5
  new: 4 `test_llm_judge_eval.py`, plus the CI file itself doesn't add
  tests).

## 2026-08-18 (session 3, Phase 2): learned scoring weights

- Added `benchmarks/learn_weights.py`: runs `ExperimentRunner.deletion_test`
  on all 14 tasks (budget 3000) to generate (item, "was removing it fatal")
  labels, fits a pure-Python (no numpy/sklearn) L2-regularized logistic
  regression over the compiler's existing `ScoreBreakdown` components.
- Result: 60 examples (13 essential / 47 not) from the 14-task benchmark.
  Leave-one-task-out CV: classification accuracy (0.78) is barely above
  the majority-class baseline (0.78, uninformative given class
  imbalance); the metric that matches how the compiler actually uses
  these weights — ranking, not classifying — is top-1: 69% vs a 25%
  chance baseline (13 folds). `recency`/`pin_bonus` are near-constant in
  this benchmark (no item age variation, nothing manually pinned), so
  their fitted values are flooring artifacts; `coefficients_to_weights`
  detects this and keeps those two at the hand-tuned default instead.
- Shipped as `LEARNED_WEIGHTS_V1` in `context_compiler.scoring` —
  explicitly **not the default**; opt in via `ContextCompiler(store,
  weights=LEARNED_WEIGHTS_V1)`. Head-to-head via
  `python benchmarks/run.py --weights learned`: B95 improved on 3/14
  tasks, regressed on 0, unchanged on the rest.
- `python -m unittest discover -s tests -v`: 17/17 PASS (12 previous + 4
  new `test_learn_weights.py` + 1 new `test_compiler.py` case).
- As with every other evaluator in this repo, the labels come from the
  keyword-check proxy, not a real agent — see `benchmarks/README.md`
  "Phase 2" for the full, honest caveats before trusting this preset on
  anything that matters.

## 2026-08-18 (session 3, Phase 3): pluggable retrieval, TF-IDF default

- Added `src/context_compiler/retrieval.py`: a `Retriever` protocol,
  `TfidfRetriever` (new default -- scores every candidate handed to it,
  including zero-token-overlap ones, instead of hard-filtering by exact
  match first) and `FTSRetriever` (wraps the original
  `ContextStore.candidate_search` for comparison/opt-in). The
  pinned/constraint/high-risk safety net moved out of `candidate_search`
  into a retriever-agnostic `augment_with_critical_items` helper that
  `ContextCompiler` applies uniformly.
- `ContextCompiler.__init__` gains a `retriever` parameter (defaults to
  `TfidfRetriever()`); `compile()`/`search()` no longer call
  `store.candidate_search` directly -- `candidate_search` itself is
  untouched and still directly usable.
- Confirmed fix, by re-running the full 14-task sweep before/after:
  `cache_ttl` B95 went from unreachable (never 100% success up to 3000
  tokens) to `700`; the other 13 tasks are unchanged; 0 regressions.
  Reproducible via `python benchmarks/run.py --retriever fts
  --tasks cache_ttl` (old behavior) vs the new default. `benchmarks/run.py`
  gained `--retriever {tfidf,fts}`.
- `python -m unittest discover -s tests -v`: 28/28 PASS (17 previous + 11
  new `test_retrieval.py`).
- TF-IDF is still lexical (no stemming/synonyms) -- it narrows README
  limitation #1, does not eliminate it. Real embedding-based retrieval
  remains open and is a drop-in `Retriever` implementation away from
  being swapped in.

## 2026-08-18 (session 3, Phase 4a): dependency graph + a determinism bug it found

**Correctness fix (independent of the graph feature, affects every
`compile()` call):** verifying the graph feature required running the
same task+repo+budget repeatedly, which exposed that `compile()` was not
deterministic -- the identical repository, re-ingested into a fresh
store, could flip a budget-boundary result between runs. Two causes in
`store.py`, both fixed:
- `ContextStore.list()`'s `ORDER BY pinned DESC, updated_at DESC` had no
  tiebreak; items ingested in the same batch commonly share `updated_at`
  down to the stored precision, and SQLite doesn't guarantee stable order
  for ties. Fixed: `ORDER BY pinned DESC, updated_at DESC, source ASC, id
  ASC` (and the FTS5 `candidate_search` query gained the same `source`
  tiebreak on its `bm25()` ordering).
- Every rendered item embeds its id in a `[CTX:...]` header, so the id's
  token count is part of every budget decision. Ids were bare
  `uuid.uuid4().hex[:16]`; `HeuristicTokenCounter` tokenizes a hex string
  starting with a letter as one token but one starting with a digit as
  two, so the same content could cost a different token count purely by
  chance (~5/8 of raw hex ids start with a digit) across separate
  ingests. Fixed: ids are now `f"c{uuid.uuid4().hex[:15]}"`, always
  one token.
- Verified: `diff` on `results.json` from two independent full-suite
  `benchmarks/run.py` runs is empty (was flaky before the fix -- directly
  reproduced the flip on `payment_idempotency` at budget 150, ~50% of
  runs, before fixing; 10/10 stable after). New regression tests:
  `test_store.py::test_generated_ids_always_tokenize_as_a_single_token`,
  `test_store.py::test_list_order_is_deterministic_across_repeated_ingests`,
  `test_compiler.py::test_compile_is_deterministic_across_separate_ingests`.
- This retroactively affects how to read earlier B95 numbers in this
  file and in `benchmarks/REPORT.md`: swings spanning multiple budgets
  (e.g. Phase 3's `cache_ttl` fix) are far too large to be this bug; a
  single-budget-step difference on an already-close-to-threshold task
  might not be. Confirmed one such case retroactively: Phase 2's
  `payment_idempotency` B95 "improvement" (250 -> 150) does not reproduce
  on a clean re-run after this fix (150 both ways) -- the other two
  Phase 2 improvements do still hold.

**The feature:** added `src/context_compiler/graph.py` (`DependencyGraph`,
`build_dependency_graph`) resolving `ingest.py`'s import strings into real
forward/reverse edges between stored items (Python relative/absolute,
JS/TS relative; best-effort basename fallback elsewhere; ambiguous
matches left unresolved rather than guessed). Also fixed `ingest.py` to
capture Python relative-import level (previously discarded, so `from .
import x` was silently dropped entirely). `ContextCompiler` gains
`use_graph` (default `True`): a candidate within one resolved hop of a
top-scoring item gets its dependency score set to its ceiling in the
second scoring pass (`scoring.dependency_score`'s new `graph_related`
param), applying even to a leaf item with no outgoing edges of its own
(the edge can run the other way).

Added a purpose-built 15th benchmark task, `dependency_graph`: task
wording lexically matches `handler.py` but the actual constraint lives in
`validators.py`, worded with deliberately no shared vocabulary --
`validators.py` is only reachable via the resolved import edge.
`benchmarks/run.py` gained `--graph {on,off}`.

**Result, on a clean re-run after the determinism fix:** B95 improved on
3/15 tasks (`feature_flag_rollout` 400->250, `cache_ttl` 700->400,
`dependency_graph` 400->250), regressed on 0, unchanged on the rest.

- `python -m compileall` + `python -m unittest discover -s tests -v`:
  44/44 PASS (28 previous + 16 new: 13 `test_graph.py` resolution/BFS
  tests, 2 `test_store.py` determinism tests, 1 `test_compiler.py`
  determinism test).

## 2026-08-18 (session 3, Phase 5): production hardening

- Bumped version 0.1.0 -> 0.2.0 (`pyproject.toml`, `__init__.__version__`)
  -- real new features plus a correctness fix warrant more than a patch
  bump.
- `ctxc --version`.
- `cli.py`: split dispatch into `main()` (top-level `try`/`except`) +
  `_dispatch()`. Previously-uncaught `FileNotFoundError` (bad `ingest`
  path, missing `--content-file`), `ValueError` (`compile --budget 0`),
  and `RuntimeError` (missing optional `tiktoken`/`mcp` dependency) now
  print a one-line `error: ...` to stderr with a deliberate exit code
  (2 for bad input/not-found, 1 for environment/dependency issues)
  instead of a raw Python traceback. Verified manually against all four
  cases plus the new `tests/test_cli.py` (6 tests: `--version`, each
  error path, an add+compile happy path, and the existing
  delete-not-found convention).
- Added `CONTRIBUTING.md` (dev setup, the benchmark before/after
  convention this repo expects for algorithmic PRs, how to add a
  benchmark task, the honesty-in-docs norm, code style) and
  `CHANGELOG.md` (0.1.0 and 0.2.0, built from actual git history, not
  from memory).
- README overhaul: added a "Highlights" section and a "30-second demo"
  with real, verified command output (`ctxc ingest` + `ctxc compile`
  against `examples/demo_repo`) ahead of the detailed sections, a status
  badge row (CI, version, Python, license, zero core dependencies), and
  a full table of contents -- the file had grown past 400 lines across
  four sessions and was hard to navigate. Also fixed a leftover "MVP"
  reference in `LICENSE` that survived the earlier rebrand.
- `python -m compileall` + `python -m unittest discover -s tests -v`:
  50/50 PASS (44 previous + 6 new `test_cli.py`).

## 2026-08-18 (session 4): SVG charts, then two real bugs from SWE-bench Verified

- Added `benchmarks/plot_results.py` (pure-stdlib SVG grouped-bar-chart
  generator, no plotting dependency) and generated/embedded charts for
  the overview and Phases 2/3/4a into both READMEs. Building the Phase 2
  chart from live current data (rather than the frozen prose) surfaced
  that `LEARNED_WEIGHTS_V1` now regresses 2 tasks against the current
  full pipeline (its `dependency` weight predates the Phase 4a graph
  feature and now suppresses it) -- documented in `benchmarks/README.md`;
  the preset is opt-in only, so nothing shipping by default is affected.
- Verified the two cached HuggingFace datasets on this machine
  (`princeton-nlp/SWE-bench_Lite`, `princeton-nlp/SWE-bench_Verified`)
  are official and complete: row counts match the published spec exactly
  (Lite test=300/dev=23, Verified test=500), zero nulls in any key
  field, zero duplicate `instance_id`s, valid HF cache symlinks, real
  verifiable sample instances.
- Ad hoc spot check (not yet a committed benchmark path): pulled 6 real
  SWE-bench Verified instances, one per repo, downloaded each real repo
  at the issue's `base_commit` via the GitHub codeload archive API,
  ingested it, and checked whether `compile()` surfaces the files the
  gold patch actually touches. 5/6 hit the oracle file at every budget
  1000-16000; `pylint-dev/pylint-7080` hit 0/5, which traced to two real
  bugs neither reachable by this repo's own short synthetic tasks -- see
  `benchmarks/README.md` "2026-08-18: two bugs a real SWE-bench Verified
  instance found" for the full root-cause writeup. Fixed both in
  `context_compiler/compiler.py`:
  - The task header embeds the task string verbatim and, unlike stored
    items, never went through L0-L4 compression -- a verbose real task
    (a GitHub issue with a pasted CLI log, 6510 tokens for pylint-7080)
    could consume an entire small/medium budget on its own, leaving zero
    tokens for actual context. Fixed via `CompilerConfig.
    max_header_fraction` (default 0.3): the task text is now truncated
    (via the existing `TokenCounter.truncate()`, so the omission is
    marked, not silent) to at most that fraction of budget.
  - The default candidate-retrieval limit (`max_candidates=120`) was a
    flat constant, hard-truncating the TF-IDF-ranked pool *before*
    scoring, independent of budget size -- a file ranked 181st of 2971
    was therefore unreachable at *any* budget, confirmed by manually
    raising `max_candidates` past 181 to recover it. Fixed by scaling the
    default with budget: `max(config.max_candidates, budget //
    config.candidate_tokens_estimate)`, `candidate_tokens_estimate`
    (default 40) being the measured token cost of an L0 rendering.
  - Verified no regression: re-ran the existing 15-task synthetic
    benchmark before/after -- all 15 B95 numbers are byte-identical
    (small budgets never hit the new floor). Re-ran the same 6 SWE-bench
    instances after the fix: pylint-7080 now reaches the oracle file at
    budget 8000/16000 (still correctly misses at 1000-4000, which
    genuinely cannot fit it); the other 5 instances are unaffected.
- New regression tests: `test_compiler.py::
  test_verbose_task_text_does_not_starve_the_working_set`,
  `test_compiler.py::test_default_candidate_limit_scales_with_budget`.
- `python -m unittest discover -s tests -v`: 52/52 PASS (50 previous + 2
  new).
- Caveat carried into `benchmarks/README.md` verbatim: the 6-instance
  check was hand-picked (smallest patch per repo, not a random/
  representative sample) and run from a throwaway script, not a
  committed benchmark path -- "5/6 hit the oracle file" should be read
  as "not obviously broken on real data," not a validated success rate.

## 2026-08-21 (session 5): real-execution SWE-bench evaluation

- Added `benchmarks/real_eval.py`: a committed, re-runnable real-execution
  harness (the file-presence spot check above was explicitly future work
  until now). For a real SWE-bench Verified instance it downloads the
  repo at `base_commit`, ingests it, builds context under a budget via
  one of three policies (`oracle`/`random`/`ours`), generates a real
  patch with `deepseek-v4-pro`, and verifies it against the instance's
  real hidden tests via the official `swebench` (3.x) Docker harness.
  Requires a dedicated venv (`pyarrow`, `swebench>=3,<4`), Docker, and a
  `DEEPSEEK_API_KEY` -- none of this is a default/core dependency, same
  pattern as `llm_judge_eval.py`.
- Patch mechanism: whole-function rewrite, located deterministically via
  `ast` (by function name, not text similarity), not SEARCH/REPLACE.
  Two real corruption bugs were found and fixed on the way to this: a
  missing-trailing-newline splice bug that glued two statements onto one
  line (`SyntaxError`), and a fuzzy-match boundary bug that left an
  orphaned line mid-block (`IndentationError`) -- both root-caused to the
  same thing (an arbitrary code snippet has no guaranteed-safe boundary
  to fuzzy-match and splice in whitespace-sensitive code); whole-function
  boundaries via `ast` are unambiguous and eliminated both. Full
  root-cause writeup in `benchmarks/README.md`.
- Real findings, 6 tasks (flask/requests/pytest/pylint/astropy/django),
  real Docker-verified pass/fail:
  - Oracle budget sweep (ground-truth-relevant context, never the gold
    patch itself): 2/6 resolved at 1k tokens, 4/6 at 4k, 5/6 at 8k, 4/6
    at 16k (one run per budget; the 16k dip is read as single-run noise
    from `temperature=0.2`, not evidence of a non-monotonic curve) --
    real evidence a sufficient-context region exists for at least some
    real tasks, on top of the earlier file-presence-only result.
  - At the fixed 8k-token budget where the oracle curve plateaus: oracle
    5/6, random (floor) 0/6 both repeats, ours 0/6 then 1/6. Reported as
    measured -- `context_compiler`'s current selection is not yet
    meaningfully distinguishable from random ordering on this small
    real sample, far from the oracle ceiling. n=6 tasks / n=2 repeats is
    not enough for a precise gap, but the direction is real and is new
    information the earlier file-presence proxy could not have shown.
- Two infrastructure failure modes found and fixed, documented in
  `benchmarks/README.md` for future runs: (1) running policies
  concurrently (3 processes at once) broke DNS resolution entirely
  (`gaierror`) and defeated the request-level hard timeout, since a
  thread stuck in a non-cancelable `getaddrinfo()` call can't be
  cancelled by `ThreadPoolExecutor.result(timeout=...)` -- fixed by
  running conditions serially; (2) `deepseek-v4-pro` is a reasoning
  model that can spend an entire `max_tokens` budget "reasoning" about
  incoherent (`random`-policy) context and emit zero actual answer --
  confirmed directly (`reasoning_tokens: 12000`, `content: ""` at
  `max_tokens=12000`); fixed by raising `max_tokens` to 24000, not by
  waiting longer.
- Added `benchmarks/plot_real_eval.py` (reuses `plot_results.py`'s
  SVG bar-chart helper) -- `real_eval_oracle_sweep.svg` and
  `real_eval_method_comparison.svg`, embedded in `benchmarks/README.md`.
- Results committed at `benchmarks/results/real_eval_results.json`
  (8 conditions: the 4-point oracle sweep plus 2 repeats each of
  random/ours at 8k -- earlier attempts that hit the DNS-storm and
  reasoning-exhaustion failures above, before both were fixed, were
  removed rather than kept as misleading 0%-signal data).
- `python -m unittest discover -s tests -v`: 52/52 PASS (no core-package
  changes this session; `real_eval.py` is a benchmarks-only script and
  isn't covered by the core test suite, same as `llm_judge_eval.py`).
- Not done yet, noted honestly in `benchmarks/README.md`: a RAG
  (embedding-retrieval) baseline (attempted, blocked by the same
  DeepSeek reliability issues); repeated runs at a sample size large
  enough for real statistical confidence; Experiment 4 (per-item
  deletion testing to find each task's empirical MSC).
