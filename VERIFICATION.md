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
