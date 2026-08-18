# Contributing

## Setup

```bash
git clone https://github.com/xiaoxiaoshikui/context-compiler.git
cd context-compiler
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[all]'            # core + mcp + tiktoken + llm_judge extras
```

The core package has zero required dependencies. Only install the extras
you're actually touching if you'd rather keep the environment minimal —
`pip install -e .` alone is enough for everything except the optional MCP
server, exact `tiktoken` counting, and the LLM-judge benchmark evaluator.

## Running tests

```bash
python -m compileall src benchmarks tests
python -m unittest discover -s tests -v
```

No test dependencies are required. CI (`.github/workflows/tests.yml`)
runs both across Python 3.10-3.13 on every push and PR to `main`.

## Before opening a PR

- Tests pass locally, including any new ones for the behavior you added
  or fixed. A bug fix without a regression test is easy to reintroduce —
  see `tests/test_store.py`'s determinism tests for the shape this should
  take (assert the *specific* thing that broke, not just "it works now").
- `python -m compileall` is clean.
- If you touched anything under `src/context_compiler/`, run the
  benchmark and look at the diff:

  ```bash
  python benchmarks/run.py
  git diff benchmarks/REPORT.md benchmarks/results/results.json
  ```

  This project's rule for algorithmic changes (scoring, retrieval,
  allocation, dependency resolution) is: **show the before/after, not
  just the after.** Every non-trivial change in `CHANGELOG.md` has a
  paired benchmark comparison in `benchmarks/README.md` — a regression on
  one task with an improvement on three others is a fine trade to make
  explicitly; a change with no comparison at all is not reviewable.

## Adding a benchmark task

Tasks live in `benchmarks/tasks.py` as a `BenchTask` pointing at a small
synthetic repo under `benchmarks/repos/<slug>/`. Look at an existing task
of the shape you want (`benchmarks/README.md` documents the five current
shapes: constraint, config-lookup, code-behavior, decision-record,
dependency-graph) before inventing a new one. Keep the repo small (a
handful of files), give it exactly one fact-carrying "oracle" file, and
choose `required_substrings` that are verifiably unique to that file
(`grep -ril` your candidate phrases across the repo before committing —
several existing tasks were tuned this way after a phrase turned out to
also appear in a distractor file or get split across a markdown line
wrap).

## Honesty norm

This repo's documentation is deliberately candid about what's proven vs.
not — see `README.md`'s "Current limitations" and every "Honest
limitations" / "Real findings" section in `benchmarks/README.md`. If your
change has a caveat (small sample size, a synthetic benchmark that can't
isolate the effect you're claiming, a metric that's misleading given
class imbalance), write the caveat down next to the claim. A modest,
well-scoped result that's reproducible is worth more here than an
impressive-sounding one that isn't.

## Code style

- No comments explaining *what* code does — names should do that. Comment
  only the non-obvious *why* (a constraint, an invariant, a workaround).
- Don't add abstractions, config flags, or error handling for scenarios
  that can't happen. Match the scope of a change to what it's actually
  fixing.
- New optional capabilities (a new evaluator, a new tokenizer backend)
  follow the existing extras pattern in `pyproject.toml`
  (`[project.optional-dependencies]`) rather than becoming a required
  dependency of the core package.
