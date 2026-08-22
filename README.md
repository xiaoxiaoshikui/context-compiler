# Minimum Context Compiler

[![tests](https://github.com/xiaoxiaoshikui/context-compiler/actions/workflows/tests.yml/badge.svg)](https://github.com/xiaoxiaoshikui/context-compiler/actions/workflows/tests.yml)
[![version](https://img.shields.io/badge/version-0.2.0-informational.svg)](CHANGELOG.md)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![core dependencies](https://img.shields.io/badge/core%20dependencies-zero-brightgreen.svg)](#quick-start)

A runnable reference implementation of the **Minimum Sufficient Context** idea:

> Given a task and a hard context budget, choose the smallest/highest-utility representation of available information that is most likely to preserve task performance.

This repository is a research/runtime substrate for that idea. The core works with Python's standard library only. Raw context is stored losslessly in SQLite, while the model-facing representation can be lossy and progressively expandable.

## Highlights

- **Zero required dependencies.** The core is pure Python + stdlib SQLite; `pip install -e .` and go.
- **Lossless storage, lossy display.** Raw content is never discarded — every summary the model sees is reversible via `expand(CTX_ID)`.
- **Five resolution levels per item** (pointer → outline → summary → excerpts → full text), chosen per item to fit a hard token budget via a marginal-utility-per-token allocator.
- **Pluggable retrieval, scoring, and dependency resolution** — swap the default TF-IDF retriever, the hand-tuned scoring weights, or use the real cross-file dependency graph, each behind a small documented interface.
- **CLI, an HTTP API with no dependencies, and an optional MCP v2 server** ship in the same package.
- **A benchmark harness with reproducible before/after numbers** for every algorithmic change in this repo's history — including a real determinism bug the benchmarking work found and fixed. See [Benchmarks](#benchmarks).
- **Validated against real SWE-bench Verified instances with real Docker-verified test execution** — not a keyword-matching proxy. The results are reported honestly, including where this compiler currently falls short of an oracle ceiling. See [Real-execution evaluation](#benchmarks).

## 30-second demo

```bash
pip install -e .
ctxc ingest ./examples/demo_repo
ctxc compile "Fix the Safari OAuth callback regression; preserve the no-replay security invariant" --budget 400
```

```text
used=382/400 items=5 candidates=5

[CTX:c999b6e2f2ab0418 L4 kind=constraint source=SECURITY_CONSTRAINTS.md]
SECURITY_CONSTRAINTS.md
# OAuth safety constraints

Authorization codes are single-use credentials. Never automatically replay an
OAuth authorization-code exchange after an ambiguous network failure. A replay
can violate provider guarantees and may create inconsistent login state.

Safari callback handling must preserve the same single-exchange invariant as
every other browser.

[CTX:c476817ef26a34b3 L4 kind=test source=test_auth.py]
test_auth.py
from auth import handle_callback


def test_safari_callback_uses_single_exchange():
    result = handle_callback("Safari", "abc", "state-1")
    assert result["token"].startswith("session:")

... plus auth.py, oauth_client.py, and README.md at full fidelity, all within budget.
```

Every file that's safety-critical or directly relevant to the Safari bug made
it into the working set at full fidelity, using 382 of a 400-token budget —
see [Quick start](#quick-start) for the full walkthrough (search, expand,
pinning a constraint by hand), or [Architecture](#architecture) for how the
budget allocator decides what to include.

## Contents

- [Highlights](#highlights)
- [30-second demo](#30-second-demo)
- [What is implemented](#what-is-implemented)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Exact-ish token counting](#exact-ish-token-counting)
- [MCP server](#mcp-server)
- [HTTP API](#http-api)
- [Python API](#python-api)
- [Experiments](#experiments)
- [Run the included demo](#run-the-included-demo)
- [Run tests](#run-tests)
- [Benchmarks](#benchmarks)
- [Current limitations](#current-limitations)
- [Recommended next research iterations](#recommended-next-research-iterations)
- [Contributing](#contributing)

## What is implemented

- Lossless SQLite `ContextStore`
- Repository/file ingestion
- Basic dependency extraction for Python / JS / TS / Go
- Pluggable first-stage retrieval (`context_compiler.retrieval.Retriever`):
  TF-IDF cosine similarity by default, the original SQLite FTS5 query
  still available as an opt-in alternative
- Task-aware candidate ranking
- Importance / omission-risk / recency / dependency scoring
- Five context resolutions:
  - `L0`: pointer
  - `L1`: metadata / symbol outline
  - `L2`: extractive summary
  - `L3`: task-relevant excerpts
  - `L4`: full raw content
- Hard token-budget allocator using marginal utility per token
- Pinned / constraint minimum representations
- Reversible `expand(CTX_ID)`
- CLI
- Zero-dependency local JSON HTTP API
- Optional MCP v2 server
- Budget-sweep, deletion-test and greedy-minimalization experiment primitives
- Optional tiktoken counter
- Unit tests and a runnable demo repository

## Architecture

```text
Files / docs / decisions / constraints / conversations
                       |
                       v
               +----------------+
               | Context Store  |  <-- raw information remains lossless
               |    SQLite      |
               +-------+--------+
                       |
             candidate retrieval
                       |
                       v
               +----------------+
Task --------> | Context Scorer |
               +-------+--------+
                       |
        relevance / importance / risk /
        recency / dependency / pinning
                       |
                       v
               +----------------+
               | Multi-resolution|
               |    Renderer     |
               +-------+--------+
                       |
            L0 L1 L2 L3 L4 variants
                       |
                       v
               +----------------+
Budget ------> |   Allocator    |
               +-------+--------+
                       |
                       v
              Compiled Working Set
                       |
                       v
                      LLM
```

The central optimization implemented here is approximately:

```text
maximize sum_i score(i) * fidelity(level_i)
subject to sum_i tokens(level_i) <= budget
```

with forced minimum fidelity for pinned/constraint items, followed by greedy marginal utility-per-token upgrades.

## Quick start

Requires Python 3.10+.

```bash
cd context_compiler_mvp
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Initialize and ingest a repository:

```bash
ctxc init
ctxc ingest ./examples/demo_repo
```

Compile a 700-token working set:

```bash
ctxc compile "Fix the Safari OAuth login regression without unsafe retries" --budget 700
```

Inspect ranked candidates:

```bash
ctxc search "Safari OAuth login regression" --limit 10
```

Expand an item exactly when more detail is needed:

```bash
ctxc expand <CTX_ID> --level L4
```

Store a hard constraint manually:

```bash
ctxc add \
  --title "OAuth retry invariant" \
  --kind constraint \
  --pinned \
  --risk 1.0 \
  --importance 1.0 \
  --content "Never automatically replay a non-idempotent authorization exchange."
```

## Exact-ish token counting

The default counter is dependency-free and intended for budgeting experiments, not billing reconciliation. For OpenAI-family tokenization experiments:

```bash
pip install -e '.[exact]'
ctxc --tokenizer tiktoken compile "..." --budget 8000
```

The tokenizer is intentionally an adapter: replace it with a model-specific counter when measuring a specific model.

## MCP server

Install the optional MCP dependency:

```bash
pip install -e '.[mcp]'
```

Run as stdio MCP server:

```bash
ctxc-mcp --db .context-compiler.db --transport stdio
```

Or through the main CLI:

```bash
ctxc --db .context-compiler.db mcp --transport stdio
```

The server exposes:

- `context_add`
- `context_ingest`
- `context_compile`
- `context_search`
- `context_expand`
- `context_stats`
- resource `context://item/{item_id}`

The intended host behavior is:

1. call `context_compile(task, budget)`;
2. work from the compiled set;
3. call `context_expand(id, level)` only if the current representation is insufficient.

That is the progressive-disclosure / virtual-memory loop.

## HTTP API

No dependencies are needed:

```bash
ctxc serve --host 127.0.0.1 --port 8765
```

Endpoints:

```text
GET  /health
GET  /stats
GET  /items?limit=100
GET  /items/{id}
GET  /expand/{id}?level=L3&task=...
POST /items
POST /search
POST /compile
```

Example:

```bash
curl -s http://127.0.0.1:8765/compile \
  -H 'content-type: application/json' \
  -d '{"task":"Fix Safari OAuth login", "budget":700}'
```

## Python API

```python
from context_compiler import ContextCompiler, ContextStore

store = ContextStore("experiment.db")
store.add(
    title="Payment invariant",
    content="Never retry a non-idempotent charge automatically.",
    kind="constraint",
    pinned=True,
    importance=1.0,
    omission_risk=1.0,
)

compiler = ContextCompiler(store)
result = compiler.compile(
    "Fix duplicate payment attempts",
    budget=2000,
)

print(result.text)
print(result.used_tokens)
print(result.selections)
```

## Experiments

`context_compiler.experiments` contains three core tools.

### 1. Budget sweep

```python
from context_compiler.experiments import EvaluationResult, ExperimentRunner


def evaluator(task: str, context: str) -> EvaluationResult:
    # Replace this with an actual agent + tests.
    ok = "never retry" in context.lower() and "oauth" in context.lower()
    return EvaluationResult(success=ok, score=float(ok))

runner = ExperimentRunner(compiler, evaluator)
points = runner.budget_sweep(
    "Fix Safari OAuth login",
    [128, 256, 512, 1024, 2048],
    repeats=5,
)
```

This estimates the task's context-performance frontier.

### 2. Deletion test

```python
compiled = compiler.compile("Fix Safari OAuth login", 2000)
effects = runner.deletion_test("Fix Safari OAuth login", compiled, repeats=5)
```

For selected item `c_i`, this estimates an empirical marginal effect:

```text
Delta_i = quality(C) - quality(C - c_i)
```

### 3. Greedy empirical minimalization

```python
result = runner.greedy_minimize(
    "Fix Safari OAuth login",
    compiled,
    min_success_rate=0.95,
    repeats=5,
)
```

This is expensive by design. It repeatedly proposes deletions and keeps removals that preserve the target success rate. Use it on a small benchmark set to estimate empirical Minimum Sufficient Context.

### Connecting a real coding agent

`JsonCommandEvaluator` lets the experiment runner call any external agent/test harness without coupling this package to one vendor SDK.

```python
from context_compiler.experiments import JsonCommandEvaluator

evaluator = JsonCommandEvaluator(["python", "my_agent_eval.py"])
```

The child process receives:

```text
CTXC_TASK_FILE
CTXC_CONTEXT_FILE
```

and must print one JSON object:

```json
{"success": true, "score": 1.0, "metadata": {"tests": "passed"}}
```

This is the recommended bridge for SWE-bench-style evaluation: your external harness owns repository checkout, agent execution and test evaluation; Context Compiler owns context selection and measurement.

## Run the included demo

```bash
rm -f demo.db
ctxc --db demo.db ingest examples/demo_repo
ctxc --db demo.db compile \
  "Fix the Safari OAuth callback regression; preserve the no-replay security invariant" \
  --budget 650 \
  --json

python examples/run_experiment.py
```

## Run tests

No test dependency is required:

```bash
python -m unittest discover -s tests -v
```

## Benchmarks

Two benchmarks live in this repo, deliberately kept distinct: a fast
synthetic suite for iterating on the algorithm (proxy evaluator, free,
runs in CI-friendly seconds), and a slow real-execution suite for
answering "does any of this actually work" (real model, real Docker
-verified tests, costs real money). Neither substitutes for the other —
see `benchmarks/README.md` for the full methodology behind both.

### Real-execution evaluation: does minimum sufficient context exist?

`benchmarks/real_eval.py` downloads real SWE-bench Verified instances,
generates a real patch with a real model, and checks it against the
instance's real hidden tests via the official `swebench` Docker harness —
execution-based pass/fail, not a keyword-check proxy.

**Experiment 1 — an Oracle budget sweep** (context limited to the files
the gold patch actually touches, never the patch itself, across an
increasing token budget) asks the most basic question first: is there
even a token budget past which more context stops helping?

![Oracle budget sweep: instances resolved per budget](benchmarks/charts/real_eval_oracle_sweep.svg)

| Budget | Resolved (of 6) |
|---|---|
| 1,000 tokens | 2 |
| 4,000 tokens | 4 |
| 8,000 tokens | 5 |
| 16,000 tokens | 4 |

Yes — on this sample, resolution rate stops improving past ~8,000 tokens.
That's real evidence for the hypothesis this whole repo is built on, not
just an assumption.

**Experiment 2 — at that same 8,000-token budget**, does `ours` (this
compiler's actual TF-IDF selection + L0-L4 compression) get anywhere
close to the oracle ceiling, compared to a random-file-ordering floor?

![Oracle ceiling vs Random floor vs Ours at 8,000 tokens](benchmarks/charts/real_eval_method_comparison.svg)

| Policy | Run 1 | Run 2 |
|---|---|---|
| **Oracle** (ceiling) | 5 / 6 | — |
| **Random** (floor) | 0 / 6 | 0 / 6 |
| **Ours**, across 4 fix stages | 0/6, 0/6, 1/6, 0/6 | 1/6, 2/6, 0/6, 0/6 |

Reported exactly as measured, including the parts that don't flatter this
project: `ours` started out barely distinguishable from randomly
ordering the repository — nowhere near the oracle ceiling, and after
five real, verified bug fixes (see below) it is *still* nowhere near the
ceiling. Across 32 real run-instances spanning every fix stage, exactly
4 resolved — 3 of them the same instance (`flask`). The most recent
stage, with every fix applied including two that directly and verifiably
raised two other instances' render fidelity from a bare pointer to a
real excerpt, resolved 0 of 6 in both repeats. That is the honest,
current result, not a rounding error being smoothed over: getting the
right file *visible at high fidelity* turned out to be necessary but not
sufficient — `requests` had its fix file rendered in full since before
any of these fixes existed and has only ever resolved once in 8 runs;
`pytest` never resolved despite the same visibility throughout every
run. That points the actual bottleneck at this sample size somewhere
downstream of context construction — most likely patch-synthesis
fidelity or raw model capability on these specific fixes, not context
selection — which is itself a useful, if unflattering, diagnostic
conclusion. Read the run-to-run noise in the table above as a real,
current limitation, not a fluke to average away — n=6 tasks / n=2
repeats is nowhere near enough to call any of this a precise, stable
gap. This is the honest answer to "is this useful" rather than the
weaker "did it find the right file" proxy the file-selection spot check
below measures. See `benchmarks/README.md`'s "2026-08-22 continued"
section for the full bug-by-bug writeup, the complete per-instance
aggregate table across all 4 fix stages, two real patch-corruption bugs
found and fixed along the way (compressed context breaks exact-match
patch snippets; the fix was whole-function rewrite via `ast`, not text
matching), a pretraining-memorization confound investigated and ruled
out as a full invalidation, and the infrastructure failure modes hit
running it (concurrent runs silently broke DNS resolution; a reasoning
model can burn its whole token budget "thinking" about incoherent
context and answer nothing at all).

```bash
python3 -m venv .venv_realeval && source .venv_realeval/bin/activate
pip install -e . pyarrow "swebench>=3,<4"   # + Docker running, DEEPSEEK_API_KEY set
python benchmarks/real_eval.py --instances pallets__flask-5014 ... \
  --policies oracle --budgets 1000 4000 8000 16000 --repeats 1 --tag mysweep
```

### Synthetic benchmark: fast, free, proxy-scored

15 synthetic tasks across five task shapes, each compiled under a budget
sweep with `ours` against two naive baselines (`full`, `random`) plus an
Oracle token-cost reference point. Success is a keyword-conjunction check
against the compiled text — "is the right fact present," not "did a
patch pass" — which is exactly the gap the real-execution evaluation
above closes. An optional `--evaluator llm_judge` mode (real Anthropic
API calls, opt-in, costs money) is a step past pure keyword matching,
`benchmarks/learn_weights.py` fits an alternative scoring-weights preset
from deletion-test labels (`run.py --weights learned`), and
`context_compiler.graph` resolves real cross-file dependency edges that
feed back into scoring (`run.py --graph off` for comparison).

```bash
python benchmarks/run.py
```

![B95 by task and method — ours vs full vs random](benchmarks/charts/overview_b95.svg)

Chart generated by `python benchmarks/plot_results.py` (plain SVG, no
charting dependency) from the committed result files — `benchmarks/README.md`
has per-phase before/after charts for every comparison mentioned above.

See `benchmarks/README.md` for exactly what is and is not being measured —
it is explicitly a starting point (15 tasks, mostly a keyword-check
evaluator), not the 30-100 task benchmark recommended below. It also
documents a real nondeterminism bug in `compile()` that this benchmarking
work found and fixed (nonstable candidate ordering plus a token-count
side effect of randomly-generated item ids) — see "Phase 4a" there, and
a separate real SWE-bench file-selection spot check (5/6 oracle-file
hit rate before the real-execution study above existed) that found and
fixed two more real bugs — see "two bugs a real SWE-bench Verified
instance found" there.

## Current limitations

This package intentionally makes the research hypothesis inspectable. It does **not** pretend that the current heuristic scorer solves context sufficiency.

Current limitations:

1. Candidate retrieval is lexical (TF-IDF cosine similarity by default
   since `context_compiler.retrieval`), not embedding-based — still no
   stemming or synonym match. *Improved:* the original SQLite FTS5
   `candidate_search` had a stronger failure mode than "worse ranking" —
   an item with zero exact FTS term overlap was dropped from the
   candidate pool entirely, at any budget (`benchmarks/README.md` "Real
   findings" #2 has a reproducible case). The new default `TfidfRetriever`
   scores every candidate instead of filtering by exact match first, so
   it degrades to weaker ranking rather than silently losing items — the
   old FTS retriever is still available as `FTSRetriever` /
   `--retriever fts` for comparison. Real embedding-based retrieval is
   still open; anything implementing the `Retriever` protocol can replace
   `TfidfRetriever` without touching `ContextCompiler`.
2. The scorer is hand-weighted rather than learned from task outcomes.
   *Started:* `benchmarks/learn_weights.py` fits an alternative preset
   (`LEARNED_WEIGHTS_V1` in `context_compiler.scoring`, not the default)
   from deletion-test labels — real but modest results on 14 tasks; see
   `benchmarks/README.md` "Phase 2" for the honest numbers.
3. Dependency extraction is shallow; it is not a full symbol/call graph.
   *Improved:* `context_compiler.graph` resolves import strings into real
   forward/reverse edges between stored items (Python relative/absolute
   imports, JS/TS relative imports; best-effort elsewhere) and feeds
   confirmed graph neighbors of top-scoring candidates back into scoring.
   Still shallow (import-level, not call-level; no `sys.path`/package-root
   modeling) — see `benchmarks/README.md` "Phase 4a".
4. L2 summarization is extractive and deterministic; no LLM is required.
5. The greedy allocator is a practical approximation, not a global optimizer.
6. No automatic post-step working-set update/eviction policy yet.
7. No learned sufficiency estimator yet.
8. No vendor-specific agent adapter is bundled; use `JsonCommandEvaluator` or MCP.

These are intentional extension points rather than hidden assumptions.

## Recommended next research iterations

1. **Build a benchmark harness** around 30-100 coding tasks. *Started:*
   `benchmarks/` has a 15-task version across five task shapes, comparing
   `ours` against `full` and `random` baselines with a keyword-check
   evaluator (plus an opt-in, paid LLM-judge evaluator) — see
   `benchmarks/README.md` for what it does and does not yet prove.
   *Also started, on real tasks:* `benchmarks/real_eval.py` runs real
   SWE-bench Verified instances end to end (real model, real Docker
   -verified tests) — 6 instances so far, not yet the 30-100 scale here.
2. Compare `Full`, `Random`, `Lexical/FTS`, `Embedding RAG`, `Ours`, and
   `Oracle` under equal budgets. *Started:* `benchmarks/run.py` now runs
   `Full`, `Random`, `Ours` (TF-IDF by default), and `Ours` with the
   original FTS retriever (`--retriever fts`) under equal budgets, plus
   the `Oracle` token-cost reference point. *Also started, with real
   execution instead of a proxy evaluator:* `benchmarks/real_eval.py`
   compares `Oracle`/`Random`/`Ours` with real Docker-verified pass/fail
   — see the finding above. `Embedding RAG` is attempted but not yet
   reliable in either harness; `Lexical/FTS` is `run.py --retriever fts`.
3. Record task success, lifecycle input tokens, context misses and latency.
   *Started:* `real_eval.py` records per-call token usage and real
   pass/fail; lifecycle-total tokens (across retries/tool calls) and
   context-miss tracking are still open.
4. Estimate `B95`: smallest budget reaching 95% of full-context baseline quality.
5. Run deletion tests on successful trajectories to generate labels for
   context marginal value. *Started:* see item 2 above.
6. Train/fit a context-value model from those labels. *Started:* see item
   2 above — a first pass (pure-Python logistic regression, no learned
   sufficiency *threshold* yet, just re-weighted feature importances).
7. Replace file-level dependency strings with a code/decision/entity graph.
   *Started:* `context_compiler.graph` resolves import strings into real
   forward/reverse edges between items and feeds them into scoring — see
   limitation #3 above. Still import-level, not a true call/entity graph.
8. Add online expansion/eviction based on agent actions.

See `DESIGN.md` for the algorithm and extension points.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, the benchmark
before/after convention this repo expects for algorithmic changes, and
how to add a benchmark task. [`CHANGELOG.md`](CHANGELOG.md) has the
release history.
