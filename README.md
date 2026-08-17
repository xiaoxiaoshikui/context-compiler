# Minimum Context Compiler

A runnable reference implementation of the **Minimum Sufficient Context** idea:

> Given a task and a hard context budget, choose the smallest/highest-utility representation of available information that is most likely to preserve task performance.

This repository is a research/runtime substrate for that idea. The core works with Python's standard library only. Raw context is stored losslessly in SQLite, while the model-facing representation can be lossy and progressively expandable.

## What is implemented

- Lossless SQLite `ContextStore`
- Repository/file ingestion
- Basic dependency extraction for Python / JS / TS / Go
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

`benchmarks/` contains a first, small slice of the benchmark harness
described below: 5 synthetic tasks, each compiled under a budget sweep with
`ours` (this compiler) against two naive baselines (`full`, `random`) plus
an Oracle reference point.

```bash
python benchmarks/run.py
```

See `benchmarks/README.md` for exactly what is and is not being measured —
it is explicitly a starting point (5 tasks, keyword-check evaluator), not
the 30-100 task benchmark recommended below.

## Current limitations

This package intentionally makes the research hypothesis inspectable. It does **not** pretend that the current heuristic scorer solves context sufficiency.

Current limitations:

1. Candidate retrieval is lexical/FTS, not embedding-based.
2. The scorer is hand-weighted rather than learned from task outcomes.
3. Dependency extraction is shallow; it is not a full symbol/call graph.
4. L2 summarization is extractive and deterministic; no LLM is required.
5. The greedy allocator is a practical approximation, not a global optimizer.
6. No automatic post-step working-set update/eviction policy yet.
7. No learned sufficiency estimator yet.
8. No vendor-specific agent adapter is bundled; use `JsonCommandEvaluator` or MCP.

These are intentional extension points rather than hidden assumptions.

## Recommended next research iterations

1. **Build a benchmark harness** around 30-100 coding tasks. *Started:*
   `benchmarks/` has a 5-task version comparing `ours` against `full` and
   `random` baselines with a keyword-check evaluator — see
   `benchmarks/README.md` for what it does and does not yet prove.
2. Compare `Full`, `Random`, `Lexical/FTS`, `Embedding RAG`, `Ours`, and `Oracle` under equal budgets.
3. Record task success, lifecycle input tokens, context misses and latency.
4. Estimate `B95`: smallest budget reaching 95% of full-context baseline quality.
5. Run deletion tests on successful trajectories to generate labels for context marginal value.
6. Train/fit a context-value model from those labels.
7. Replace file-level dependency strings with a code/decision/entity graph.
8. Add online expansion/eviction based on agent actions.

See `DESIGN.md` for the algorithm and extension points.
