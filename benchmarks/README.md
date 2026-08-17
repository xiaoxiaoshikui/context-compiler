# Starter benchmark

A first, runnable slice of README.md's "Recommended next research
iterations" (items 1-3): a small benchmark harness that compares the
scored/allocated compiler (`ours`) against two naive baselines under a
budget sweep, plus a per-task Oracle reference point.

## Run it

```bash
python benchmarks/run.py
```

Writes `benchmarks/results/results.json` (raw sweep points) and
`benchmarks/REPORT.md` (human-readable table), and prints the report to
stdout.

## What is being measured

Each task in `tasks.py` points at a small synthetic repo under `repos/`
containing a handful of distractor files plus exactly one file carrying a
safety-critical fact (an operational policy the task cannot be solved
correctly without, e.g. "never retry a charge without a confirmed
idempotency key"). The evaluator is a deterministic, case/whitespace-
insensitive keyword-conjunction check against the compiled context text:
does the working set contain the invariant, at any budget.

Methods compared:

- `ours` — `ContextCompiler` (scored, multi-resolution, budget-allocated).
- `full` — dump every file at full fidelity in path-sorted order, stop at
  the first file that would overflow the budget. Models "just paste the
  repo".
- `random` — shuffle files, greedily keep whatever fits (20 repeats per
  budget to smooth the estimate). Models "grab an arbitrary subset".
- `oracle_tokens` — the smallest single-representation cost of the one
  file that alone satisfies the evaluator. Not a method under test; a
  lower bound on how cheap a perfectly-informed selector could be.

## Honest limitations of this slice

This is explicitly a **starter** harness, not the benchmark the README
roadmap describes:

1. **5 tasks, not 30-100.** Too small to draw statistically confident
   conclusions; treat results as illustrative, not definitive.
2. **The evaluator is a keyword check, not a real agent/test harness.**
   It answers "is the fact present in the working set", never "did an
   LLM produce a correct patch". Wire in `JsonCommandEvaluator` from
   `context_compiler.experiments` with a real coding agent to get an
   actual quality signal.
3. **Synthetic, hand-written repos.** They are sized and worded by the
   same person who tuned the scorer, which risks unconsciously favoring
   `ours`. A real benchmark should pull tasks/repos the compiler's
   author did not write (e.g. SWE-bench-style tasks).
4. **No `Embedding RAG` or true `Lexical/FTS`-only baseline yet** — only
   `full` and `random`, since those need no new dependencies. Embedding
   retrieval is still open (see top-level README limitation #1).

## A real finding this harness already surfaced

The first run showed `ours` losing to the naive `full` baseline at the
tightest budget (150 tokens) on two tasks (`db_migration`, `rate_limit`).
Root causes, both real:

- The compiler's forced-minimum level for *ingested* (non-manually-pinned)
  constraint items defaulted to `L1`, which only ever renders a fixed
  lead line — for a markdown policy doc structured as `# Title` followed
  by the actual invariant in the next paragraph, `L1` surfaced the title,
  not the invariant. Fixed by moving the default to `L2` (task-aware
  extractive summary) in `src/context_compiler/compiler.py`
  (`CompilerConfig.constraint_min_level`).
- Independent of that fix, `ours` carries a larger fixed header
  ("Use the following task-specific working set...") than the naive
  baselines' bare `TASK:` line, which costs real tokens at extremely
  tight budgets. This is a genuine trade-off, not a bug — left as-is and
  documented rather than tuned away.
