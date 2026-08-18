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

Filter or override the sweep without editing the script:

```bash
python benchmarks/run.py --tasks oauth_safari,cache_ttl --budgets 150,700,2000
```

## What is being measured

Each task in `tasks.py` points at a small synthetic repo under `repos/`
containing a handful of distractor files plus exactly one file carrying the
fact the task cannot be solved correctly without. Tasks come in four
shapes, on purpose — an earlier version of this harness was 100%
"constraint doc," which structurally favored `ours` (constraints get a
forced minimum render level) and never exercised the scorer's actual
relevance ranking:

- **constraint** (5 tasks) — a policy/constraint doc must survive into the
  working set, e.g. "never retry a charge without a confirmed idempotency
  key." Gets the compiler's forced-minimum-level safety net.
- **config-lookup** (3 tasks) — the fact is a specific value in a config
  file (`.env`/`.yaml`), no policy doc involved. No safety net.
- **code-behavior** (3 tasks) — the fact is in a function's actual
  default/logic (a cache TTL, a retry count), not a doc. No safety net.
- **decision-record** (3 tasks) — a doc explaining "we chose X over Y, for
  reason Z"; the correct action is to respect the decision. No safety net.

14 tasks total. The evaluator is a deterministic, case/whitespace-
insensitive keyword-conjunction check against the compiled context text:
does the working set contain the fact, at any budget.

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

## A second evaluator: LLM judge (opt-in, costs real money)

The default keyword evaluator answers "is the fact present in the working
set," never "did an LLM decide the context was sufficient." A first step
past pure keyword matching:

```bash
pip install -e '.[llm_judge]'
export ANTHROPIC_API_KEY=...
python benchmarks/run.py --evaluator llm_judge --tasks oauth_safari --budgets 700 --yes
```

`benchmarks/llm_judge_eval.py` sends the compiled context and task to
`claude-haiku-4-5` and asks it to judge sufficiency — a real model call,
not a keyword check. It is **not** part of the default sweep (that stays
free and instant) and refuses to run without `--yes`, since cost scales
with tasks × budgets × methods. It is still a judge over text, not a real
coding agent that produces and tests a patch — see "Honest limitations"
below.

## Phase 2: learned scoring weights (opt-in, not the default)

```bash
python benchmarks/learn_weights.py
```

Runs `ExperimentRunner.deletion_test` against every task's compiled
working set (budget 3000, generous enough that nearly every candidate is
selected), pairing each selected item's raw `ScoreBreakdown` components
(relevance, importance, risk, recency, dependency, kind_prior, pin_bonus)
with a label: did removing this item break the task. That is exactly the
"direct but expensive" labeling strategy DESIGN.md describes for a future
Context Sufficiency / Marginal Value Estimator. A pure-Python (no
numpy/sklearn — confirmed unavailable in this environment) L2-regularized
logistic regression fits those labels, and the result is written to
`benchmarks/results/learned_weights.json`.

**Results, honestly:** 60 examples from 14 tasks (13 "essential" / 47
not). Leave-one-task-out CV: classification accuracy 0.78 is barely above
the 0.78 majority-class baseline (always guessing "not essential") — not
informative given the class imbalance. The metric that actually matches
how the compiler uses these weights — to *rank* candidates, not classify
them — is whether the top-ranked item in a held-out task is the essential
one: **69%, against a 25% chance baseline**. That's a real signal on a
small sample, not proof. Two features (`recency`, `pin_bonus`) are
near-constant across every example in this benchmark (every item has the
same age; nothing is manually pinned), so their fitted coefficients are
floor-value artifacts, not learned signal — `coefficients_to_weights`
detects this and keeps those two at their hand-tuned default instead of
using the artifact.

The fitted result ships as `LEARNED_WEIGHTS_V1` in
`context_compiler.scoring` — **not the default**, opt in explicitly:

```python
from context_compiler import ContextCompiler, ContextStore
from context_compiler.scoring import LEARNED_WEIGHTS_V1

compiler = ContextCompiler(store, weights=LEARNED_WEIGHTS_V1)
```

or from this benchmark's own sweep, to compare directly against the
default:

```bash
python benchmarks/run.py --weights learned
```

Doing that head-to-head comparison across all 14 tasks: **B95 improved on
3/14 tasks (`payment_idempotency`, `feature_flag_rollout`,
`retry_backoff`), regressed on none, unchanged on the rest** (`cache_ttl`
still fails at every budget either way — reweighting scores among
candidates that already passed FTS retrieval can't recover an item FTS
never let into the pool in the first place; see finding #2 below). A
small, real, honestly-bounded improvement — not remotely enough evidence
to make this the shipped default on 14 self-authored synthetic tasks
scored by a keyword-check proxy, which is exactly why it isn't.

## Honest limitations of this slice

This is explicitly a **starter** harness, not the benchmark the README
roadmap describes:

1. **14 tasks, not 30-100.** Larger and more varied than the first version
   (5 tasks, one shape), but still too small to draw statistically
   confident conclusions; treat results as illustrative, not definitive.
2. **Two evaluators, neither is a real agent/test harness.** The default
   keyword check answers "is the fact present." `llm_judge` is a step
   closer — an actual model judgment — but it's still a judge over
   *compiled text*, not a coding agent that produces a patch and runs the
   task's real tests. Wire in `JsonCommandEvaluator` from
   `context_compiler.experiments` with a real coding-agent harness to get
   that signal.
3. **Synthetic, hand-written repos.** They are sized and worded by the
   same person who tuned the scorer, which risks unconsciously favoring
   `ours`. A real benchmark should pull tasks/repos the compiler's
   author did not write (e.g. SWE-bench-style tasks).
4. **No `Embedding RAG` or true `Lexical/FTS`-only baseline yet** — only
   `full` and `random`, since those need no new dependencies. Embedding
   retrieval is still open (see top-level README limitation #1).

## Real findings this harness has already surfaced

**1. A forced-minimum-level gap (fixed).** The first run showed `ours`
losing to the naive `full` baseline at the tightest budget (150 tokens) on
two constraint tasks (`db_migration`, `rate_limit`). Root causes, both
real:

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

**2. Lexical retrieval can drop a relevant item entirely, at any budget
(open, not fixed).** On the `cache_ttl` task (task wording: "cached
prices"; the fact lives in `cache.py`, whose content never uses the word
"cached"), `ours` never reaches 100% success even at the largest swept
budget (3000 tokens), while `full` and `random` succeed from budget 400
onward. Cause: `ContextCompiler.compile` first calls
`ContextStore.candidate_search`, which runs a SQLite FTS5 query against
the task's literal terms — no stemming, no synonym match. If a relevant
item scores zero FTS hits and isn't pinned/constraint/high-risk, it never
enters the candidate pool the allocator sees, so **no budget, however
large, can recover it**. This is a direct, reproducible demonstration of
the top-level README's limitation #1 ("Candidate retrieval is
lexical/FTS, not embedding-based") — left unfixed here since retrieval
architecture is explicitly out of scope for this pass; it's the strongest
concrete argument yet for prioritizing that work.
