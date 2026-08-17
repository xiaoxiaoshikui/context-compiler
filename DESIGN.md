# Design Notes: Minimum Sufficient Context

## Problem statement

Let:

- `T` be a task,
- `M` a model/agent policy,
- `U` the available information universe,
- `B` a hard context budget,
- `Q(T, M, C)` the task quality under selected context `C`.

There are two useful formulations.

### Fixed budget

```text
maximize Q(T, M, C)
subject to tokens(C) <= B
```

### Fixed quality

```text
minimize tokens(C)
subject to Q(T, M, C) >= Q_target
```

The second is the Minimum Sufficient Context formulation.

## Why this implementation uses representations, not just chunks

Each stored item has multiple renderings:

```text
L0 pointer
L1 metadata / outline
L2 summary
L3 relevant excerpts
L4 full raw content
```

The choice is therefore not only:

```text
include(item) / exclude(item)
```

but:

```text
choose representation(item, task, budget)
```

This matters because the same fact may only need a pointer during discovery but full raw evidence during execution.

## Store losslessly, present lossily

`ContextStore` always keeps raw content. Lossy transformations are ephemeral views.

This makes compression reversible:

```text
pointer -> summary -> excerpts -> raw
```

The model/agent can request a higher-resolution view by CTX id if the current working set is insufficient.

## Scoring model in v0.1

The current hand-tuned score is:

```text
score =
  w_r * relevance
+ w_i * importance
+ w_o * omission_risk
+ w_t * recency
+ w_d * dependency
+ w_k * kind_prior
+ w_p * pin_bonus
```

This is deliberately transparent. The long-term goal is to learn the marginal value of context from actual task outcomes.

## Budget allocator

For each item, rendering levels have increasing cost and fidelity.

Conceptually, moving from level `l` to `l+1` is an upgrade:

```text
marginal_utility = item_score * (fidelity[l+1] - fidelity[l])
marginal_cost    = tokens[l+1] - tokens[l]
```

The allocator:

1. reserves budget requested by caller;
2. forces minimum levels for pinned / constraint items where possible;
3. repeatedly selects the feasible upgrade with best:

```text
marginal_utility / marginal_cost
```

4. stops when no positive upgrade fits.

This resembles a multiple-choice knapsack solved by a greedy approximation.

## What should become learned

The most important future model is not a summarizer. It is a **Context Sufficiency / Marginal Value Estimator**.

Desired prediction target for item `c_i`:

```text
Delta_i = E[Q(C) - Q(C \ c_i)]
```

Deletion experiments provide direct but expensive labels. Cheaper proxy labels can come from:

- whether the agent later expands/re-fetches an omitted item;
- whether a missing item appears in a successful oracle set;
- whether adding the item changes the proposed plan/patch;
- test-result improvement after expansion;
- judge-model estimates, calibrated against executable outcomes.

## Context graph roadmap

The current MVP stores flat items plus dependency strings. A stronger runtime should represent:

```text
Task
  -> symbols
  -> files
  -> tests
  -> API contracts
  -> decisions
  -> people/owners
  -> external tools
  -> safety constraints
```

Retrieval should answer:

> What information does completion of this task depend on?

rather than only:

> What text is semantically similar to this query?

## Online runtime loop

A production Context Runtime should eventually implement:

```text
1. Understand next agent action
2. Retrieve candidate information
3. Estimate omission risk / marginal value
4. Allocate budget
5. Render at adaptive resolution
6. Execute agent step
7. Observe tool calls / uncertainty / failures
8. Expand or fetch misses
9. Update working set
10. Evict stale/low-value context
```

The current package implements steps 2-5 plus explicit expansion, and provides the primitives needed to study 7-10.

## Benchmark design

For each task:

```text
budgets = [1k, 2k, 4k, 8k, 16k, ...]
methods = [Random, Recency, RAG, Ours, Oracle]
repeats >= 3
```

Measure:

- task success / executable test result;
- total lifecycle input tokens;
- context misses / expansions;
- latency;
- cost;
- selected item count and resolution mix.

The primary research curve is the context-performance frontier.

Define:

```text
B95 = smallest budget where quality >= 0.95 * full-context baseline quality
```

The goal of the compiler is to move the quality/token frontier toward the oracle frontier.
