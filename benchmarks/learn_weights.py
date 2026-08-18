"""Fit learned ScoringWeights from deletion-test labels on the benchmark.

A first, small pass at README.md's "Recommended next research iterations"
items 5-6: run deletion tests on the benchmark's compiled working sets to
generate direct labels for each item's marginal value ("did removing this
item break the task"), then fit a small linear model over the same raw
score components ContextCompiler already computes (relevance, importance,
risk, recency, dependency, kind_prior, pin_bonus) to approximate them.

This is deliberately NOT wired in as the compiler's new default -- see
"Honest limitations" in benchmarks/README.md. It produces one alternative,
documented preset (`LEARNED_WEIGHTS_V1` in context_compiler.scoring) that
the benchmark can compare against the hand-tuned default via
`run.py --weights learned`, not a replacement backed by enough evidence to
ship as the default.

Usage:
    python benchmarks/learn_weights.py

Writes benchmarks/results/learned_weights.json and prints a report,
including leave-one-task-out cross-validation (14 folds -- this dataset is
small enough that a train/test split would leave too little data on either
side to mean anything, so LOTO is the least-bad honest estimate of whether
the fit generalizes across tasks at all).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_ROOT))

from context_compiler import ContextCompiler, ContextStore  # noqa: E402
from context_compiler.experiments import ExperimentRunner  # noqa: E402
from context_compiler.ingest import RepositoryIngestor  # noqa: E402
from context_compiler.scoring import ScoringWeights  # noqa: E402
from context_compiler.tokenizer import HeuristicTokenCounter  # noqa: E402

from tasks import TASKS, BenchTask, make_evaluator  # noqa: E402

FEATURES = ("relevance", "importance", "risk", "recency", "dependency", "kind_prior", "pin_bonus")
FIT_BUDGET = 3000  # generous enough that nearly every candidate gets selected
DEFAULT_WEIGHT_SUM = sum(getattr(ScoringWeights(), f) for f in FEATURES)

Row = tuple[list[float], int, str]  # (features, label, item_id)


def collect_examples(task_def: BenchTask) -> list[Row]:
    """One (features, label, item_id) row per item deletion-tested at a generous budget."""
    tmp = tempfile.NamedTemporaryFile(prefix="ctxc-learn-", suffix=".db", delete=False)
    tmp.close()
    store = ContextStore(tmp.name)
    RepositoryIngestor(store).ingest(task_def.repo)
    counter = HeuristicTokenCounter()
    compiler = ContextCompiler(store, counter=counter)
    evaluate = make_evaluator(task_def)

    compiled = compiler.compile(task_def.task, FIT_BUDGET)
    breakdowns = {s.item_id: s.breakdown for s in compiled.selections}

    runner = ExperimentRunner(compiler, evaluate)
    effects = runner.deletion_test(task_def.task, compiled, repeats=1)

    rows: list[Row] = []
    for effect in effects:
        b = breakdowns[effect.item_id]
        features = [getattr(b, f) for f in FEATURES]
        label = 1 if effect.delta_success_rate > 0 else 0
        rows.append((features, label, effect.item_id))
    return rows


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic(
    rows: list[Row],
    *,
    l2: float = 0.2,
    lr: float = 0.3,
    epochs: int = 3000,
) -> list[float]:
    """Pure-Python L2-regularized logistic regression via batch gradient descent.

    Returns [intercept, *feature_weights], features in FEATURES order. No
    numpy/sklearn -- confirmed unavailable in this environment and
    consistent with the project's zero-required-dependency core.
    """
    n = len(rows)
    d = len(FEATURES) + 1  # + intercept
    w = [0.0] * d
    xs = [[1.0] + list(f) for f, _, _ in rows]
    ys = [label for _, label, _ in rows]
    for _ in range(epochs):
        grad = [0.0] * d
        for xi, yi in zip(xs, ys):
            z = sum(wj * xj for wj, xj in zip(w, xi))
            err = _sigmoid(z) - yi
            for j in range(d):
                grad[j] += err * xi[j]
        for j in range(d):
            reg = 0.0 if j == 0 else l2 * w[j]  # never regularize the intercept
            w[j] -= lr * (grad[j] / n + reg)
    return w


def predict(coefs: list[float], features: list[float]) -> float:
    z = coefs[0] + sum(c * f for c, f in zip(coefs[1:], features))
    return _sigmoid(z)


def coefficients_to_weights(
    coefs: list[float], *, untestable_features: frozenset[str] = frozenset()
) -> ScoringWeights:
    """Map fitted [intercept, *feature_weights] onto a non-negative ScoringWeights.

    Negative coefficients are floored near zero rather than inverted -- the
    compiler's allocator assumes "higher score = more valuable" for every
    term, so a feature the data found anti-correlated with necessity is
    downweighted, not flipped.

    `untestable_features` (e.g. recency/pin_bonus in this benchmark, where
    every example has nearly the same value -- see the spread check in
    main()) keep their hand-tuned default instead of the fitted
    coefficient: in a near-zero-gradient regime the fitted value is an
    artifact of the flooring step above, not a learned signal, and silently
    zeroing out a deliberately-designed weight (pin_bonus exists to boost
    manually pinned items, which this benchmark never uses) would be a
    regression the data has no way to justify. The remaining, actually
    -varied features are rescaled to fill whatever weight budget is left
    after reserving the untestable features' default share, so the total
    still matches the hand-tuned defaults' sum.
    """
    default = ScoringWeights()
    raw = {f: max(c, 0.005) for f, c in zip(FEATURES, coefs[1:])}
    reserved = sum(getattr(default, f) for f in untestable_features)
    learnable = [f for f in FEATURES if f not in untestable_features]
    budget = DEFAULT_WEIGHT_SUM - reserved
    scale = budget / sum(raw[f] for f in learnable) if learnable else 1.0
    weights = {f: getattr(default, f) for f in untestable_features}
    weights.update({f: raw[f] * scale for f in learnable})
    return ScoringWeights(**weights)


def leave_one_task_out_cv(examples_by_task: dict[str, list[Row]]) -> dict:
    """Classification accuracy is a misleading headline here -- ~78% of items are
    "not essential," so a classifier that always predicts 0 already scores ~0.78.
    `top1_essential_item_ranked_first` is the metric that matches how the
    compiler actually uses these weights (to rank, not classify): does the
    highest-scored item in a held-out task turn out to be the one whose
    removal breaks the task. Compared against chance (1 / items in that
    task, averaged), not against 0.5.
    """
    slugs = list(examples_by_task.keys())
    correct = 0
    total = 0
    n_positive = 0
    top1_hits = 0
    top1_total = 0
    chance_top1 = 0.0
    for held_out in slugs:
        train_rows = [row for slug in slugs if slug != held_out for row in examples_by_task[slug]]
        test_rows = examples_by_task[held_out]
        if not train_rows or not test_rows:
            continue
        coefs = fit_logistic(train_rows)
        for features, label, _ in test_rows:
            pred = 1 if predict(coefs, features) >= 0.5 else 0
            correct += int(pred == label)
            total += 1
            n_positive += label
        if any(label == 1 for _, label, _ in test_rows):
            ranked = sorted(test_rows, key=lambda r: predict(coefs, r[0]), reverse=True)
            top1_hits += int(ranked[0][1] == 1)
            top1_total += 1
            chance_top1 += 1.0 / len(test_rows)
    return {
        "accuracy": correct / total if total else None,
        "majority_class_baseline_accuracy": 1 - n_positive / total if total else None,
        "n": total,
        "top1_essential_item_ranked_first": top1_hits / top1_total if top1_total else None,
        "top1_chance_baseline": chance_top1 / top1_total if top1_total else None,
        "top1_n": top1_total,
    }


def main() -> None:
    examples_by_task: dict[str, list[Row]] = {}
    for task_def in TASKS:
        examples_by_task[task_def.slug] = collect_examples(task_def)

    all_rows = [row for rows in examples_by_task.values() for row in rows]
    n_pos = sum(1 for _, label, _ in all_rows if label == 1)
    print(
        f"Collected {len(all_rows)} (item, label) examples across {len(TASKS)} tasks "
        f"({n_pos} essential / {len(all_rows) - n_pos} not-essential)."
    )

    cv = leave_one_task_out_cv(examples_by_task)
    print(
        "Leave-one-task-out CV (14 folds):\n"
        f"  classification accuracy = {cv['accuracy']:.2f}  "
        f"(majority-class baseline = {cv['majority_class_baseline_accuracy']:.2f} -- "
        f"barely beats always-guess-'not essential')\n"
        f"  top-ranked-item-is-essential rate = {cv['top1_essential_item_ranked_first']:.2f}  "
        f"(chance baseline = {cv['top1_chance_baseline']:.2f}, n={cv['top1_n']} tasks -- "
        f"this is the metric that matches how the compiler actually uses the weights, "
        f"to rank candidates, not classify them)"
    )

    flat = {f: [row[0][i] for row in all_rows] for i, f in enumerate(FEATURES)}
    spreads = {f: max(v) - min(v) for f, v in flat.items()}
    untestable = frozenset(f for f, s in spreads.items() if s < 0.05)

    coefs = fit_logistic(all_rows)
    learned = coefficients_to_weights(coefs, untestable_features=untestable)
    default = ScoringWeights()

    print("\nfeature        default  learned  (range across examples)")
    for f in FEATURES:
        values = flat[f]
        flag = "  <- near-constant in this benchmark; kept at default, not learned" if f in untestable else ""
        print(f"{f:<13}  {getattr(default, f):>6.3f}  {getattr(learned, f):>6.3f}  [{min(values):.2f}, {max(values):.2f}]{flag}")

    out = {
        "n_examples": len(all_rows),
        "n_positive": n_pos,
        "cv": cv,
        "intercept": coefs[0],
        "coefficients": dict(zip(FEATURES, coefs[1:])),
        "untestable_features": sorted(untestable),
        "learned_weights": {f: getattr(learned, f) for f in FEATURES},
        "default_weights": {f: getattr(default, f) for f in FEATURES},
    }
    results_dir = BENCH_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "learned_weights.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote benchmarks/results/learned_weights.json")


if __name__ == "__main__":
    main()
