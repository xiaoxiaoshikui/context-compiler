"""Starter benchmark: Ours vs Full-context vs Random, with an Oracle reference point.

This is a first, runnable slice of README.md's "Recommended next research
iterations" item 1-3 (build a benchmark harness, compare against naive
baselines, estimate B95). It intentionally does NOT reach the recommended
30-100 task scale yet, and the evaluator is a deterministic keyword check,
not a real coding agent -- see tasks.py and README.md for the caveats.

Usage:
    python benchmarks/run.py
    python benchmarks/run.py --evaluator llm_judge --yes   # costs real API calls; see below

Writes benchmarks/results/results.json and benchmarks/REPORT.md (keyword
evaluator only -- llm_judge runs write to results.llm_judge.json and print
to stdout instead, so they never silently overwrite the keyword baseline).

The default evaluator is the free/instant keyword check. `--evaluator
llm_judge` swaps in benchmarks/llm_judge_eval.py, a real Anthropic API call
per (method, budget) -- see benchmarks/README.md for what it does and does
not prove. It requires `pip install -e '.[llm_judge]'`, a real
ANTHROPIC_API_KEY, and `--yes` to confirm the cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_ROOT))

from context_compiler import ContextCompiler, ContextStore  # noqa: E402
from context_compiler.baselines import FullContextBaseline, RandomContextBaseline  # noqa: E402
from context_compiler.experiments import ExperimentRunner, b95  # noqa: E402
from context_compiler.ingest import RepositoryIngestor  # noqa: E402
from context_compiler.models import RenderLevel  # noqa: E402
from context_compiler.render import ContextRenderer  # noqa: E402
from context_compiler.scoring import ScoringWeights  # noqa: E402
from context_compiler.tokenizer import HeuristicTokenCounter  # noqa: E402

from tasks import TASKS, BenchTask, make_evaluator  # noqa: E402

BUDGETS = [80, 150, 250, 400, 700, 1200, 2000, 3000]
REPEATS_DETERMINISTIC = 1  # "ours" and "full" have no randomness
REPEATS_RANDOM = 20
TARGET_SUCCESS_RATE = 1.0  # evaluator's ceiling is a binary "does it contain the fact"

_LEVELS = [RenderLevel.L0, RenderLevel.L1, RenderLevel.L2, RenderLevel.L3, RenderLevel.L4]


def load_store(repo_path: Path) -> ContextStore:
    tmp = tempfile.NamedTemporaryFile(prefix="ctxc-bench-", suffix=".db", delete=False)
    tmp.close()
    store = ContextStore(tmp.name)
    RepositoryIngestor(store).ingest(repo_path)
    return store


def oracle_tokens(store: ContextStore, task_def: BenchTask, counter: HeuristicTokenCounter) -> int | None:
    """Smallest single-item representation that alone satisfies the evaluator."""
    item = store.get_by_source(task_def.oracle_source)
    if item is None:
        return None
    renderer = ContextRenderer(counter)
    evaluate = make_evaluator(task_def)
    header = f"[CONTEXT-COMPILER]\nTASK: {task_def.task}"
    for level in _LEVELS:
        variant = renderer.render(item, task_def.task, level)
        if evaluate(task_def.task, header + "\n\n" + variant.text).success:
            return variant.token_count
    return None


def make_llm_judge_evaluator(task_def: BenchTask):
    from context_compiler.experiments import JsonCommandEvaluator

    return JsonCommandEvaluator([sys.executable, str(BENCH_ROOT / "llm_judge_eval.py")])


EVALUATOR_FACTORIES = {
    "keyword": make_evaluator,
    "llm_judge": make_llm_judge_evaluator,
}


def load_learned_weights() -> ScoringWeights:
    path = BENCH_ROOT / "results" / "learned_weights.json"
    if not path.exists():
        raise SystemExit(
            "benchmarks/results/learned_weights.json not found -- run "
            "`python benchmarks/learn_weights.py` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScoringWeights(**data["learned_weights"])


def run_task(
    task_def: BenchTask,
    *,
    evaluator_name: str,
    budgets: list[int],
    weights: ScoringWeights | None = None,
) -> dict:
    store = load_store(task_def.repo)
    counter = HeuristicTokenCounter()
    evaluate = EVALUATOR_FACTORIES[evaluator_name](task_def)

    # llm_judge is a paid, non-instant call -- one shot per (method, budget),
    # no repeats. The keyword evaluator is free/instant, so it can afford
    # 20 repeats on the random baseline to smooth out its variance.
    # `weights` only affects "ours" -- the naive baselines don't score at all.
    if evaluator_name == "keyword":
        methods = {
            "ours": (ContextCompiler(store, counter=counter, weights=weights), REPEATS_DETERMINISTIC),
            "full": (FullContextBaseline(store, counter=counter), REPEATS_DETERMINISTIC),
            "random": (RandomContextBaseline(store, counter=counter, seed=1234), REPEATS_RANDOM),
        }
    else:
        methods = {
            "ours": (ContextCompiler(store, counter=counter, weights=weights), 1),
            "full": (FullContextBaseline(store, counter=counter), 1),
            "random": (RandomContextBaseline(store, counter=counter, seed=1234), 1),
        }

    result: dict = {"slug": task_def.slug, "task": task_def.task, "methods": {}}
    for name, (compiler, repeats) in methods.items():
        runner = ExperimentRunner(compiler, evaluate)
        points = runner.budget_sweep(task_def.task, budgets, repeats=repeats)
        result["methods"][name] = {
            "points": [p.to_dict() for p in points],
            "b95": b95(points, TARGET_SUCCESS_RATE),
        }
    result["oracle_tokens"] = oracle_tokens(store, task_def, counter)
    return result


def render_report(all_results: list[dict]) -> str:
    lines = [
        "# Benchmark report",
        "",
        "Auto-generated by `benchmarks/run.py`. Do not hand-edit; re-run the script instead.",
        "",
        "Methods compared: `ours` (ContextCompiler), `full` (dump everything in path order,",
        "truncate at the budget), `random` (shuffle items, greedily keep what fits, 20 repeats).",
        "`oracle_tokens` is the smallest single-representation cost of the one file that alone",
        "satisfies the task's evaluator -- a lower bound, not a method under test.",
        "",
        "B95 here means: smallest budget at which the method's success rate reaches 100%",
        "(the evaluator is a binary keyword check, so its own ceiling is 1.0, not a",
        "full-context baseline's ceiling). A blank cell means the method never reached 100%",
        "success within the swept budgets.",
        "",
        "| task | oracle tokens | B95 ours | B95 full | B95 random |",
        "|---|---|---|---|---|",
    ]
    for r in all_results:
        m = r["methods"]
        lines.append(
            f"| {r['slug']} | {r['oracle_tokens']} "
            f"| {m['ours']['b95'] if m['ours']['b95'] is not None else '-'} "
            f"| {m['full']['b95'] if m['full']['b95'] is not None else '-'} "
            f"| {m['random']['b95'] if m['random']['b95'] is not None else '-'} |"
        )
    lines.append("")
    for r in all_results:
        lines.append(f"## {r['slug']}")
        lines.append("")
        lines.append(f"Task: {r['task']}")
        lines.append("")
        lines.append("| budget | ours | full | random |")
        lines.append("|---|---|---|---|")
        budgets = [p["budget"] for p in r["methods"]["ours"]["points"]]
        for i, budget in enumerate(budgets):
            row = [str(budget)]
            for name in ("ours", "full", "random"):
                row.append(f"{r['methods'][name]['points'][i]['success_rate']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the context-compiler benchmark suite.")
    p.add_argument(
        "--evaluator",
        choices=sorted(EVALUATOR_FACTORIES),
        default="keyword",
        help="keyword (free, default) or llm_judge (real Anthropic API calls, costs money)",
    )
    p.add_argument("--tasks", help="comma-separated task slugs to run (default: all)")
    p.add_argument(
        "--budgets", help="comma-separated token budgets (default: the built-in sweep)"
    )
    p.add_argument(
        "--weights",
        choices=["default", "learned"],
        default="default",
        help="default (hand-tuned, ships with the compiler) or learned "
        "(fit by benchmarks/learn_weights.py; run that first)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the cost confirmation prompt for --evaluator llm_judge",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = TASKS
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in TASKS if t.slug in wanted]
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else BUDGETS
    weights = load_learned_weights() if args.weights == "learned" else None

    if args.evaluator == "llm_judge":
        call_count = len(tasks) * len(budgets) * 3  # 3 methods, 1 repeat each
        print(
            f"--evaluator llm_judge makes ~{call_count} real Anthropic API call(s) "
            f"(claude-haiku-4-5) against {len(tasks)} task(s) x {len(budgets)} budget(s) "
            "x 3 methods. This costs real money.",
            file=sys.stderr,
        )
        if not args.yes:
            print("Re-run with --yes to proceed.", file=sys.stderr)
            raise SystemExit(1)

    all_results = [
        run_task(t, evaluator_name=args.evaluator, budgets=budgets, weights=weights)
        for t in tasks
    ]

    results_dir = BENCH_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    is_default_run = args.evaluator == "keyword" and args.weights == "default"
    suffix = "" if args.evaluator == "keyword" else f".{args.evaluator}"
    suffix += "" if args.weights == "default" else f".{args.weights}_weights"
    (results_dir / f"results{suffix}.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    report = render_report(all_results)
    if is_default_run:
        (BENCH_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
