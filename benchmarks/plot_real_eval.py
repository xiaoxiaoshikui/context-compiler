"""Charts for the real-execution SWE-bench evaluation (see real_eval.py and
benchmarks/README.md "Does minimum sufficient context exist?"). Reuses
plot_results.py's SVG bar-chart helper (pure stdlib, no plotting library).

Unlike plot_results.py's B95 charts, more is better here (resolved count) --
noted explicitly in each chart's subtitle since the shared helper's docstring
default assumption is the opposite.

Usage:
    python benchmarks/plot_real_eval.py

Writes benchmarks/charts/real_eval_*.svg.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_ROOT))

from plot_results import bar_chart_svg  # noqa: E402

RESULTS_PATH = BENCH_ROOT / "results" / "real_eval_results.json"
CHARTS_DIR = BENCH_ROOT / "charts"


def main() -> None:
    data = {c["tag"]: c for c in json.loads(RESULTS_PATH.read_text())}
    n_tasks = len(next(iter(data.values()))["records"])

    def resolved_count(tag: str) -> int:
        return len(data[tag]["harness"].get("resolved_ids", []))

    # --- Chart 1: Oracle budget sweep -- the Context-Performance Frontier ---
    budgets = [1000, 4000, 8000, 16000]
    counts = [resolved_count(f"full2_oracle_{b}_r0") for b in budgets]
    svg = bar_chart_svg(
        title="Does minimum sufficient context exist? Oracle budget sweep",
        subtitle=(
            f"6 real SWE-bench Verified instances, ground-truth-relevant context only (never the gold "
            f"patch itself), real DeepSeek-generated patches, real Docker test execution. Higher = better "
            f"(resolved out of {n_tasks}). One run per budget -- noisy, illustrative."
        ),
        categories=[f"{b:,} tokens" for b in budgets],
        series=[("resolved", "#2563eb", counts)],
        y_label=f"instances resolved (of {n_tasks})",
    )
    (CHARTS_DIR / "real_eval_oracle_sweep.svg").write_text(svg)
    print("wrote", CHARTS_DIR / "real_eval_oracle_sweep.svg")

    # --- Chart 2: method comparison at a fixed budget ---
    budget = 8000
    oracle_n = resolved_count(f"full2_oracle_{budget}_r0")
    random_runs = [resolved_count(f"full4_random_{budget}_r{i}") for i in (0, 1)]
    # Every "ours" run collected across every fix stage, oldest to newest,
    # shown in full rather than cherry-picking a flattering before/after
    # pair -- three real, verified bugs were fixed between full4 and
    # topk3 (see benchmarks/README.md "2026-08-22 continued"), and the
    # resolved count did not move in a clean line: 0,1 -> 0,2 -> 1,0 ->
    # 0,0. That noise is itself the honest finding at this sample size,
    # not something to average away by only showing two points.
    ours_stages = [
        ("full4", "before any fix"),
        ("topk1", "+ top_k_full_text"),
        ("topk2", "+ item cap"),
        ("topk3", "+ graph rescue, k=7"),
    ]
    ours_runs = {
        stage: [resolved_count(f"{stage}_ours_{budget}_r{i}") for i in (0, 1)]
        for stage, _ in ours_stages
    }
    categories = ["Oracle (ceiling)", "Random (floor) r0", "Random (floor) r1"]
    values = [oracle_n, random_runs[0], random_runs[1]]
    for stage, label in ours_stages:
        categories += [f"Ours ({label}) r0", f"Ours ({label}) r1"]
        values += ours_runs[stage]
    svg = bar_chart_svg(
        title=f"Oracle ceiling vs Random floor vs Ours, at a fixed {budget:,}-token budget",
        subtitle=(
            f"Same 6 tasks, same budget, same model -- only how the context was chosen differs. "
            f"Every 'Ours' stage is a real, verified bug fix (see benchmarks/README.md), shown in "
            f"full rather than a single before/after pair. Repeats shown as separate bars -- a real "
            f"model is stochastic, n=2 per stage is not enough to separate luck from signal, and "
            f"the run-to-run noise visible here is itself part of the honest result. Higher = better."
        ),
        categories=categories,
        series=[("resolved", "#2563eb", values)],
        y_label=f"instances resolved (of {n_tasks})",
    )
    (CHARTS_DIR / "real_eval_method_comparison.svg").write_text(svg)
    print("wrote", CHARTS_DIR / "real_eval_method_comparison.svg")


if __name__ == "__main__":
    main()
