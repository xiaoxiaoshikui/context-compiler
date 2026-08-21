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
    ours_before = [resolved_count(f"full4_ours_{budget}_r{i}") for i in (0, 1)]
    ours_after = [resolved_count(f"topk1_ours_{budget}_r{i}") for i in (0, 1)]
    svg = bar_chart_svg(
        title=f"Oracle ceiling vs Random floor vs Ours, at a fixed {budget:,}-token budget",
        subtitle=(
            f"Same 6 tasks, same budget, same model -- only how the context was chosen differs. "
            f"'Ours (before)' spread the budget over many low-fidelity items; 'Ours (after)' forces "
            f"the top-3 candidates to full text first (top_k_full_text). Repeats shown as separate "
            f"bars -- a real model is stochastic, n=2 is not enough to separate luck from signal. "
            f"Higher = better."
        ),
        categories=[
            "Oracle (ceiling)",
            "Random (floor) r0", "Random (floor) r1",
            "Ours (before) r0", "Ours (before) r1",
            "Ours (after) r0", "Ours (after) r1",
        ],
        series=[("resolved", "#2563eb", [
            oracle_n,
            random_runs[0], random_runs[1],
            ours_before[0], ours_before[1],
            ours_after[0], ours_after[1],
        ])],
        y_label=f"instances resolved (of {n_tasks})",
    )
    (CHARTS_DIR / "real_eval_method_comparison.svg").write_text(svg)
    print("wrote", CHARTS_DIR / "real_eval_method_comparison.svg")


if __name__ == "__main__":
    main()
