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

    # --- Chart 3: same context, different downstream model ---
    # The direct test of "is the gap really about context, or about the
    # model/patch-synthesis step downstream of it" -- everything else
    # (context construction: topk3, i.e. every fix in this repo's history;
    # budget; instances) held fixed, only the model that turns context
    # into a patch changes.
    deepseek_runs = [resolved_count(f"topk3_ours_{budget}_r{i}") for i in (0, 1)]
    # gpt and gemini share a tag+seed (both "modelswap_ours_8000_r0") but
    # differ by "model" -- the tag-keyed `data` dict above collides on
    # that, so look these two up directly by scanning the raw list.
    model_counts = {}
    for c in json.loads(RESULTS_PATH.read_text()):
        if c["tag"] == "modelswap_ours_8000_r0":
            model_counts[c.get("model", "?")] = len(c["harness"].get("resolved_ids", []))
    svg = bar_chart_svg(
        title="Same context, different downstream model -- is the gap really about context?",
        subtitle=(
            f"Identical compiled context (every context-construction fix applied), identical "
            f"budget ({budget:,} tokens), identical 6 instances -- only the model synthesizing "
            f"the patch changes. DeepSeek: 2 repeats already collected. GPT-5.2 / Gemini: 1 "
            f"repeat each so far. Higher = better (resolved out of {n_tasks})."
        ),
        categories=["DeepSeek r0", "DeepSeek r1", "GPT-5.2 r0", "Gemini r0"],
        series=[("resolved", "#2563eb", [
            deepseek_runs[0], deepseek_runs[1],
            model_counts.get("gpt", 0), model_counts.get("gemini", 0),
        ])],
        y_label=f"instances resolved (of {n_tasks})",
    )
    (CHARTS_DIR / "real_eval_model_swap.svg").write_text(svg)
    print("wrote", CHARTS_DIR / "real_eval_model_swap.svg")

    # --- Chart 4: same model, different retrieval algorithm ---
    # The other half of "where's the problem" -- hold the model fixed
    # (DeepSeek throughout), swap only the first-stage retriever that
    # decides which ~200 candidates exist before any scoring/rendering
    # runs at all.
    tfidf_tags = ["full4_ours_8000_r0", "full4_ours_8000_r1",
                  "topk1_ours_8000_r0", "topk1_ours_8000_r1",
                  "topk2_ours_8000_r0", "topk2_ours_8000_r1",
                  "topk3_ours_8000_r0", "topk3_ours_8000_r1"]
    tfidf_counts = [resolved_count(t) for t in tfidf_tags]
    embed_counts = [resolved_count("embed_ours_8000_r0"), resolved_count("embed_ours_8000_r1")]
    svg = bar_chart_svg(
        title="Same model, different retrieval algorithm -- TF-IDF vs real embeddings",
        subtitle=(
            f"DeepSeek throughout, identical budget ({budget:,} tokens), identical 6 instances -- "
            f"only context_compiler's first-stage retriever changes. TF-IDF bars are every 'ours' "
            f"run across every context-construction fix stage in this file (8 runs); Embedding is "
            f"EmbeddingRetriever (real OpenAI embeddings), 2 repeats so far. Higher = better "
            f"(resolved out of {n_tasks})."
        ),
        categories=[f"TF-IDF r{i}" for i in range(len(tfidf_counts))] + ["Embedding r0", "Embedding r1"],
        series=[("resolved", "#2563eb", tfidf_counts + embed_counts)],
        y_label=f"instances resolved (of {n_tasks})",
    )
    (CHARTS_DIR / "real_eval_retriever_swap.svg").write_text(svg)
    print("wrote", CHARTS_DIR / "real_eval_retriever_swap.svg")


if __name__ == "__main__":
    main()
