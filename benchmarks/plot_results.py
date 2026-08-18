"""Generate SVG charts from the committed benchmarks/results/*.json files.

No dependencies (no matplotlib, no plotting library) -- consistent with
this project's zero-required-dependency core, and SVG is plain text, so
the generated files diff sensibly in git like everything else here. This
script only *reads* already-run result files; it never runs the
benchmark itself (run `python benchmarks/run.py` -- and its --weights /
--retriever / --graph variants -- first if you've changed something and
want the charts to reflect it).

Usage:
    python benchmarks/plot_results.py

Writes benchmarks/charts/*.svg, embedded from benchmarks/README.md and
README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_ROOT / "results"
CHARTS_DIR = BENCH_ROOT / "charts"

_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

Series = tuple[str, str, list[float | None]]


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _load(name: str) -> dict[str, dict]:
    data = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
    return {r["slug"]: r for r in data}


def _b95(results: dict[str, dict], slug: str, method: str = "ours") -> float | None:
    r = results.get(slug)
    if r is None:
        return None
    return r["methods"][method]["b95"]


def bar_chart_svg(
    *,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[Series],
    y_label: str,
    width: int = 860,
    row_height: int = 26,
) -> str:
    """Horizontal grouped bar chart. Lower is better (B95 = smaller budget
    reaching 100%% success is better) -- read a shorter bar as a win, same
    as any bar chart, just noting the direction since "more" isn't "better"
    here.
    """
    n_series = len(series)
    all_values = [v for _, _, vals in series for v in vals if v is not None]
    y_max = (max(all_values) * 1.15) if all_values else 1.0

    margin_left = 200
    margin_right = 90
    subtitle_lines = _wrap(subtitle, max_chars=int((width - margin_left) / 6.4))
    margin_top = 46 + 18 * len(subtitle_lines) + 22
    margin_bottom = 46
    group_h = row_height * n_series + 12
    plot_w = width - margin_left - margin_right
    height = margin_top + margin_bottom + group_h * len(categories)

    def x(v: float) -> float:
        return (v / y_max) * plot_w

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{_FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="27" font-size="16.5" font-weight="600" fill="#111827">{title}</text>',
    ]
    for i, line in enumerate(subtitle_lines):
        parts.append(
            f'<text x="{margin_left}" y="{46 + 16 * i}" font-size="12" fill="#4b5563">{line}</text>'
        )

    lx, ly = margin_left, 46 + 18 * len(subtitle_lines) + 14
    for name, color, _ in series:
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 17}" y="{ly}" font-size="12" fill="#374151">{name}</text>')
        lx += 17 + 7 * len(name) + 26

    plot_bottom = margin_top + group_h * len(categories)
    n_ticks = 5
    for i in range(n_ticks + 1):
        gx = margin_left + plot_w * i / n_ticks
        val = y_max * i / n_ticks
        parts.append(
            f'<line x1="{gx:.1f}" y1="{margin_top - 6}" x2="{gx:.1f}" y2="{plot_bottom}" '
            f'stroke="#eef0f2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{plot_bottom + 18}" font-size="11" fill="#6b7280" '
            f'text-anchor="middle">{val:.0f}</text>'
        )
    parts.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 6}" font-size="11.5" '
        f'fill="#6b7280" text-anchor="middle">{y_label}</text>'
    )

    bar_h = row_height - 7
    for ci, cat in enumerate(categories):
        gy = margin_top + ci * group_h
        parts.append(
            f'<text x="{margin_left - 12}" y="{gy + group_h / 2 + 4:.1f}" font-size="12" '
            f'fill="#111827" text-anchor="end">{cat}</text>'
        )
        for si, (_, color, vals) in enumerate(series):
            v = vals[ci]
            by = gy + 6 + si * row_height
            if v is None:
                parts.append(
                    f'<text x="{margin_left + 6}" y="{by + bar_h / 2 + 4:.1f}" font-size="11" '
                    f'fill="#9ca3af" font-style="italic">not reached</text>'
                )
                continue
            bw = max(2.0, x(v))
            parts.append(
                f'<rect x="{margin_left}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" '
                f'rx="3" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{margin_left + bw + 6:.1f}" y="{by + bar_h / 2 + 4:.1f}" '
                f'font-size="11.5" fill="#111827">{v:.0f}</text>'
            )
        parts.append(
            f'<line x1="{margin_left}" y1="{gy + group_h - 2}" x2="{width - margin_right}" '
            f'y2="{gy + group_h - 2}" stroke="#f3f4f6" stroke-width="1"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _write(name: str, svg: str) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    path = CHARTS_DIR / name
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path.relative_to(BENCH_ROOT.parent)}")


def main() -> None:
    default = _load("results.json")
    slugs = list(default.keys())

    svg = bar_chart_svg(
        title="B95 by task and method",
        subtitle="Smallest budget (tokens) reaching 100% evaluator success. Shorter = better. Current code, default settings.",
        categories=slugs,
        series=[
            ("ours", "#2563eb", [_b95(default, s, "ours") for s in slugs]),
            ("full", "#9ca3af", [_b95(default, s, "full") for s in slugs]),
            ("random", "#d1d5db", [_b95(default, s, "random") for s in slugs]),
        ],
        y_label="B95 (tokens)",
    )
    _write("overview_b95.svg", svg)

    fts = _load("results.fts_retriever.json")
    svg = bar_chart_svg(
        title="Phase 3: retrieval, FTS vs TF-IDF",
        subtitle="'ours' B95 before (SQLite FTS5 candidate_search) vs after (TfidfRetriever, current default). cache_ttl never reached 100% under FTS at any swept budget.",
        categories=slugs,
        series=[
            ("FTS (pre-Phase 3)", "#9ca3af", [_b95(fts, s, "ours") for s in slugs]),
            ("TF-IDF (current)", "#2563eb", [_b95(default, s, "ours") for s in slugs]),
        ],
        y_label="B95 (tokens)",
    )
    _write("phase3_retrieval.svg", svg)

    no_graph = _load("results.no_graph.json")
    svg = bar_chart_svg(
        title="Phase 4a: dependency graph, off vs on",
        subtitle="'ours' B95 with the dependency-graph score boost disabled vs enabled (current default). 3 tasks improve, 0 regress.",
        categories=slugs,
        series=[
            ("graph off (pre-Phase 4a)", "#9ca3af", [_b95(no_graph, s, "ours") for s in slugs]),
            ("graph on (current)", "#2563eb", [_b95(default, s, "ours") for s in slugs]),
        ],
        y_label="B95 (tokens)",
    )
    _write("phase4a_graph.svg", svg)

    learned = _load("results.learned_weights.json")
    svg = bar_chart_svg(
        title="Phase 2: hand-tuned vs learned scoring weights",
        subtitle="'ours' B95, default ScoringWeights vs LEARNED_WEIGHTS_V1 (not the default), both on today's full pipeline (TF-IDF + dependency graph). LEARNED_WEIGHTS_V1 predates the dependency graph and now regresses the 2 tasks that rely on it most -- see benchmarks/README.md Phase 2.",
        categories=slugs,
        series=[
            ("default weights", "#2563eb", [_b95(default, s, "ours") for s in slugs]),
            ("learned weights", "#d97706", [_b95(learned, s, "ours") for s in slugs]),
        ],
        y_label="B95 (tokens)",
    )
    _write("phase2_weights.svg", svg)


if __name__ == "__main__":
    main()
