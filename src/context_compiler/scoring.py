from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

from .models import ContextItem, ContextKind, ScoreBreakdown


_TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-./]*|[\u3400-\u9fff]", re.UNICODE)


def terms(text: str) -> list[str]:
    return [t.lower() for t in _TERM_RE.findall(text)]


def term_set(text: str) -> set[str]:
    return set(terms(text))


@dataclass(slots=True)
class ScoringWeights:
    relevance: float = 0.55
    importance: float = 0.13
    risk: float = 0.14
    recency: float = 0.04
    dependency: float = 0.08
    kind_prior: float = 0.03
    pin_bonus: float = 0.18


_KIND_PRIOR: dict[ContextKind, float] = {
    ContextKind.CONSTRAINT: 1.0,
    ContextKind.DECISION: 0.85,
    ContextKind.TEST: 0.70,
    ContextKind.CODE: 0.68,
    ContextKind.CONFIG: 0.62,
    ContextKind.DOC: 0.52,
    ContextKind.TOOL: 0.50,
    ContextKind.CONVERSATION: 0.45,
    ContextKind.NOTE: 0.40,
}


def lexical_relevance(task: str, item: ContextItem) -> float:
    q = term_set(task)
    if not q:
        return 0.0

    title = term_set(item.title)
    tags = set(t.lower() for t in item.tags)
    source = term_set(item.source)
    # Restrict content scanning to keep ranking predictable on huge files.
    body_terms = terms(item.content[:120_000])
    body = set(body_terms)

    coverage_body = len(q & body) / len(q)
    coverage_title = len(q & title) / len(q)
    coverage_tags = len(q & tags) / len(q)
    coverage_source = len(q & source) / len(q)

    tf_bonus = 0.0
    if body_terms:
        counts: dict[str, int] = {}
        for t in body_terms:
            if t in q:
                counts[t] = counts.get(t, 0) + 1
        tf_bonus = sum(min(3, c) for c in counts.values()) / (3 * len(q))

    phrase_bonus = 0.12 if task.strip().lower() in item.content.lower() else 0.0
    score = (
        0.38 * coverage_body
        + 0.28 * coverage_title
        + 0.12 * coverage_tags
        + 0.12 * coverage_source
        + 0.10 * tf_bonus
        + phrase_bonus
    )
    return max(0.0, min(1.0, score))


def recency_score(updated_at: str, half_life_days: float = 60.0) -> float:
    if not updated_at:
        return 0.3
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        return math.pow(0.5, age_days / half_life_days)
    except ValueError:
        return 0.3


def dependency_score(task: str, item: ContextItem, activated_terms: set[str] | None = None) -> float:
    if not item.dependencies:
        return 0.0
    q = term_set(task)
    active = activated_terms or set()
    deps: set[str] = set()
    for dep in item.dependencies:
        deps |= term_set(dep)
        deps.add(PurePosixPath(dep.replace("\\", "/")).name.lower())
    if not deps:
        return 0.0
    task_overlap = len(deps & q) / max(1, len(q))
    active_overlap = len(deps & active) / max(1, len(active)) if active else 0.0
    return min(1.0, 0.7 * task_overlap + 0.3 * active_overlap)


def score_item(
    task: str,
    item: ContextItem,
    *,
    weights: ScoringWeights | None = None,
    activated_terms: set[str] | None = None,
) -> ScoreBreakdown:
    w = weights or ScoringWeights()
    rel = lexical_relevance(task, item)
    imp = max(0.0, min(1.0, item.importance))
    risk = max(0.0, min(1.0, item.omission_risk))
    rec = recency_score(item.updated_at)
    dep = dependency_score(task, item, activated_terms)
    kp = _KIND_PRIOR.get(item.kind, 0.4)
    pin = 1.0 if item.pinned else 0.0
    total = (
        w.relevance * rel
        + w.importance * imp
        + w.risk * risk
        + w.recency * rec
        + w.dependency * dep
        + w.kind_prior * kp
        + w.pin_bonus * pin
    )
    # A small floor for pinned items prevents a lexical miss from evicting them.
    if item.pinned:
        total = max(total, 0.55)
    return ScoreBreakdown(
        relevance=rel,
        importance=imp,
        risk=risk,
        recency=rec,
        dependency=dep,
        kind_prior=kp,
        pin_bonus=pin,
        total=min(1.25, total),
    )


def activated_dependency_terms(scored: list[tuple[ContextItem, ScoreBreakdown]], top_n: int = 12) -> set[str]:
    active: set[str] = set()
    for item, _ in sorted(scored, key=lambda x: x[1].total, reverse=True)[:top_n]:
        active |= term_set(item.title)
        active |= term_set(item.source)
        for dep in item.dependencies:
            active |= term_set(dep)
    return active
