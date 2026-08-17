from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ContextKind(str, Enum):
    CODE = "code"
    TEST = "test"
    DOC = "doc"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    CONVERSATION = "conversation"
    TOOL = "tool"
    CONFIG = "config"
    NOTE = "note"


class RenderLevel(str, Enum):
    L0 = "L0"  # pointer
    L1 = "L1"  # metadata / outline
    L2 = "L2"  # extractive summary
    L3 = "L3"  # relevant excerpts
    L4 = "L4"  # full raw content


@dataclass(slots=True)
class ContextItem:
    id: str
    title: str
    content: str
    kind: ContextKind = ContextKind.NOTE
    source: str = ""
    tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    importance: float = 0.5
    omission_risk: float = 0.2
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["tags"] = list(self.tags)
        data["dependencies"] = list(self.dependencies)
        return data


@dataclass(slots=True)
class ScoreBreakdown:
    relevance: float
    importance: float
    risk: float
    recency: float
    dependency: float
    kind_prior: float
    pin_bonus: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class RenderedVariant:
    item_id: str
    level: RenderLevel
    text: str
    token_count: int
    fidelity: float


@dataclass(slots=True)
class SelectedContext:
    item_id: str
    title: str
    kind: str
    level: RenderLevel
    text: str
    token_count: int
    score: float
    utility: float
    source: str
    breakdown: ScoreBreakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "kind": self.kind,
            "level": self.level.value,
            "text": self.text,
            "token_count": self.token_count,
            "score": self.score,
            "utility": self.utility,
            "source": self.source,
            "breakdown": self.breakdown.to_dict(),
        }


@dataclass(slots=True)
class CompiledContext:
    task: str
    budget: int
    used_tokens: int
    body_tokens: int
    selections: list[SelectedContext]
    text: str
    candidate_count: int
    omitted_ids: list[str]
    audit: list[str]

    @property
    def utilization(self) -> float:
        return 0.0 if self.budget <= 0 else self.used_tokens / self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "budget": self.budget,
            "used_tokens": self.used_tokens,
            "body_tokens": self.body_tokens,
            "utilization": round(self.utilization, 4),
            "candidate_count": self.candidate_count,
            "selections": [s.to_dict() for s in self.selections],
            "omitted_ids": self.omitted_ids,
            "audit": self.audit,
            "text": self.text,
        }
