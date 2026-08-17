"""Task definitions for the starter benchmark.

Each task points at a small synthetic repository containing distractor
code plus exactly one file that carries the safety-critical fact the task
cannot be solved correctly without (an operational policy/constraint).
The evaluator is a deterministic keyword-conjunction check against the
compiled context text, not a real agent/test harness -- it answers "does
the working set contain the invariant", not "did an LLM produce a correct
patch". See README.md in this directory for why that distinction matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from context_compiler.experiments import EvaluationResult

_WHITESPACE_RE = re.compile(r"\s+")

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent


@dataclass(frozen=True, slots=True)
class BenchTask:
    slug: str
    repo: Path
    task: str
    required_substrings: tuple[str, ...]  # ANDed, case-insensitive
    oracle_source: str  # path (relative to repo) of the file that must be present to pass


def _normalize(text: str) -> str:
    # Collapse whitespace (including line wraps inside a markdown paragraph)
    # so a required phrase that happens to wrap across two lines still matches.
    return _WHITESPACE_RE.sub(" ", text.lower())


def _contains_all(text: str, required: tuple[str, ...]) -> bool:
    low = _normalize(text)
    return all(_normalize(r) in low for r in required)


def make_evaluator(task_def: BenchTask):
    def evaluate(task: str, context: str) -> EvaluationResult:
        ok = _contains_all(context, task_def.required_substrings)
        return EvaluationResult(success=ok, score=1.0 if ok else 0.0)

    return evaluate


TASKS: list[BenchTask] = [
    BenchTask(
        slug="oauth_safari",
        repo=REPO_ROOT / "examples" / "demo_repo",
        task="Fix the Safari OAuth callback regression; preserve the no-replay security invariant",
        required_substrings=("never automatically replay", "single-use"),
        oracle_source="SECURITY_CONSTRAINTS.md",
    ),
    BenchTask(
        slug="payment_idempotency",
        repo=BENCH_ROOT / "repos" / "payment_idempotency",
        task="Fix duplicate customer charges when the payment gateway times out",
        required_substrings=("idempotency key", "never retry"),
        oracle_source="PAYMENT_POLICY.md",
    ),
    BenchTask(
        slug="db_migration",
        repo=BENCH_ROOT / "repos" / "db_migration",
        task="Add a migration that drops the legacy sessions_v1 table",
        required_substrings=("verified backup", "destructive schema migration"),
        oracle_source="MIGRATION_POLICY.md",
    ),
    BenchTask(
        slug="rate_limit",
        repo=BENCH_ROOT / "repos" / "rate_limit",
        task="Debug why legitimate users are being rate-limited during a traffic spike",
        required_substrings=("never disable", "credential-stuffing"),
        oracle_source="RATE_LIMIT_POLICY.md",
    ),
    BenchTask(
        slug="secrets_rotation",
        repo=BENCH_ROOT / "repos" / "secrets_rotation",
        task="Rotate the expired API key used by the billing webhook",
        required_substrings=("never commit", "secrets manager"),
        oracle_source="SECRETS_POLICY.md",
    ),
]
