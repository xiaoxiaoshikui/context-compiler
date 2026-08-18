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
    # --- config-lookup: the fact is a specific value buried in a config
    # file, not a policy statement. No constraint doc exists in these
    # repos, so the compiler's forced-minimum-level safety net for
    # constraints never kicks in -- these tasks test relevance ranking
    # on its own.
    BenchTask(
        slug="db_pool_timeout",
        repo=BENCH_ROOT / "repos" / "db_pool_timeout",
        task="Diagnose why orders-api requests intermittently fail with "
        "'timed out waiting for a database connection' during peak traffic",
        required_substrings=("db_pool_size=5",),
        oracle_source="service.env",
    ),
    BenchTask(
        slug="feature_flag_rollout",
        repo=BENCH_ROOT / "repos" / "feature_flag_rollout",
        task="Explain why only some users see the new checkout flow after "
        "it was marked enabled",
        required_substrings=("new_checkout_rollout_percent: 10",),
        oracle_source="flags.yaml",
    ),
    BenchTask(
        slug="rate_config_mismatch",
        repo=BENCH_ROOT / "repos" / "rate_config_mismatch",
        task="Investigate why the billing client raises a timeout error "
        "even though the billing service itself responds within 400ms",
        required_substrings=("upstream_timeout_ms: 500",),
        oracle_source="client_config.yaml",
    ),
    # --- code-behavior: the fact is in a function's actual default/logic,
    # not a doc. Also no constraint doc in these repos.
    BenchTask(
        slug="cache_ttl",
        repo=BENCH_ROOT / "repos" / "cache_ttl",
        task="Explain why cached prices shown to users are sometimes stale "
        "for up to half a minute after an admin updates a price",
        required_substrings=("ttl_seconds=30",),
        oracle_source="cache.py",
    ),
    BenchTask(
        slug="retry_backoff",
        repo=BENCH_ROOT / "repos" / "retry_backoff",
        task="Explain why a downstream outage causes this service's "
        "requests to back off for several seconds before giving up entirely",
        required_substrings=("max_attempts=5",),
        oracle_source="backoff.py",
    ),
    BenchTask(
        slug="pagination_limit",
        repo=BENCH_ROOT / "repos" / "pagination_limit",
        task="Explain why a customer with 40 orders only sees 25 of them "
        "in the export tool",
        required_substrings=("page_size=25",),
        oracle_source="orders_export.py",
    ),
    # --- decision-record: the fact is "why we chose X over Y"; the correct
    # action is to respect the decision rather than what looks locally
    # sensible. Also no constraint doc in these repos.
    BenchTask(
        slug="queue_delivery_decision",
        repo=BENCH_ROOT / "repos" / "queue_delivery_decision",
        task="A teammate wants to switch the order events queue to "
        "at-most-once delivery to simplify consumer code -- evaluate it",
        required_substrings=("at-least-once", "idempotent"),
        oracle_source="DECISIONS.md",
    ),
    BenchTask(
        slug="sync_vs_async_decision",
        repo=BENCH_ROOT / "repos" / "sync_vs_async_decision",
        task="Propose moving email sending back into the synchronous "
        "request handler for lower latency -- evaluate it",
        required_substrings=("thread pool exhaustion", "asynchronously"),
        oracle_source="DECISIONS.md",
    ),
    BenchTask(
        slug="soft_delete_decision",
        repo=BENCH_ROOT / "repos" / "soft_delete_decision",
        task="Propose switching customer record deletion from soft delete "
        "to a hard SQL DELETE for simplicity -- evaluate it",
        required_substrings=("soft delete", "audit trail"),
        oracle_source="DECISIONS.md",
    ),
    # --- dependency-graph: purpose-built to stress cross-file dependency
    # propagation (context_compiler.graph), not lexical relevance. The task
    # wording lexically matches handler.py ("refund", "handler", "amount")
    # but the actual constraint lives in validators.py, worded with
    # different vocabulary ("reimbursement", "surpass", "draining funds")
    # deliberately chosen to minimize lexical overlap with the task text.
    # validators.py is only reachable by knowing handler.py imports it.
    BenchTask(
        slug="dependency_graph",
        repo=BENCH_ROOT / "repos" / "dependency_graph",
        task="Customers are being sent payouts bigger than what they were "
        "originally charged at checkout; patch the endpoint so that stops happening",
        required_substrings=("cannot surpass the initial transaction total", "draining more funds"),
        oracle_source="validators.py",
    ),
]
