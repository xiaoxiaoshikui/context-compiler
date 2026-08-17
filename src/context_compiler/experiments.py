from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Protocol

from .compiler import ContextCompiler
from .models import CompiledContext, SelectedContext


@dataclass(slots=True)
class EvaluationResult:
    success: bool
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class Evaluator(Protocol):
    def __call__(self, task: str, context: str) -> EvaluationResult: ...


@dataclass(slots=True)
class SweepPoint:
    budget: int
    runs: int
    successes: int
    success_rate: float
    mean_score: float
    mean_used_tokens: float
    mean_items: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DeletionEffect:
    item_id: str
    title: str
    baseline_success_rate: float
    ablated_success_rate: float
    delta_success_rate: float
    baseline_score: float
    ablated_score: float
    delta_score: float
    removed_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRunner:
    def __init__(self, compiler: ContextCompiler, evaluator: Evaluator) -> None:
        self.compiler = compiler
        self.evaluator = evaluator

    def budget_sweep(
        self,
        task: str,
        budgets: Iterable[int],
        *,
        repeats: int = 3,
    ) -> list[SweepPoint]:
        points: list[SweepPoint] = []
        for budget in budgets:
            results: list[EvaluationResult] = []
            used: list[int] = []
            item_counts: list[int] = []
            for _ in range(repeats):
                compiled = self.compiler.compile(task, int(budget))
                results.append(self.evaluator(task, compiled.text))
                used.append(compiled.used_tokens)
                item_counts.append(len(compiled.selections))
            successes = sum(1 for r in results if r.success)
            points.append(
                SweepPoint(
                    budget=int(budget),
                    runs=repeats,
                    successes=successes,
                    success_rate=successes / repeats,
                    mean_score=statistics.fmean(r.score for r in results),
                    mean_used_tokens=statistics.fmean(used),
                    mean_items=statistics.fmean(item_counts),
                )
            )
        return points

    def deletion_test(
        self,
        task: str,
        compiled: CompiledContext,
        *,
        repeats: int = 3,
    ) -> list[DeletionEffect]:
        baseline = [self.evaluator(task, compiled.text) for _ in range(repeats)]
        baseline_rate = sum(1 for r in baseline if r.success) / repeats
        baseline_score = statistics.fmean(r.score for r in baseline)
        effects: list[DeletionEffect] = []
        for selection in compiled.selections:
            text = build_text_without(compiled, selection.item_id)
            ablated = [self.evaluator(task, text) for _ in range(repeats)]
            rate = sum(1 for r in ablated if r.success) / repeats
            score = statistics.fmean(r.score for r in ablated)
            effects.append(
                DeletionEffect(
                    item_id=selection.item_id,
                    title=selection.title,
                    baseline_success_rate=baseline_rate,
                    ablated_success_rate=rate,
                    delta_success_rate=baseline_rate - rate,
                    baseline_score=baseline_score,
                    ablated_score=score,
                    delta_score=baseline_score - score,
                    removed_tokens=selection.token_count,
                )
            )
        effects.sort(key=lambda e: (e.delta_success_rate, e.delta_score), reverse=True)
        return effects

    def greedy_minimize(
        self,
        task: str,
        compiled: CompiledContext,
        *,
        min_success_rate: float = 0.95,
        repeats: int = 5,
    ) -> dict:
        """Empirically remove low-impact items while preserving target success.

        This is intentionally expensive: each proposed removal is evaluated.
        Use it to estimate an empirical MSC on small benchmark subsets.
        """
        current = list(compiled.selections)
        history: list[dict] = []

        def eval_set(selections: list[SelectedContext]) -> tuple[float, float, str]:
            text = build_text(task, selections)
            results = [self.evaluator(task, text) for _ in range(repeats)]
            success_rate = sum(1 for r in results if r.success) / repeats
            score = statistics.fmean(r.score for r in results)
            return success_rate, score, text

        base_rate, base_score, _ = eval_set(current)
        target = min(base_rate, min_success_rate)

        while len(current) > 1:
            candidates: list[tuple[float, float, int, list[SelectedContext], str]] = []
            for idx, item in enumerate(current):
                proposal = current[:idx] + current[idx + 1 :]
                rate, score, text = eval_set(proposal)
                candidates.append((rate, score, idx, proposal, text))
            feasible = [c for c in candidates if c[0] >= target]
            if not feasible:
                break
            # Prefer the feasible proposal with the fewest tokens; use score as tie-breaker.
            feasible.sort(
                key=lambda c: (
                    sum(s.token_count for s in c[3]),
                    -c[1],
                )
            )
            rate, score, idx, proposal, _ = feasible[0]
            removed = current[idx]
            current = proposal
            history.append(
                {
                    "removed_item_id": removed.item_id,
                    "removed_title": removed.title,
                    "success_rate": rate,
                    "score": score,
                    "remaining_items": len(current),
                    "remaining_body_tokens": sum(s.token_count for s in current),
                }
            )

        final_rate, final_score, final_text = eval_set(current)
        return {
            "baseline_success_rate": base_rate,
            "baseline_score": base_score,
            "target_success_rate": target,
            "final_success_rate": final_rate,
            "final_score": final_score,
            "initial_items": len(compiled.selections),
            "final_items": len(current),
            "initial_body_tokens": sum(s.token_count for s in compiled.selections),
            "final_body_tokens": sum(s.token_count for s in current),
            "history": history,
            "remaining": [s.to_dict() for s in current],
            "text": final_text,
        }


def build_text(task: str, selections: list[SelectedContext]) -> str:
    header = (
        "[CONTEXT-COMPILER]\n"
        "Use the following task-specific working set.\n"
        f"TASK: {task.strip()}"
    )
    if not selections:
        return header
    return header + "\n\n" + "\n\n".join(s.text for s in selections)


def build_text_without(compiled: CompiledContext, item_id: str) -> str:
    return build_text(
        compiled.task,
        [s for s in compiled.selections if s.item_id != item_id],
    )


def b95(points: list[SweepPoint], baseline_success_rate: float) -> int | None:
    threshold = 0.95 * baseline_success_rate
    eligible = [p.budget for p in points if p.success_rate >= threshold]
    return min(eligible) if eligible else None


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

class JsonCommandEvaluator:
    """Adapter for plugging any external agent/test harness into experiments.

    The command is executed with two additional environment variables:
      CTXC_TASK_FILE    path to a UTF-8 task file
      CTXC_CONTEXT_FILE path to the compiled context file

    It must print one JSON object to stdout, for example:
      {"success": true, "score": 1.0, "metadata": {"tests": "passed"}}
    """

    def __init__(self, command: list[str], *, timeout_seconds: int = 900) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds

    def __call__(self, task: str, context: str) -> EvaluationResult:
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="ctxc-eval-") as tmp:
            root = Path(tmp)
            task_file = root / "task.txt"
            context_file = root / "context.txt"
            task_file.write_text(task, encoding="utf-8")
            context_file.write_text(context, encoding="utf-8")
            env = dict(os.environ)
            env["CTXC_TASK_FILE"] = str(task_file)
            env["CTXC_CONTEXT_FILE"] = str(context_file)
            proc = subprocess.run(
                self.command,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                return EvaluationResult(
                    success=False,
                    score=0.0,
                    metadata={
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-4000:],
                        "stderr": proc.stderr[-4000:],
                    },
                )
            try:
                data = json.loads(proc.stdout.strip())
            except json.JSONDecodeError:
                return EvaluationResult(
                    success=False,
                    score=0.0,
                    metadata={"error": "evaluator did not emit JSON", "stdout": proc.stdout[-4000:]},
                )
            return EvaluationResult(
                success=bool(data.get("success", False)),
                score=float(data.get("score", 1.0 if data.get("success") else 0.0)),
                metadata=data.get("metadata", {}),
            )
