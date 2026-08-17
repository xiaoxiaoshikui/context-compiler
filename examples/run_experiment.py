from __future__ import annotations

import json
import tempfile
from pathlib import Path

from context_compiler import ContextCompiler, ContextStore
from context_compiler.experiments import EvaluationResult, ExperimentRunner, b95
from context_compiler.ingest import RepositoryIngestor


def evaluator(task: str, context: str) -> EvaluationResult:
    """Deterministic toy evaluator used only to demonstrate the experiment API."""
    low = context.lower()
    has_auth = "oauth" in low and "safari" in low
    has_constraint = "single-use" in low or "never automatically replay" in low
    score = 0.5 * float(has_auth) + 0.5 * float(has_constraint)
    return EvaluationResult(success=has_auth and has_constraint, score=score)


def main() -> None:
    demo_repo = Path(__file__).parent / "demo_repo"
    with tempfile.TemporaryDirectory() as tmp:
        store = ContextStore(Path(tmp) / "demo.db")
        RepositoryIngestor(store).ingest(demo_repo)
        compiler = ContextCompiler(store)
        runner = ExperimentRunner(compiler, evaluator)
        task = "Fix the Safari OAuth callback regression while preserving the no-replay security invariant"
        points = runner.budget_sweep(task, [120, 180, 260, 400, 650, 1000], repeats=3)
        print(json.dumps([p.to_dict() for p in points], indent=2))
        print("B95 relative to perfect baseline:", b95(points, baseline_success_rate=1.0))

        compiled = compiler.compile(task, 650)
        effects = runner.deletion_test(task, compiled, repeats=2)
        print("\nDeletion effects:")
        print(json.dumps([e.to_dict() for e in effects], indent=2))


if __name__ == "__main__":
    main()
