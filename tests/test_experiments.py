import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextCompiler, ContextStore
from context_compiler.experiments import EvaluationResult, ExperimentRunner, b95


class ExperimentTests(unittest.TestCase):
    def test_sweep_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            store.add(
                title="critical OAuth invariant",
                content="Safari OAuth codes are single use. Never retry them.",
                kind="constraint",
                pinned=True,
                importance=1.0,
                omission_risk=1.0,
            )
            store.add(title="auth.py", content="def safari_oauth(): return True", kind="code")
            compiler = ContextCompiler(store)

            def evaluator(task, context):
                ok = "Safari" in context and "single use" in context
                return EvaluationResult(success=ok, score=float(ok))

            runner = ExperimentRunner(compiler, evaluator)
            points = runner.budget_sweep("Fix Safari OAuth", [80, 160, 320], repeats=2)
            self.assertEqual(len(points), 3)
            self.assertIsNotNone(b95(points, 1.0))

            compiled = compiler.compile("Fix Safari OAuth", 320)
            effects = runner.deletion_test("Fix Safari OAuth", compiled, repeats=1)
            self.assertGreaterEqual(len(effects), 1)


if __name__ == "__main__":
    unittest.main()
