"""Tests for benchmarks/llm_judge_eval.py -- fully mocked, no real network/API calls."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest import mock

BENCH_ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

import llm_judge_eval  # noqa: E402


def _write_task_context(tmp: Path, task: str, context: str) -> dict:
    task_file = tmp / "task.txt"
    context_file = tmp / "context.txt"
    task_file.write_text(task, encoding="utf-8")
    context_file.write_text(context, encoding="utf-8")
    return {"CTXC_TASK_FILE": str(task_file), "CTXC_CONTEXT_FILE": str(context_file)}


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


def _fake_anthropic_module(reply_text: str) -> ModuleType:
    module = ModuleType("anthropic")

    class FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse(reply_text)

    class FakeAnthropic:
        def __init__(self, *a, **kw) -> None:
            self.messages = FakeMessages()

    class APIStatusError(Exception):
        def __init__(self, message: str = "err", status_code: int = 500) -> None:
            super().__init__(message)
            self.message = message
            self.status_code = status_code

    module.Anthropic = FakeAnthropic
    module.AuthenticationError = type("AuthenticationError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.APIStatusError = APIStatusError
    return module


class LlmJudgeEvalTests(unittest.TestCase):
    def test_missing_env_vars_fails_gracefully(self):
        env = dict(os.environ)
        env.pop("CTXC_TASK_FILE", None)
        env.pop("CTXC_CONTEXT_FILE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(llm_judge_eval.main(), 1)

    def test_missing_anthropic_package_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _write_task_context(Path(tmp), "do X", "some context")
            with mock.patch.dict(os.environ, env), mock.patch.dict(
                sys.modules, {"anthropic": None}
            ):
                self.assertEqual(llm_judge_eval.main(), 1)

    def test_parses_sufficient_verdict(self):
        fake_module = _fake_anthropic_module(
            json.dumps({"sufficient": True, "reason": "has the invariant"})
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = _write_task_context(Path(tmp), "do X", "some context")
            with mock.patch.dict(os.environ, env), mock.patch.dict(
                sys.modules, {"anthropic": fake_module}
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = llm_judge_eval.main()
                self.assertEqual(rc, 0)
                payload = json.loads(buf.getvalue())
                self.assertTrue(payload["success"])
                self.assertEqual(payload["score"], 1.0)

    def test_parses_insufficient_verdict(self):
        fake_module = _fake_anthropic_module(
            json.dumps({"sufficient": False, "reason": "missing the constraint"})
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = _write_task_context(Path(tmp), "do X", "some context")
            with mock.patch.dict(os.environ, env), mock.patch.dict(
                sys.modules, {"anthropic": fake_module}
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = llm_judge_eval.main()
                self.assertEqual(rc, 0)
                payload = json.loads(buf.getvalue())
                self.assertFalse(payload["success"])
                self.assertEqual(payload["score"], 0.0)

    def test_malformed_judge_reply_is_treated_as_failure(self):
        fake_module = _fake_anthropic_module("not json")
        with tempfile.TemporaryDirectory() as tmp:
            env = _write_task_context(Path(tmp), "do X", "some context")
            with mock.patch.dict(os.environ, env), mock.patch.dict(
                sys.modules, {"anthropic": fake_module}
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = llm_judge_eval.main()
                self.assertEqual(rc, 0)
                payload = json.loads(buf.getvalue())
                self.assertFalse(payload["success"])


if __name__ == "__main__":
    unittest.main()
