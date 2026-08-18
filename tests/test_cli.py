import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from context_compiler import __version__
from context_compiler.cli import main


def _run(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(args)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_version_flag(self):
        code, out, _ = _run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_ingest_missing_path_is_a_clean_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "ctx.db")
            code, _, err = _run(["--db", db, "ingest", str(Path(tmp) / "does-not-exist")])
            self.assertEqual(code, 2)
            self.assertIn("error:", err)

    def test_compile_invalid_budget_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "ctx.db")
            self.assertEqual(_run(["--db", db, "init"])[0], 0)
            code, _, err = _run(["--db", db, "compile", "some task", "--budget", "0"])
            self.assertEqual(code, 2)
            self.assertIn("budget must be > 0", err)

    def test_tiktoken_without_optional_dependency_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "ctx.db")
            self.assertEqual(_run(["--db", db, "init"])[0], 0)
            code, _, err = _run(
                ["--db", db, "--tokenizer", "tiktoken", "compile", "task", "--budget", "100"]
            )
            self.assertEqual(code, 1)
            self.assertIn("error:", err)

    def test_add_and_compile_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "ctx.db")
            self.assertEqual(_run(["--db", db, "init"])[0], 0)
            code, out, _ = _run(
                [
                    "--db",
                    db,
                    "add",
                    "--title",
                    "Invariant",
                    "--content",
                    "Never replay authorization codes.",
                    "--kind",
                    "constraint",
                    "--pinned",
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Invariant", out)

            code, out, _ = _run(["--db", db, "compile", "Never replay codes", "--budget", "200"])
            self.assertEqual(code, 0)
            self.assertIn("Never replay authorization codes", out)

    def test_delete_unknown_item_returns_not_found_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "ctx.db")
            self.assertEqual(_run(["--db", db, "init"])[0], 0)
            code, _, _ = _run(["--db", db, "delete", "does-not-exist"])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
