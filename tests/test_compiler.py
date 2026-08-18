import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextCompiler, ContextStore
from context_compiler.scoring import LEARNED_WEIGHTS_V1, ScoringWeights


class CompilerTests(unittest.TestCase):
    def test_budget_and_pinned_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            constraint = store.add(
                title="OAuth replay invariant",
                content=(
                    "OAuth authorization codes are single use. Never automatically replay "
                    "the exchange after an ambiguous failure."
                ),
                kind="constraint",
                pinned=True,
                importance=1.0,
                omission_risk=1.0,
            )
            store.add(
                title="auth.py",
                source="auth.py",
                kind="code",
                content="def callback(browser, code):\n    return exchange(code)\n",
                tags=["oauth", "safari"],
                importance=0.7,
            )
            for i in range(20):
                store.add(
                    title=f"noise-{i}.txt",
                    content="unrelated rendering cache image css analytics text " * 20,
                    kind="doc",
                    importance=0.1,
                )

            compiler = ContextCompiler(store)
            result = compiler.compile("Fix Safari OAuth callback", budget=260)
            self.assertLessEqual(result.used_tokens, 260)
            self.assertIn(constraint.id, {s.item_id for s in result.selections})
            self.assertIn("OAuth", result.text)

    def test_learned_weights_v1_is_a_usable_alternative_preset(self):
        # Not the default -- must be opted into explicitly. See
        # benchmarks/README.md "Phase 2" for how it was fit and its caveats.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            constraint = store.add(
                title="OAuth replay invariant",
                content="Never automatically replay the exchange after an ambiguous failure.",
                kind="constraint",
                pinned=True,
                importance=1.0,
                omission_risk=1.0,
            )
            compiler = ContextCompiler(store, weights=LEARNED_WEIGHTS_V1)
            result = compiler.compile("Fix Safari OAuth callback", budget=260)
            self.assertIn(constraint.id, {s.item_id for s in result.selections})
            # recency/pin_bonus were never testable by the fitting benchmark
            # (see scoring.py) and must be untouched from the hand-tuned default.
            default = ScoringWeights()
            self.assertEqual(LEARNED_WEIGHTS_V1.recency, default.recency)
            self.assertEqual(LEARNED_WEIGHTS_V1.pin_bonus, default.pin_bonus)

    def test_compile_is_deterministic_across_separate_ingests(self):
        # Regression test for a real bug found via benchmarks/run.py: a
        # budget-boundary compile() result flipped between success and
        # failure across repeated runs against *identical* repo content,
        # traced to two compounding nondeterminism sources now fixed --
        # ContextStore.list()'s ORDER BY lacked a stable tiebreak for items
        # sharing the same `updated_at`, and randomly-generated item ids
        # (embedded in every rendered [CTX:...] header) tokenized as either
        # one or two tokens depending on whether the random hex happened to
        # start with a digit, shifting reported token counts by +-1 at
        # random. Re-ingesting the same content into a fresh store must now
        # produce byte-identical compile() output at a tight, boundary-prone
        # budget.
        # Note: raw output *text* legitimately differs run to run -- it embeds
        # each item's freshly-generated random id -- so compare used_tokens
        # and the (source, level) selection set instead, which must not.
        def compile_once() -> tuple[int, tuple[tuple[str, str], ...]]:
            with tempfile.TemporaryDirectory() as tmp:
                store = ContextStore(Path(tmp) / "ctx.db")
                store.add(
                    title="POLICY.md",
                    source="POLICY.md",
                    kind="constraint",
                    content=(
                        "Payment retry policy.\n\n"
                        "Never retry a charge without a confirmed idempotency key; "
                        "a blind retry after a gateway timeout can double-bill the customer."
                    ),
                )
                store.add(title="charge.py", source="charge.py", kind="code", content="def charge(): ...")
                store.add(title="gateway.py", source="gateway.py", kind="code", content="def call(): ...")
                store.add(title="README.md", source="README.md", kind="doc", content="Payments service.")
                compiler = ContextCompiler(store)
                result = compiler.compile("Fix duplicate charges on gateway timeout", budget=150)
                selection_shape = tuple(sorted((s.source, s.level.value) for s in result.selections))
                return result.used_tokens, selection_shape

        outputs = {compile_once() for _ in range(15)}
        self.assertEqual(len(outputs), 1, f"compile() shape varied across ingests: {outputs}")

    def test_verbose_task_text_does_not_starve_the_working_set(self):
        # Regression test for a real bug found by running a real SWE-bench
        # Verified instance (pylint-dev/pylint-7080) through the pipeline:
        # a genuine GitHub issue can paste a long CLI/log dump into its body
        # (thousands of tokens). The header embeds the task verbatim and
        # isn't compressed the way stored items are, so at a small-to-medium
        # budget the header alone consumed the *entire* budget, leaving zero
        # tokens -- and zero selections -- for the actual working set. The
        # 15 short, one-line synthetic benchmark tasks never exercised this
        # path. compile() must now always reserve most of the budget for
        # real context regardless of how verbose the task text is.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            store.add(
                title="cache.py",
                source="cache.py",
                kind="code",
                content="TTL_SECONDS = 30\ndef get(key): ...",
                importance=0.8,
            )
            huge_task = "Investigate stale cached prices.\n" + (
                "irrelevant pasted log line with unique noise words " * 400
            )
            compiler = ContextCompiler(store)
            result = compiler.compile(huge_task, budget=500)
            self.assertLessEqual(result.used_tokens, 500)
            self.assertGreater(
                len(result.selections), 0, "verbose task text left no room for any context item"
            )
            self.assertTrue(
                any("truncated" in a for a in result.audit),
                "expected an audit entry noting the task text was truncated",
            )

    def test_default_candidate_limit_scales_with_budget(self):
        # Regression test for a second bug found via the same real SWE-bench
        # instance: retrieval hard-truncated to the top `max_candidates`
        # (120) *before* scoring, independent of how large the budget was.
        # A real fix file that TF-IDF ranked 181st out of ~3000 files was
        # therefore permanently unreachable at *any* budget, not just small
        # ones. The default candidate limit must now grow with budget so a
        # generous budget gets a deeper look, while small-budget behavior
        # (where 120 candidates was already more than could ever be
        # afforded) stays exactly as before.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            task = "investigate the timeout retry bug in the payment gateway"
            for i in range(130):
                store.add(
                    title=f"filler-{i:03d}.py",
                    source=f"aaa_filler_{i:03d}.py",
                    kind="code",
                    content="timeout retry bug payment gateway " * 10,
                    importance=0.5,
                )
            target = store.add(
                title="unrelated-but-important.py",
                source="zzz_target.py",
                kind="code",
                content="widget rendering color palette layout spacing",
                importance=0.9,
                omission_risk=0.6,
            )
            compiler = ContextCompiler(store)

            small = compiler.compile(task, budget=2000)  # 2000 // 40 == 50 < 120, unchanged floor
            small_considered = {s.item_id for s in small.selections} | set(small.omitted_ids)
            self.assertNotIn(
                target.id,
                small_considered,
                "target should be past the default candidate cutoff at a small budget",
            )

            large = compiler.compile(task, budget=6000)  # 6000 // 40 == 150 > 130
            large_considered = {s.item_id for s in large.selections} | set(large.omitted_ids)
            self.assertIn(
                target.id,
                large_considered,
                "a larger budget should widen the candidate pool enough to reach the target",
            )

    def test_reversible_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            item = store.add(title="doc", content="alpha beta gamma", kind="doc")
            compiler = ContextCompiler(store)
            expanded = compiler.expand(item.id, level="L4")
            self.assertIn("alpha beta gamma", expanded["text"])
            self.assertEqual(expanded["item"]["content"], "alpha beta gamma")


if __name__ == "__main__":
    unittest.main()
