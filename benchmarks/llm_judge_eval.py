"""Optional LLM-judge evaluator for the benchmark harness.

Not part of the default `python benchmarks/run.py` sweep -- that stays on
the free/instant keyword evaluator so routine runs and CI cost nothing.
This script matches the `context_compiler.experiments.JsonCommandEvaluator`
contract: read CTXC_TASK_FILE / CTXC_CONTEXT_FILE, print one JSON line to
stdout.

Requires the optional `anthropic` package and a real ANTHROPIC_API_KEY --
every invocation is a billed API call:

    pip install -e '.[llm_judge]'
    python benchmarks/run.py --evaluator llm_judge

This is a judge over the compiled context text ("does this give the
engineer what they need"), not a full patch-and-test coding-agent loop --
see benchmarks/README.md for why that distinction matters and what a real
next step would look like.
"""

from __future__ import annotations

import json
import os
import sys

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 256

_JUDGE_PROMPT = """You are judging whether a compiled context bundle gives an engineer everything they need to safely and correctly complete a task -- not whether the bundle is well-written.

TASK:
{task}

COMPILED CONTEXT GIVEN TO THE ENGINEER:
{context}

Respond with exactly one JSON object and nothing else:
{{"sufficient": true or false, "reason": "one sentence"}}
"""


def main() -> int:
    task_file = os.environ.get("CTXC_TASK_FILE")
    context_file = os.environ.get("CTXC_CONTEXT_FILE")
    if not task_file or not context_file:
        print("CTXC_TASK_FILE and CTXC_CONTEXT_FILE must be set", file=sys.stderr)
        return 1

    try:
        import anthropic
    except ImportError:
        print(
            "the 'anthropic' package is required: pip install -e '.[llm_judge]'",
            file=sys.stderr,
        )
        return 1

    with open(task_file, encoding="utf-8") as f:
        task = f.read()
    with open(context_file, encoding="utf-8") as f:
        context = f.read()
    prompt = _JUDGE_PROMPT.format(task=task, context=context)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        print("ANTHROPIC_API_KEY is missing or invalid", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"network error calling the Anthropic API: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"Anthropic API error ({exc.status_code}): {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # client construction (e.g. no credentials) or unexpected SDK error
        print(f"could not call the Anthropic API: {exc}", file=sys.stderr)
        return 1

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        verdict = json.loads(text)
        success = bool(verdict.get("sufficient", False))
        reason = verdict.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        success = False
        reason = f"could not parse judge response: {text[:200]!r}"

    print(
        json.dumps(
            {
                "success": success,
                "score": 1.0 if success else 0.0,
                "metadata": {"reason": reason, "model": _MODEL},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
