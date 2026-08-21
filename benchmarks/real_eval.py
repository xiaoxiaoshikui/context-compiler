"""Real-execution evaluation: does "minimum sufficient context" exist, and
how close does context_compiler get to an oracle ceiling?

Every other evaluator in this repo (`tasks.py`'s keyword-conjunction check,
`llm_judge_eval.py`'s judge-over-text) is a proxy: none of them execute the
proposed fix. This script does. For a real SWE-bench Verified instance, it:

  1. Downloads the real repository at the issue's base_commit and ingests it
     into a `context_compiler` store.
  2. Builds a context under a token budget via one of several *policies*:
       - oracle:  issue + full text of the files the gold patch touches +
                  the FAIL_TO_PASS/PASS_TO_PASS test files (ground truth
                  file *identity* only -- never the gold patch's diff
                  content, so the answer is never leaked). This is a
                  ceiling: "if we magically knew which files mattered, how
                  far can this be compressed?"
       - random:  a fixed-seed random ordering of every ingested file.
       - ours:    context_compiler's own `ContextCompiler.compile()` output
                  (its real L0-L4 compression, not a stripped-down version).
  3. Asks a real model (DeepSeek, cheap enough to run this many times) to
     propose a fix, using a whole-function-rewrite patch format (see
     "Why whole-function rewrite" below).
  4. Applies the resulting patch to a fresh checkout and runs the instance's
     real hidden tests via the official `swebench` evaluation harness
     (Docker), producing a real resolved/unresolved verdict.

Why whole-function rewrite, not a diff or SEARCH/REPLACE snippet:
    Two earlier approaches were tried and both produced real, reproducible
    corruption. Asking the model to hand-write a unified diff fails `git
    apply` outright (models get hunk line numbers wrong). Asking for
    SEARCH/REPLACE snippets (find this exact text, replace with this)
    works when the model sees verbatim original text, but breaks when it
    only sees compressed/summarized context (as `ours` intentionally does):
    the model transcribes an *approximate* snippet, so locating it requires
    fuzzy text matching -- and an arbitrary line range has no
    guaranteed-safe boundary to fuzzy-match and splice in whitespace
    -sensitive code. Confirmed twice: a dropped trailing newline glued two
    statements onto one line (SyntaxError); a fuzzy match left an orphaned
    line mid-block (IndentationError). A whole function/method is the
    smallest unit with an unambiguous boundary: it's located deterministically
    via Python's `ast` module (by name, not by text similarity), and the
    replacement is syntax-checked before splicing. This eliminates the
    boundary-corruption bugs entirely. The real remaining risk, observed
    directly (see benchmarks/README.md), is *content drift*: because the
    model reconstructs the whole function from its own understanding rather
    than surgically touching only what changed, it can silently drop or
    alter code it didn't mean to touch -- especially when it only saw a
    compressed view. That's now a real property of the method being
    measured, not a bug in this script.

Requirements (none of this is installed by default -- see README):
    pip install pyarrow                    # read the SWE-bench parquet directly
    pip install "swebench>=3,<4"           # the *classic* eval harness; 4.x/5.x
                                            # changed to a registry-image model
                                            # that doesn't work against the raw
                                            # HF dataset used here
    Docker Desktop running                 # the harness builds/pulls real
                                            # per-instance execution images
    DEEPSEEK_API_KEY in the environment    # every patch-generation call is
                                            # billed (cheap, but not free)

This is a research script, not a library the core package depends on --
consistent with `llm_judge_eval.py`, it lives here precisely so nobody pays
its cost by default. See "Real findings" in benchmarks/README.md for what
running it has actually shown.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import difflib
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import textwrap
import time
import urllib.request
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
CACHE_DIR = BENCH_ROOT / ".cache" / "swebench"
REPOS_DIR = CACHE_DIR / "repos"
DBS_DIR = CACHE_DIR / "dbs"
RUNS_DIR = CACHE_DIR / "runs"
RESULTS_PATH = BENCH_ROOT / "results" / "real_eval_results.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from context_compiler.compiler import ContextCompiler  # noqa: E402
from context_compiler.store import ContextStore  # noqa: E402
from context_compiler.ingest import RepositoryIngestor  # noqa: E402
from context_compiler.tokenizer import HeuristicTokenCounter  # noqa: E402

COUNTER = HeuristicTokenCounter()
DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)

VERIFIED_PARQUET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/"
    "resolve/main/data/test-00000-of-00001.parquet"
)


# --------------------------------------------------------------------------
# SWE-bench Verified data loading (no `datasets` dependency -- read the
# published parquet directly with pyarrow, caching it locally after the
# first download).
# --------------------------------------------------------------------------

def load_verified_dataset() -> list[dict]:
    import pyarrow.parquet as pq

    local = CACHE_DIR / "swebench_verified_test.parquet"
    if not local.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading SWE-bench Verified (one-time) -> {local}")
        with urllib.request.urlopen(VERIFIED_PARQUET_URL, timeout=120) as resp:
            local.write_bytes(resp.read())
    return pq.read_table(local).to_pylist()


def select_tasks(rows: list[dict], instance_ids: list[str]) -> list[dict]:
    by_id = {r["instance_id"]: r for r in rows}
    missing = [i for i in instance_ids if i not in by_id]
    if missing:
        raise ValueError(f"instance_ids not found in SWE-bench Verified: {missing}")
    return [by_id[i] for i in instance_ids]


# --------------------------------------------------------------------------
# Repo fetch + ingest
# --------------------------------------------------------------------------

def fetch_repo(repo: str, commit: str, instance_id: str) -> Path:
    dest = REPOS_DIR / instance_id
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        top = members[0].name.split("/")[0] + "/"
        for m in members:
            if not m.name.startswith(top):
                continue
            rel = m.name[len(top):]
            if not rel:
                continue
            m.name = rel
        tf.extractall(dest, members=members, filter="data")
    return dest


def ingest_repo(instance_id: str, repo_path: Path) -> ContextStore:
    db_path = DBS_DIR / f"{instance_id}.db"
    DBS_DIR.mkdir(parents=True, exist_ok=True)
    fresh = not db_path.exists()
    store = ContextStore(str(db_path))
    if fresh:
        RepositoryIngestor(store).ingest(repo_path)
    return store


# --------------------------------------------------------------------------
# Oracle universe: ground-truth *file identity* only, never the gold patch
# diff content -- see module docstring.
# --------------------------------------------------------------------------

def oracle_file_set(row: dict) -> set[str]:
    files = set()
    for a, b in DIFF_HEADER_RE.findall(row["patch"]):
        files.add(a)
        files.add(b)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for node_id in json.loads(row[key]):
            files.add(node_id.split("::")[0])
    return files


def build_oracle_universe(row: dict, store: ContextStore) -> str:
    files = sorted(oracle_file_set(row))
    by_source = {i.source: i for i in store.list(limit=10000)}
    parts = [f"ISSUE:\n{row['problem_statement'].strip()}"]
    for f in files:
        item = by_source.get(f)
        if item is not None:
            parts.append(f"### FILE: {f}\n{item.content}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Context-construction policies
# --------------------------------------------------------------------------

def context_oracle(row: dict, store: ContextStore, budget: int) -> str:
    return COUNTER.truncate(build_oracle_universe(row, store), budget)


def context_random(store: ContextStore, budget: int, seed: int = 0) -> str:
    items = store.list(limit=10000)
    order = items[:]
    random.Random(seed).shuffle(order)
    parts = [f"### FILE: {it.source}\n{it.content}" for it in order]
    return COUNTER.truncate("\n\n".join(parts), budget)


def context_ours(task_text: str, store: ContextStore, budget: int) -> str:
    """context_compiler's real output -- its L0-L4 compression and all.

    (An earlier attempt fed the model full raw text of whichever files
    compile() selected instead, to make an exact-match patch mechanism
    work. That failed for an interesting reason of its own: a single
    highly-relevant file's full text can exceed the whole budget alone
    (confirmed: one core module was 16.6k tokens against an 8k budget),
    silently crowding out the actual fix location -- which strips out
    compile()'s core mechanism (spend a little on many relevant files
    instead of everything on one) rather than fairly testing it. Fixed
    downstream instead, in the patch mechanism -- see module docstring.)
    """
    return ContextCompiler(store).compile(task_text, budget).text


CONTEXT_POLICIES = {
    "oracle": lambda row, store, budget, seed: context_oracle(row, store, budget),
    "random": lambda row, store, budget, seed: context_random(store, budget, seed),
    "ours": lambda row, store, budget, seed: context_ours(row["problem_statement"], store, budget),
}


# --------------------------------------------------------------------------
# Patch generation: whole-function rewrite (see module docstring for why)
# --------------------------------------------------------------------------

FUNCTION_REWRITE_SYSTEM_PROMPT = (
    "You are a careful software engineer. You will be given a GitHub issue and "
    "some repository context (which may be summarized/compressed due to a "
    "token budget). Propose a fix by rewriting one or more whole functions or "
    "methods, using exactly this format:\n\n"
    "<<<<<<< REWRITE_FUNCTION path/to/file.py::function_name\n"
    "def function_name(...):\n"
    "    ...\n"
    "    (the complete new body of the function, nothing omitted)\n"
    ">>>>>>> END\n\n"
    "For a method inside a class, use ClassName.method_name as the target, "
    "and indent the function exactly as it appears inside the class (usually "
    "4 spaces before 'def'). Write out the ENTIRE function from 'def' to its "
    "last line -- it replaces the whole function, not just the changed part. "
    "Only rewrite functions that actually need to change. If the fix requires "
    "adding a genuinely new function (not modifying an existing one), you may "
    "invent a target function_name that doesn't exist yet; it will be "
    "reported as unresolved rather than guessed at. Output ONLY these "
    "blocks, no explanation."
)

_FUNC_BLOCK_RE = re.compile(
    r"<{5,}\s*REWRITE_FUNCTION\s+(?P<target>\S+)\s*\n(?P<body>.*?)\n>{5,}\s*END",
    re.DOTALL,
)


def parse_function_rewrites(text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in _FUNC_BLOCK_RE.finditer(text):
        target = m["target"]
        if "::" not in target:
            continue
        path, func_name = target.split("::", 1)
        out.append((path, func_name, m["body"]))
    return out


def _find_function(source: str, name: str):
    """Locate a FunctionDef/AsyncFunctionDef by exact qualified name
    (ClassName.method_name) or, if unambiguous, by bare name alone.
    Deterministic -- no similarity threshold, no guessing."""
    tree = ast.parse(source)
    candidates: list[tuple[str, str, ast.AST]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _visit_func(self, node):
            qualified = ".".join(self.stack + [node.name])
            candidates.append((qualified, node.name, node))
            self.generic_visit(node)

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func

    _Visitor().visit(tree)
    for qualified, _bare, node in candidates:
        if qualified == name:
            return node
    matches = [node for _q, bare, node in candidates if bare == name]
    return matches[0] if len(matches) == 1 else None


def _function_span(source: str, node) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end_line = node.end_lineno
    start = sum(len(l) for l in lines[:start_line - 1])
    end = sum(len(l) for l in lines[:end_line])
    return start, end


def function_rewrites_to_patch(rewrites: list[tuple[str, str, str]], by_source: dict) -> tuple[str, list[str]]:
    edits: dict[str, str] = {}
    errors = []
    for path, func_name, new_body in rewrites:
        item = by_source.get(path)
        if item is None:
            errors.append(f"unknown file in REWRITE_FUNCTION block: {path}")
            continue
        original = edits.get(path, item.content)
        try:
            node = _find_function(original, func_name)
        except SyntaxError as exc:
            errors.append(f"{path} does not parse as Python, cannot locate functions: {exc}")
            continue
        if node is None:
            errors.append(f"function {func_name!r} not found (or ambiguous) in {path}")
            continue
        new_body_text = new_body.rstrip("\n") + "\n"
        try:
            ast.parse(textwrap.dedent(new_body_text))
        except SyntaxError as exc:
            errors.append(f"replacement for {func_name!r} in {path} is not valid Python: {exc}")
            continue
        start, end = _function_span(original, node)
        edits[path] = original[:start] + new_body_text + original[end:]

    patch_parts = []
    for path, new_content in edits.items():
        old_content = by_source[path].content
        if new_content == old_content:
            continue
        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if not diff_lines:
            continue
        patch_parts.append(f"diff --git a/{path} b/{path}\n" + "".join(diff_lines))
    return "\n".join(patch_parts), errors


# --------------------------------------------------------------------------
# DeepSeek call -- hard wall-clock timeout + retries.
#
# Two distinct real failure modes were found running this against
# incoherent context (the `random` policy's shuffled multi-file dump):
#   1. Genuine network stalls (a direct curl to the same endpoint moments
#      later returned in ~1s) -- `urlopen`'s own `timeout` only resets on
#      each socket read, so a server trickling bytes without ever going
#      fully idle can outlast it. The ThreadPoolExecutor wall-clock
#      timeout below enforces a real hard cap regardless (the stuck
#      thread is abandoned, not joined, on timeout).
#   2. Reasoning-budget exhaustion, confirmed directly: incoherent context
#      can drive this reasoning model to spend its *entire* max_tokens on
#      reasoning_content and emit zero actual answer (finish_reason=
#      "length", content=""). This is not a hang -- the HTTP call
#      completes normally in ~200s -- it's the model failing to converge
#      within budget. A larger max_tokens (24000) gives it enough room to
#      reason *and* still answer; the timeouts below are sized to match.
# --------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"


def _fetch(req) -> dict:
    with urllib.request.urlopen(req, timeout=280) as resp:
        return json.loads(resp.read())


def call_deepseek(system: str, user_msg: str, attempts: int = 4) -> tuple[str, dict]:
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 24000,
        "temperature": 0.2,
    }).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            DEEPSEEK_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            },
        )
        t0 = time.time()
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            data = ex.submit(_fetch, req).result(timeout=300)
        except Exception as exc:  # noqa: BLE001 -- retry on anything, log what it was
            ex.shutdown(wait=False)
            last_exc = exc
            print(f"    deepseek call attempt {attempt}/{attempts} failed after {time.time() - t0:.1f}s: {exc!r}")
            continue
        ex.shutdown(wait=False)
        elapsed = time.time() - t0
        msg = data["choices"][0]["message"]
        raw = msg.get("content") or ""
        if not raw.strip() and data["choices"][0].get("finish_reason") == "length":
            last_exc = RuntimeError("empty content, finish_reason=length (ran out of budget while reasoning)")
            print(f"    deepseek call attempt {attempt}/{attempts}: {last_exc}")
            continue
        u = data.get("usage", {})
        usage = {
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"),
            "elapsed_s": round(elapsed, 1),
            "attempts": attempt,
        }
        return raw, usage
    raise RuntimeError(f"deepseek call failed after {attempts} attempts: {last_exc!r}")


def generate_patch(context: str, budget: int, by_source: dict) -> tuple[str, dict, list[str]]:
    user_msg = f"{context}\n\n---\nProduce the REWRITE_FUNCTION blocks now for budget={budget} tokens of context above."
    raw, usage = call_deepseek(FUNCTION_REWRITE_SYSTEM_PROMPT, user_msg)
    blocks = parse_function_rewrites(raw)
    patch, errors = function_rewrites_to_patch(blocks, by_source)
    usage["n_blocks"] = len(blocks)
    return patch, usage, errors


def check_patch_applies(patch: str, repo_dir: Path) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    diff_path = CACHE_DIR / "_check.diff"
    diff_path.write_text(patch)
    result = subprocess.run(
        ["git", "apply", "--check", str(diff_path)],
        cwd=repo_dir, capture_output=True, text=True,
    )
    return result.returncode == 0, result.stderr.strip()


# --------------------------------------------------------------------------
# Real evaluation: run a (task, policy, budget, repeat) condition end to
# end, then verify with the real swebench Docker harness.
# --------------------------------------------------------------------------

def run_one(row: dict, store: ContextStore, repo_dir: Path, policy: str, budget: int, seed: int) -> dict:
    by_source = {i.source: i for i in store.list(limit=10000)}
    ctx = CONTEXT_POLICIES[policy](row, store, budget, seed)
    ctx_tokens = COUNTER.count(ctx)
    try:
        patch, usage, errors = generate_patch(ctx, budget, by_source)
    except Exception as exc:  # noqa: BLE001 -- one instance's API failure shouldn't kill the sweep
        return {
            "instance_id": row["instance_id"], "policy": policy, "budget": budget, "seed": seed,
            "context_tokens": ctx_tokens, "applies": False, "error": f"generation failed: {exc!r}",
            "patch": "", "usage": {},
        }
    ok, apply_err = check_patch_applies(patch, repo_dir)
    return {
        "instance_id": row["instance_id"], "policy": policy, "budget": budget, "seed": seed,
        "context_tokens": ctx_tokens, "applies": ok, "apply_err": "" if ok else apply_err,
        "n_errors": len(errors), "patch": patch if ok else "", "usage": usage,
    }


def run_harness(predictions: list[dict], tag: str) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RUNS_DIR / f"preds_{tag}.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified", "--split", "test",
        "--predictions_path", str(pred_path),
        "--run_id", f"real_eval_{tag}", "--max_workers", "2",
        "--report_dir", str(RUNS_DIR),
    ]
    proc = subprocess.run(cmd, cwd=RUNS_DIR, capture_output=True, text=True, timeout=900)
    report_path = RUNS_DIR / f"{tag}.real_eval_{tag}.json"
    if not report_path.exists():
        return {"error": "no report written", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    report = json.loads(report_path.read_text())
    return {k: report[k] for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids")}


# --------------------------------------------------------------------------
# CLI driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True, help="SWE-bench Verified instance_ids")
    ap.add_argument("--policies", nargs="+", default=["oracle"], choices=list(CONTEXT_POLICIES))
    ap.add_argument("--budgets", nargs="+", type=int, default=[8000])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tag", required=True, help="label for this run's predictions/report files")
    args = ap.parse_args()

    for d in (REPOS_DIR, DBS_DIR, RUNS_DIR, RESULTS_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)

    rows = select_tasks(load_verified_dataset(), args.instances)
    stores, repo_dirs = {}, {}
    for row in rows:
        iid = row["instance_id"]
        print(f"preparing {iid} ({row['repo']}) ...")
        repo_dirs[iid] = fetch_repo(row["repo"], row["base_commit"], iid)
        stores[iid] = ingest_repo(iid, repo_dirs[iid])

    all_results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    for policy in args.policies:
        for budget in args.budgets:
            for seed in range(args.repeats):
                cond_tag = f"{args.tag}_{policy}_{budget}_r{seed}"
                print(f"--- {cond_tag} ---")
                records = []
                for row in rows:
                    iid = row["instance_id"]
                    rec = run_one(row, stores[iid], repo_dirs[iid], policy, budget, seed)
                    print(f"  {iid:30s} ctx={rec['context_tokens']:5d}tok applies={rec['applies']}")
                    records.append(rec)
                predictions = [
                    {"instance_id": r["instance_id"], "model_patch": r["patch"], "model_name_or_path": cond_tag}
                    for r in records
                ]
                harness = run_harness(predictions, cond_tag)
                resolved = set(harness.get("resolved_ids", []))
                for r in records:
                    r["resolved"] = r["instance_id"] in resolved
                print(f"  resolved={sorted(resolved)}")
                all_results.append({
                    "tag": cond_tag, "policy": policy, "budget": budget, "seed": seed,
                    "records": records, "harness": harness,
                })
                RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
