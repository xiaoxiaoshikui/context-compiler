# Verification

Validated in the build environment on 2026-08-17:

- `python -m compileall` on the source package: PASS
- `python -m unittest discover -s tests -v`: 5/5 PASS
- editable install with local build tooling / no dependencies: PASS
- CLI repository ingestion: PASS
- CLI hard-budget compilation: PASS (`385 / 650` heuristic tokens on demo before final constraint-classification tweak)
- experiment demo: PASS; toy evaluator produced a visible budget frontier and `B95=180`
- HTTP `/health` and `/compile`: PASS
- MCP adapter: source/API shape checked against current official MCP Python SDK v2 docs; runtime dependency is not installed in the sandbox, so end-to-end MCP transport was not executed here

The bundled demo/evaluator is synthetic and is only a plumbing test. It is not evidence of model-quality gains.

## 2026-08-17 (session 2): baseline benchmark harness

- Initialized git; first commit is the MVP as delivered.
- Added `src/context_compiler/baselines.py` (`FullContextBaseline`,
  `RandomContextBaseline`) and `benchmarks/` (5 synthetic tasks, `run.py`
  budget-sweep driver, generated `REPORT.md`/`results/results.json`).
- `python -m unittest discover -s tests -v`: 7/7 PASS (5 previous + 2 new
  baseline tests).
- `python benchmarks/run.py`: PASS; produced a real (not just plumbing)
  finding — see `benchmarks/README.md`, "A real finding this harness
  already surfaced" — which led to changing
  `CompilerConfig.constraint_min_level` from `L1` to `L2` in
  `compiler.py`. Existing tests still pass after the change.
- As with the original demo evaluator, the benchmark's keyword-conjunction
  evaluator is a plumbing/proxy check, not evidence of real model-quality
  gains. See `benchmarks/README.md` for the full list of caveats.
