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
