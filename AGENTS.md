# Agent Notes for Lensemble

This repository is an active research and education codebase for federated
JEPA-style world models. Treat docs, evidence files, model-card text, and issue
comments as public claim surfaces, not scratch notes.

## Working Loop

- Prefer `uv run ...` for local commands. The CI-shaped gates are configured in
  `pyproject.toml`; use the narrowest relevant gate first, then broaden when a
  change touches shared behavior.
- Use `rg`/`rg --files` for repo inspection. Keep edits scoped to the module,
  roadmap, or artifact contract implied by the task.
- Do not stage `thoughts/`, `.agents/`, or `.claude/` unless the user explicitly
  asks. They are local notes/cache surfaces, not release artifacts.
- Do not rewrite generated evidence artifacts by hand. Regenerate them through
  their producer scripts or update the human-facing docs that summarize them.

## Claim Discipline

- The SO-100 MVP is a gauge-only result. It demonstrates that anchored
  federation controls the latent frame where naive FedAvg fails; it does not
  prove downstream robotics usefulness.
- The dynamic-env #273 result is an educational systems demo, not a binding
  benchmark win. The current scratch dynamic-env federated run reaches
  `state_probe_r2=0.8885337114` and beats random and naive-FedAvg, but
  local-only reaches `0.8838405609`, so the federated margin is only
  `0.0046931505`, below the required `0.05`.
- It is fair to present the project as a Tapestry-like federated-training
  paradigm applied to JEPA world models: sovereign participants, shared protocol,
  secure aggregation/DP plumbing, artifact contracts, and browser inference
  demo. Do not present the current dynamic-env run as evidence that federation
  materially outperforms local-only.
- Keep non-claims explicit: no cryptographic honest-computation proof, no
  paper-scale LeWorldModel performance claim, no closed-loop physical SO-100
  success claim, and no browser training claim.

## Main Surfaces

- `README.md` and `SPEC.md` are the top-level narrative and corpus index.
- `docs/rfcs/` contains decision records; RFC-0017 defines the dynamic-env
  ground-truth metric hierarchy.
- `docs/roadmap/` tracks implementation/evidence state. Keep roadmap status in
  sync with live benchmark outcomes.
- `lensemble/` contains the Python package; `tests/` contains CPU-oriented
  contract tests.
- `deploy/hfjobs/` contains launchers and HF Jobs run documentation.
- `web/dynamic-env-demo/` is an ONNX inference and JS/Canvas environment demo
  only.

## Useful Validation Commands

```bash
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
uv run pytest tests/ml/test_dynamic_env_browser_demo.py tests/ml/test_dynamic_env_evidence_bundle.py
git diff --check
```
