# Execution ledger

Plan: `docs/superpowers/plans/2026-11-13-gameformer.md`.

- 2026-11-13: read paper/supplement and pinned source; read reviewed playbook
  `707c64b`. No silent fallback to the old provisional branch.
- User asked to keep working after joint-first scope was proposed. Continue
  without another approval round; no full training is authorized.
- Workspace decision: use assigned empty directory on a new topic branch of
  the containing series checkout. Do not initialize a competing nested repository
  or touch the other model projects. Sandbox writes remain local except necessary
  git metadata/CI installation, which require automatic approval review.
- Preflight interfaces: model outputs and both objectives use stacked
  `[K+1,B,P,M,T,...]`; synthetic/cache and trainer share explicit masks.
- Task 1 RED: science suite failed on the absent `gameformer` package. GREEN:
  seven analytical/model tests passed. New data/evaluation tests passed too.
- Task 2 RED: integration suite failed on absent `gameformer.engine`. GREEN:
  all 12 current tests passed in 6.06 seconds, including exact epoch-boundary
  resume, partial final accumulation and a real local SQLite MLflow run.
- Runtime installed locally: Python 3.11.14, torch 2.15.0+cpu, NumPy 2.4.6,
  pytest 9.2.0. Exact transitive dependencies are in `uv.lock`.
- External warning: MLflow's SQLAlchemy store currently uses deprecated `noload`;
  the tracking test passes. Do not hide the upstream warning globally.
- Tasks 2–4 remain in progress: CLI/data workflows, benchmarks/tuner, notebook,
  CI, documentation and current-head external review.
- Continued execution: CLI/prepare/benchmark/tuner and the standalone illustrated
  notebook now exist. Source parity at D256/E1/K1/M2 mapped 114 tensors and
  matched outputs, objective and parameter gradients; this is operator evidence.
- 18 final palette PNGs were individually inspected (all labels and arithmetic,
  masks/gradient barriers and arrow semantics); the notebook is about 601 kB.
  Assets deterministically regenerate and 24 local definitions pass AST/parity.
- MLflow runtime-config resume regression: observed an immutable-parameter error
  when permitted prefetch changed; fixed by preserving the initial parameter
  and logging the current runtime configuration in a tag. RED -> GREEN.
- Worker measurement stalled under the sandbox's local IPC restriction. Stopped
  only this task's process. Added a positive configurable worker timeout and a
  real slow-worker regression (RED -> GREEN). Worker tuning requires local IPC
  permission; remaining trials continue after timeout rather than hanging.
- Added analytical practical-map geometry/signal fixtures, full planning backward,
  exact partial-group accumulation and actual 40-update tiny-batch learning.
- Ruling: CPU-only smoke execution is the authorized compute budget; use signed
  topic commits and a PR, with no dataset download, GPU use or merge. Missing
  real/GPU evidence limits scientific claims, not ongoing local implementation.
- CI foundation: independent stages, scientific-core coverage floor 90%, clean
  wheel installation and installed train/eval pass. Root workflow is installed;
  full docs/provenance gate and current-head review remain pending.
- Access: SSH authenticates as bskkimm; gh's stored token is invalid. The GitHub
  connector reads repository/workflow metadata but branch protection returns 403;
  rulesets read as empty. Existing pra policy permits live review of own repos,
  but gh cannot dispatch its workflow. No policy has been weakened or bypassed.
