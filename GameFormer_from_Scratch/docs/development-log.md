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

