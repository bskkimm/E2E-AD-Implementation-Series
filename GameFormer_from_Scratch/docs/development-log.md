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
- Source audit found planning inference must choose each agent's mode separately
  (`open_loop_test_utils.select_future`), whereas joint inference sums scores.
  Added a discriminating two-agent fixture (RED -> GREEN), threaded task through
  diagnostics/figures, and rebuilt the notebook. Joint behavior is preserved.
  Full suite: 24 passed; Mypy: five stable source files passed.
- Full local nine-stage gate passed on 2026-11-13: pre-commit, Ruff/notebook
  formatting plus actionlint, Mypy, scientific unit coverage (>=90%), integration,
  standalone notebook/parity, clean wheel/CLI, dependency audit and docs/config.
  Current suite contains 25 tests. Dependency audit queried all 170 selected
  dependencies with zero known advisories; +cpu uses the public Torch release
  for advisory lookup, and unqueried packages now fail the gate (RED -> GREEN).
- Tasks 1–3 complete. Source measurements are pinned to c80cd7e plus aggregate
  source/config hashes. Task 4 local implementation/checks complete; fresh review,
  signed delivery commit, PR, hosted checks and external-review evidence pending.
- Final review: fresh `/root/whole_branch_review` (gpt-6-astra xhigh), base
  f3d5512 through fd62eff. Five Important findings accepted; no Critical/Minor.
  [Findings and dispositions](whole-branch-review.md) record the concrete triggers.
- Final: fixed persistent-worker recovery —
  `test_epoch_boundary_resume_matches_uninterrupted` RED→GREEN, suite 32/32.
- Final: fixed changed-cache resume and retained provenance —
  `test_resume_rejects_changed_cache_manifest_before_overwriting_provenance`
  RED→GREEN for both splits, suite 32/32; local/MLflow sessions verified.
- Final: fixed absent lane boundary normalization —
  `test_missing_lane_boundaries_remain_zero_after_world_translation` RED→GREEN,
  suite 32/32; translated/rotated absence and type-0 presence covered.
- Final: fixed tuner probe memory —
  `test_tuner_correctness_probe_respects_microbatch_memory_bound` RED→GREEN,
  suite 32/32; actual forwards remain microbatch-sized.
- Final: fixed host-dependent figure rendering —
  `test_figure_rendering_does_not_depend_on_host_fontconfig` RED→GREEN,
  suite 32/32. All 18 Pillow/64-color PNGs individually reinspected.
- Final: Ruling: keep real SDK/data/GPU/paper scores as unmet gates — unavailable
  assets/budget; synthetic evidence is bounded — real behavior/accuracy may differ.
- Final: Ruling: retain bounded in-memory metric adapter — full-dataset streaming
  is outside current scope — unbounded evaluation can exhaust host RAM.
- Final: Ruling: restrict accumulation equivalence to uniform validity/zero
  dropout — sample-weighted microbatch objectives are the defined behavior —
  changing batch partition may change mixed-mask/dropout gradients.
- Final: Ruling: exclude nuPlan, marginal EM and submissions — selected WOMD
  scope — those workflows and their performance claims remain unavailable.
- Final: Ruling: keep trusted pra/protection gates separate — auth/admin access
  is missing — review/enforcement remain unestablished and the PR stays draft.
- Final: Ruling: reject checkpoint schema 1 — sampler semantics changed and data
  hashes were absent — pre-fix diagnostic runs need a fresh output/run.
- Draft PR #1 opened and signed branch published. First hosted run 43404779502
  passed ten matrix jobs; host Cairo/font regeneration failed notebook/aggregate.
  The portable renderer fix requires a successful final hosted rerun.
- Deferred minors: none. No re-review agent; fix regressions/full suite are the
  plan's verification contract, distinct from the unmet trusted-review gate.
