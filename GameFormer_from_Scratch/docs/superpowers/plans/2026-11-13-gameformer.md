# GameFormer Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with a fresh
> whole-branch reviewer at the end. Continue without another approval round per
> the user's request to keep working.

**Goal:** A tested plain-PyTorch GameFormer with reproducible training and a
standalone illustrated notebook.

**Architecture:** Shared scene encoder and level-k decoder for WOMD joint
prediction and open-loop planning. SDK boundaries, training and teaching stay
separate from model computation.

**Tech stack:** Python >=3.11, PyTorch, NumPy, optional MLflow/Waymo SDK,
Matplotlib, pytest, Ruff, Mypy, nbformat/nbclient and uv lock.

**Spec:** `docs/superpowers/specs/2026-11-13-gameformer-design.md`.

## Global constraints

- Pin playbook `707c64b138d5e6c534af3d197fd4f278aab03d45` and official source
  `fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128`.
- Plain local PyTorch; explicit masks; no full dataset training without budget.
- Notebook local definitions, empty outputs, offline attachments and parity.
- Do not merge, force-push, weaken checks or fabricate review/reproduction.

## Review focus

- Valid positions at the origin survive transforms, pooling and losses.
- Entirely missing labels produce zero finite gradients and no bogus metrics.
- Partial accumulation groups and resumed schedules retain sample weighting.
- Joint modes/scores and official 2 Hz metric layouts keep agents aligned.
- Optional SDK/tracking failures are explicit and never substitute synthetic data.

### Task 1: Data contracts, model and objectives

Files: `gameformer/{config,geometry,data,model,losses,evaluation}.py`, configs,
`tests/test_science.py`, `pyproject.toml`, lock and setup documentation.
Interfaces: `GameFormer(ModelConfig)(dict[str,Tensor])` returns stacked
trajectories/scores; `objective(outputs,batch,task)` returns component tensors;
`synthetic_sample(config,index,seed)` and `SceneDataset(path,config)` share schema.

- [x] Write analytical tests for coordinate round trips/valid origin, segment
  masks, joint mode/NLL, empty supervision, forward and finite gradients.
- [x] Run tests first and confirm the missing implementation failure.
- [x] Implement the local data/model/loss interfaces and named JSON recipes.
- [x] Run science tests, including tiny-batch learning; record evidence.
- [x] Commit the independently testable scientific foundation.

### Task 2: Training, tracking and performance

Files: `gameformer/{engine,tracking,performance,cli}.py`, integration/CLI tests.
Consumes Task 1 model/batch/loss; produces train/eval/benchmark/tune CLI and
epoch-boundary `latest.pt` with complete resume state.

- [x] Write failing resume, partial-accumulation, ETA, MLflow and benchmark tests.
- [x] Implement one shared training step, resume, diagnostics and CLI paths.
- [x] Verify resumed vs uninterrupted weights, MLflow on/off and missing assets.
- [x] Measure bounded CPU smoke performance, write machine-readable evidence.
- [x] Commit training and measurement tooling.

### Task 3: Illustrated standalone notebook

Files: `gameformer/teaching.py`, notebook, editable assets and notebook tests.
Consumes Task 1 definitions and Task 2 infrastructure; no production model import
is allowed as a substitute for notebook architecture definitions.

- [x] Publish exact figure and definition inventories; write failing notebook
  freshness/attachments/parity checks.
- [x] Generate worked SVG/PNG figures and definition cells; inspect every image.
- [x] Execute fresh offline CPU notebook; compare model/loss/gradient parity.
- [x] Verify deterministic regeneration, numerical examples and empty outputs.
- [x] Commit the teaching resource.

### Task 4: CI, documentation and delivery

Files: workflow, pre-commit, README, COMMAND_GUIDE, guides/results manifest/tests.
Consumes all prior interfaces.

- [x] Add independent lint/type/unit/integration/notebook/package/audit/docs jobs
  plus an always-running gate, SHA-pinned actions and CPU Python matrix.
- [x] Run full local gate and clean environment package/CLI tests.
- [ ] Obtain a fresh scientific/engineering code review and fix valid findings.
- [ ] Commit, push topic branch, open PR and check CI/reviewer freshness when
  authorized access allows; report any unmet external gate exactly.
