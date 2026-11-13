# GameFormer design

User direction: reimplement GameFormer using the shared playbook, then “keep
working on”. Continue autonomously with the previously recommended joint-first
scope; do not add another design-approval stop. Technical decisions below are
implementation choices, not fabricated user selections.

## Intent and approaches

Build readable, local PyTorch code and a full teaching notebook, with evidence
for scientific operations and reproducible experiments. We considered the WOMD
joint predictor, WOMD planning model and nuPlan planner. Start with the released
joint predictor because the supplied repository implements it directly; include
its shared-model open-loop planning variant. nuPlan requires a separate planner,
route search, refinement and simulation and is outside this initial delivery.

Use the pinned sources and deviation table in `docs/paper-to-code.md`. Preserve
release-specific attention, future-feature scaling, detach and optional legacy
lane/heading semantics. Fix validity with explicit masks and declare the effect.
Named recipes distinguish release, paper-hyperparameter reconstruction, corrected
practical behavior and reduced synthetic smoke. All parameters start randomly.

## Interfaces and components

`ModelConfig` defines actors, predicted agents, history/future lengths, map sizes,
width, heads, encoder depth and interaction levels. `RunConfig` includes the model,
task, seed, microbatch, accumulation, epochs, optimizer/schedule, device/precision,
data directories, optional tracking URI and timezone. JSON resolves every field.

`GameFormer(config)(batch)` returns `trajectories` [K+1,B,P,M,T,4] and `scores`
[K+1,B,P,M]. The batch is a dictionary of tensors: ego_state [B,H,9],
neighbors_state [B,C-1,H,9], actor_valid [B,C,H], map_lanes [B,P,L,Q,17],
lane_valid [B,P,L,Q], map_crosswalks [B,P,W,R,3], crosswalk_valid [B,P,W,R],
future [B,P,T,2], future_valid [B,P,T]. C includes the P predicted actors first.
Explicit validity is mandatory. Missing future labels are permitted in inference.

Use two LSTMs, categorical embeddings, lane/crosswalk MLPs, sinusoidal lane
positions, segment max pooling, shared scene attention, level-0 queries, shared
future MLP and independent level-k attention/prediction blocks. Losses supervise
every stage. Joint loss selects one shared mode; planning has the additional
cross-level reciprocal-distance potential field with a 3 m cutoff and weight .1.

Keep processed-cache IO and optional official SDK interfaces outside the model.
Cache records preserve scenario IDs, split, source identity and explicit masks;
no pickle loading. Synthetic data is deterministic per index. Legacy upstream
caches lacking validity fail clearly rather than silently guess missing labels.
Geometry exposes tested world/local transforms. Official evaluation is opt-in.

## Training, evidence and performance

Plain AdamW; effective batch is microbatch times accumulation times workers.
Support one process initially and label that adaptation. Normalize partial final
accumulation groups by their actual sample counts. Save atomic epoch-boundary
checkpoints: model/optimizer/scheduler/scaler, resolved config, epoch/global step,
RNG and sampler state, active/session time and tracking run ID. Resume rejects
recipe changes; CPU uninterrupted/resumed weights must agree exactly.

MLflow is optional. With it enabled log source/config/environment/data identity,
loss components, LR, gradient norms, validation diagnostics and representative
trajectories. Progress summaries include total percent, active and wall time,
epoch duration, remaining duration and finish timestamp in Asia/Tokyo. Rate
estimates use new-session measurements including observed evaluation/save costs;
early estimates are provisional. Logging errors are visible.

Benchmarks share construction and loss/update code. Measure loader-only, cached
batch training and end-to-end paths separately with warmup, CUDA synchronization,
memory, loader wait, startup and software/hardware provenance. A bounded opt-in
tuner changes runtime settings only, uses isolated states and deterministic
fixtures, checks output/loss/gradient/short-learning parity and records all trials.
Unproven improvements fall back to FP32 baseline; no full training is launched.

## Notebook and delivery

Local definitions in ordered cells: intuition, equation, shapes, implementation,
probe, then full loss/backward/update, decoding and opt-in infrastructure. Generate
definitions from an explicit architecture inventory; compare matching weights,
inputs, outputs and gradients to the package. No stored outputs/network default.
Publish a model-specific figure inventory before drawing, use editable SVGs and
compact PNG attachments, verify hashes/freshness/arithmetic and inspect each.

Deliver setup/lock, README/commands, data/evaluation guides, named configs, source
map, results manifest, benchmark report, tests and CI. CI includes pre-commit,
Ruff, stable-interface types, unit/integration/notebook, clean package install,
dependency audit and docs/config checks with an always-running aggregate gate.
Use topic branches, signed commits and reviewable PRs. Verify required checks
and current-head scientific/engineering review; report unavailable enforcement,
reviewer, data or GPU access as unmet gates. Never merge without authorization.
