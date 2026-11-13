# Evidence and limits

**No GameFormer paper result has been reproduced.** The single source of recorded
measurements is [results-manifest.json](results-manifest.json), which identifies
code/config/source/checkpoint hashes, seeds, synthetic splits, metric protocol,
environment and artifact paths. Local checkpoints stay under ignored `outputs/`.
An evidence record can name an earlier code commit than the final documentation
commit; CI also checks its aggregate source hash against current model/config files.

Tested evidence:

- Analytical geometry, explicit validity (including true zero coordinates), map
  pooling and attention padding, joint mode selection, Gaussian likelihood and
  directional planning potential. The practical Scenario boundary has worked
  rotated lanes/boundaries/signals and missing-history tests using SDK-shaped
  records; these do not substitute for a real SDK/data run.
- Full local model forward and finite backward for joint and planning fixtures.
  Planning decoding uses each agent's score argmax; joint decoding shares one
  summed-score mode. A real 40-update fixed-batch overfit substantially lowers
  the actual objective: [learning record](evidence/tiny-learning.json).
- Exact epoch-boundary interrupted/uninterrupted weights, scheduler and update
  counts, including a final partial accumulation group. Uniform-validity,
  zero-dropout accumulation gradients match concatenation. For mixed validity,
  the loss is a ratio over valid labels; changing batch composition need not be
  equivalent. Dropout also prevents a universal equality claim.
- Actual SQLite MLflow metrics, progress/ETA, figure artifacts and same-run resume
  with permitted runtime changes; disabled tracking works independently. One
  upstream SQLAlchemy `noload` deprecation warning remains visible.
- Standalone CPU notebook execution in a fresh Python process, all 24 architecture
  definitions, matching package outputs/loss/gradients, empty committed outputs,
  deterministic offline attachments, and [individual visual inspection](visual-inspection.md).
- Clean wheel installation and installed train/eval execution from a temporary
  directory; named JSON config consistency; static lint and stable-interface types.

The [official-source fixture](upstream-parity.json) maps 114 parameter/buffer
tensors at pinned source `fcb0d4a`, D256/E1/K1/M2 in evaluation mode. Maximum absolute
errors were approximately 7.2e-7 for trajectories, 3.6e-7 for scores and 4.2e-5
for parameter gradients. It compares the actual local and released objectives on
a fully valid fixture; it does not establish trained-weight, data-pipeline or
full-recipe equivalence.

[The CPU smoke run](evidence/cpu-smoke.json) trains for two tiny synthetic epochs,
then evaluates inference-selected trajectories on a different seeded synthetic
validation set. ADE is average distance per valid timestep in meters; FDE is
distance at each agent's last valid label. These are deliberately separate from
official Waymo minADE/minFDE/miss/overlap/mAP. Small synthetic runs are execution
evidence; their displacement errors have no real-world accuracy interpretation.

[Benchmark](benchmark.md) evidence covers loader, cached-step and end-to-end
training scopes. No full training, dataset download, paid compute or working GPU
was used. The first restricted worker trial was stopped after local IPC failed;
final worker measurements used local IPC access and a configurable timeout.

Unmet scientific gates are real cache/source/split validation, actual official SDK
execution, CUDA numerical/memory tests, budgeted full training, and results on the
official split/test protocol. nuPlan simulation/refinement, marginal EM ensembling
and submission packaging are outside the chosen scope. [PR/review/enforcement
status](ci-review.md) is an independent delivery gate.
