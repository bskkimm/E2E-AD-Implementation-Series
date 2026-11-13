# GameFormer reimplementation

Read the private `bskkimm/paper-reimplementation-playbook` at reviewed commit
`707c64b138d5e6c534af3d197fd4f278aab03d45`: `standards/paper-reimplementation.md`,
`standards/pr-and-ci.md`, `prompts/implement-paper.md`, and
`standards/notebook-visualizations.md`. These were read from the persistent clone
and the revision checked against authenticated GitHub on 2026-11-13. PRs #1 and #2
are merged; this is not the provisional bootstrap standard.

Read `docs/paper-brief.md` and `docs/paper-to-code.md` before scientific changes.
Implement architecture, losses, training and benchmarks locally in plain PyTorch.
Keep masks, units, frames, gradient barriers and recipe differences explicit.
The notebook defines the complete architecture locally and runs offline on CPU.
Keep its outputs empty and verify definitions, outputs and gradients against the
package. Illustrations need editable sources, offline attachments and inspection.

Use topic branches and reviewable PRs; preserve commit signing. Never merge,
force-push, weaken checks, auto-approve or automatically resolve review threads.
Use the existing pra integration if available; report access/provider-policy gaps.
Synthetic tests are execution evidence, not reproduced paper results. Do not
launch full training or download a dataset without an explicit compute budget.
Keep datasets, checkpoints, environments and tracking stores out of Git.

Current implementation work is in the assigned directory on
`feat/gameformer-reimplementation` of the containing series repository.
