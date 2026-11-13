# Fresh whole-branch review and dispositions

The fresh reviewer `/root/whole_branch_review` (gpt-6-astra, xhigh) reviewed
`f3d5512e08bf9228f761692818b59903dbc2f53d` through
`fd62eff51436b6ea8f0d15ed2e93edbb523d39d6`. The reviewer independently ran the
then-current 25 tests and official-source parity. Its verdict was **with fixes**:
five Important findings, zero Critical and zero Minor. This local review is
separate from the trusted external pra/current-head review gate.

All five findings were accepted at Important severity and handled in one fix pass.
Every regression below was observed failing against the prior implementation,
then passing after the fix. The complete CPU suite passed **32 tests**, with the
existing visible upstream MLflow/SQLAlchemy deprecation warning. Mypy passed its
five stable modules; notebook execution/parity and deterministic rendering passed.

| Finding / concrete trigger | Final behavior | Regression evidence |
| --- | --- | --- |
| Persistent workers consume a different startup seed after restarting, changing next-epoch order and actual weights | Explicit RandomSampler owns the saved generator; worker startup uses an independent generator | `test_epoch_boundary_resume_matches_uninterrupted`: workers 1 failed exact weight equality before the fix; workers 0 and 1 pass afterward |
| Changed cache contents at the same configured path are accepted on resume and original provenance is overwritten | Checkpoint schema 2 saves both manifest identities; comparison precedes restoration/artifact writes. Original config/environment remain intact; unique local and MLflow session snapshots preserve each restart | `test_resume_rejects_changed_cache_manifest_before_overwriting_provenance`: failed to raise before; both splits rejected afterward. `test_tracking_resume_records_permitted_runtime_changes` verifies retained originals and two local/MLflow sessions |
| Absent lane boundaries become fictitious coordinates after moving the scene | Boundary presence follows concatenation/resampling and zeros absent geometry after frame normalization; type 0 and true zero coordinates remain valid | `test_missing_lane_boundaries_remain_zero_after_world_translation`: failed by up to 2,000 m before; translated and rotated cases pass afterward; rotated present-boundary fixture uses type 0 |
| Tuner forwards the whole effective batch during its correctness probe | Probe forwards/backpropagates sample-weighted microbatches; only detached CPU outputs are concatenated | `test_tuner_correctness_probe_respects_microbatch_memory_bound`: observed batch 8 with configured microbatch 1 before; every forward is batch 1 afterward |
| Hosted Ubuntu and local Cairo/font rendering produce different asset bytes | Shared editable SVG/Pillow layout, wheel-bundled fonts, renderer versions and font hashes; no host Fontconfig/Cairo dependency | `test_figure_rendering_does_not_depend_on_host_fontconfig`: normal vs empty Fontconfig PNGs differed before and match afterward; deterministic asset and fresh-process parity checks pass locally |

The renderer fix was prompted by hosted run
[43404779502](https://github.com/bskkimm/E2E-AD-Implementation-Series/actions/runs/43404779502):
ten matrix jobs passed; notebook regeneration and the aggregate failed. A later
hosted run must establish the final portable-rendering result. All 18 final PNGs
were individually reinspected; see [visual evidence](visual-inspection.md).

Rulings on subjects the reviewer declined to judge, in review order:

1. Actual SDK/real-data fidelity, GPU precision/memory and paper scores remain
   unmet scientific gates because assets, hardware and training budget were not
   supplied. Synthetic/operator evidence stands within its stated scope. Cost
   if wrong: real-world behavior or reproduced accuracy may differ.
2. Full-dataset metric streaming remains outside the bounded in-memory subset
   adapter. Cost if wrong: an unbounded evaluation can exhaust host memory;
   the data guide explicitly limits it to bounded subsets.
3. Concatenation equivalence remains limited to uniform validity and zero dropout.
   The trainer's defined behavior is sample-weighted microbatch objectives. Cost
   if wrong: changing batch partition with mixed masks/dropout may change gradients;
   tuning must reject a candidate that fails its numerical comparison.
4. nuPlan refinement/simulation, marginal EM ensemble and submission packaging
   remain outside the selected WOMD joint/open-loop scope. Cost if wrong: this
   implementation does not provide those workflows or their performance claims.
5. Trusted pra review and server protection remain separate unmet access gates:
   invalid gh authentication and app administration scope cannot be replaced by
   this local review. Cost if wrong: current-head review and enforced merge
   checks are unestablished; the PR stays draft.

Compatibility ruling: schema 1 checkpoints are rejected because the old sampler
state and absent data hashes cannot support the revised recovery guarantee.
Cost: pre-fix diagnostic runs require a fresh output/run rather than exact resume.
No minors are deferred. No re-review agent was dispatched; the execution plan
requires regressions and a green suite for this fix pass.
