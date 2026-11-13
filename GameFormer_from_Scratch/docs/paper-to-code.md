# Source audit and paper-to-code map

The source revision and exact target are recorded in [the brief](paper-brief.md).
This is a reimplementation; measurements must say whether they use synthetic
scenes, a real subset or the official benchmark. No paper score has been reproduced.

## Computation

```text
agent histories -> two LSTMs ---------+
lane/crosswalk points -> MLP/pooling -+-> scene Transformer
                                           |
modality + agent queries -> level 0 futures + scores
                                           |
 previous xy -> detached kinematics -> shared future MLP -> time max
 previous scores -> probability pooling -> future self-attention
                                           |
 previous query + future features -> masked cross-attention -> level k
                                           |
                all-level loss -> backward -> AdamW
```

Positions are meters in the first target's current coordinate frame; +x is its
heading, +y is left. Time interval is .1 seconds. History rows are
`x,y,heading,vx,vy,length,width,height,type_id`; future Gaussian rows are
`mu_x,mu_y,log_sigma_x,log_sigma_y`. A mode index is shared by the predicted
agents in the joint model. The first target is not necessarily the autonomous car.

| Paper / reference | Local module | Verification |
| --- | --- | --- |
| §3.2, supplement A.1/A.2; data_process.py/data_utils.py | `gameformer/data.py`, `geometry.py` | round-trip frames, valid origin, cache contracts |
| §3.2; AgentEncoder/LaneEncoder/CrosswalkEncoder | `gameformer/model.py` | shapes, masks, finite gradients |
| §3.2 map max pooling and scene Transformer | `gameformer/model.py` | partial/empty segment fixtures |
| §3.3, Fig. 2/3; initial and interaction decoders | `gameformer/model.py` | all stages, shared future encoder, detach behavior |
| §3.4 Eqs. 2/3, supplement A.1; inter_pred_utils.py | `gameformer/losses.py` | analytical NLL, joint winner, empty targets |
| §3.4 Eqs. 4/5, supplement A.2; open_loop_train_utils.py | `gameformer/losses.py` | cross-level potential field, validity and gradients |
| §4.1/4.2 and supplement training recipes | `gameformer/config.py`, `engine.py` | named configs, update and resume |
| Official motion metric configuration | `gameformer/evaluation.py` | adapter layout; real SDK gate separate |
| Complete architecture teaching | `notebooks/gameformer_e2e_walkthrough.ipynb` | fresh CPU execution and output/gradient parity |

## Source differences and deliberate deviations

| Detail | Evidence | Treatment / likely impact |
| --- | --- | --- |
| Prediction depth | §4.2.1/Table 8: K6; released train.py default K3 | Keep released K3 and a separately named K6 paper-recipe reconstruction; do not mix results. |
| Background count | supplement A.1: up to 20; released preprocessing: 32 | Released recipe keeps 32; paper reconstruction keeps 20. Context and cost change. |
| Agent features | supplement: one-hot 11 channels; source: LSTM on 8 plus categorical embedding | Follow the released operator; exact unpublished paper parameter parity is unverified. |
| Prediction schedule/batch | supplement A.1 vs released train.py/train.sh | Preserve both descriptions. Single-device accumulation is a named practical adaptation. |
| Collision loss | supplement A.1 explicitly uses imitation only; A.2 uses .1 interaction | Prediction omits it intentionally; planning includes it for k >= 1. |
| NLL printed Eq. 3 repeats dx/sigma_x | PDF page 3907 inspected; source uses dx/sigma_x and dy/sigma_y | Use both coordinates, as diagonal Gaussian likelihood requires. |
| Future pooling | source `(features * softmax(scores)).mean(mode)` | Preserve the extra 1/M scale for release compatibility; normalized sum is a separate experiment. |
| Future gradient barrier | FutureEncoder detaches kinematic features | Preserve this barrier. Previous query content and scores retain gradients. |
| Future heading | source clamps dx positive before atan2 | Preserve in released compatibility; geometric atan2 is an explicit corrected option. |
| Future mask | source masks the agent's future token after self-attention | Preserve placement. This is not a proof of total information isolation through mixed tokens. |
| Lane feature offsets | preprocessing writes stop-point/interpolating/stop-sign at 14/15/16; encoder reads 14/15 as interpolating/stop-sign | Explicit compatibility option; corrected columns form a separately named practical experiment. |
| Validity | release infers absence from coordinates or their sum | New caches carry explicit official validity. Valid x=0/y=0 is retained; invalid values are excluded from loss reductions. This changes edge-case behavior deliberately. |
| Segment mask | source masks a pooled segment if any constituent point has x=0 | Compatibility requires all points valid; optional partial-segment policy is named. Empty pooled features are zero and stay masked. |
| Planning interaction directions | source ego responds to prior neighbors; neighbors respond to prior other neighbors | Implement that released objective explicitly; it differs from an all-ordered-pairs reading of Eq. 4. |
| Paper scores | paper Table 1 includes test-server results | Local diagnostics/validation are distinctly named. Missing SDK/data/checkpoint access cannot establish reproduction. |
| Raw preprocessing | `gameformer/prepare.py` uses nearest lane seeds/forward paths, resampling and nearby polygon vertices; boundaries use nearest polyline points | This is an explicitly practical preprocessing recipe, not released lane-selection/cache parity. Record `practical_vector_map_v1` in every produced manifest. History gaps remain zero with explicit masks instead of the release's velocity imputation. Do not claim full data-pipeline fidelity until validated on real scenarios. |

Recipe names are `released_joint`, `paper_joint_reconstruction`, `practical_joint`,
`released_planning`, and `smoke`. The reconstruction combines reported paper
hyperparameters with published operators; it is a hypothesis, not exact paper
code recovery. The smoke recipe reduces width/depth/context only and retains the
prediction horizon needed by the real objective.

The official release does not include the 64-mode marginal EM ensemble, WOMD
closed-loop simulator or prediction submission packaging. nuPlan uses the
separate [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner), with
route generation/refinement and a different compact model (supplement A.5).
Those capabilities must not be implied by a successful WOMD forward pass.
