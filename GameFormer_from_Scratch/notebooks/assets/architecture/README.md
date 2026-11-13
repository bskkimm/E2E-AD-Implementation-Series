# GameFormer illustration inventory

Exactly **18 distinct instructional figures**, each with an editable SVG, compact
PNG and offline Markdown attachment. There is exactly one opening overview.
All values in mechanism figures are reduced illustrative examples, not learned
weights or measured benchmark results. Full released dimensions are D256, C34,
P2, M6, H11, T80; the default executable probes use the named D32 CPU smoke model.

| ID / notebook anchor | Placement and learning question | Source |
| --- | --- | --- |
| overview | Introduction, before setup: how does one scene become joint futures? | model.py |
| scene-tensors | Before sample preparation: how do local frames and explicit masks preserve zero? | geometry.py, data.py |
| agent-history | Before AgentEncoder: how do LSTM gates accumulate history? | model.py |
| lane-encoding | Before LaneEncoder: how do lane attributes and positions enter a token? | model.py |
| crosswalk-encoding | Before CrosswalkEncoder: what does a point MLP transform? | model.py |
| map-segments | Before segment_points: which points and channels survive max pooling? | model.py |
| scene-attention | Before AttentionBlock: which masked keys contribute to a context? | model.py |
| initial-query | Beside level-0 query construction: how do mode and agent queries differ? | model.py |
| gaussian-head | Before GaussianHead: how do local offsets become Gaussian means? | model.py |
| future-encoding | Before FutureEncoder: how do xy differences become detached kinematic features? | model.py |
| future-interaction | Before InteractionDecoder: how are modes pooled and future agents mixed? | model.py |
| reasoning-mask | Beside level-k mask probe: where is the own-future token hidden? | model.py |
| full-assembly | Before GameFormer: which dimensions and levels are stacked? | model.py |
| joint-objective | Before objective: which joint mode receives likelihood gradients? | losses.py |
| optimizer-update | Before real backward/update: how does a gradient change a parameter? | engine.py |
| decoding-metrics | Before inference/metrics: which mode is selected without targets? | evaluation.py |
| checkpoint-resume | Beside opt-in recovery: what state must survive an interruption? | engine.py |
| tracking-eta | Beside tracking: how is remaining time computed after resume? | tracking.py |

Scene preparation merges coordinate transforms, validity and targets into one
worked example. Lane attribute and position mechanisms are merged; future mode
pooling and self-attention share a figure. Repeated decoder levels use one worked
mechanism plus the full stack figure. No sensor/backbone or marginal-EM workflow
exists in this target, so those are not pictured.

Generate with `python -m gameformer.teaching build`. Check inventory, exact
placement, source freshness, attachment hashes, arithmetic, deterministic rendering
and package output/gradient parity with `python -m gameformer.teaching check`.
`manifest.json` records source hashes, render versions, captions and alt text.
All diagrams are authored programmatically; no generative image tool is used.
