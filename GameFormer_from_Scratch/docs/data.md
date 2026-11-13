# Data contracts and optional SDK

The primary target is [Waymo Open Motion Dataset](https://waymo.com/open/), v1.1,
with `scenario/training` and `scenario/validation_interactive` for joint prediction.
Acquire it under its own access terms. No data was supplied or downloaded here.
The official source pin and split assumptions are recorded in [the brief](paper-brief.md).

Each cache contains `manifest.json` and checksummed NPZ records with no pickled
objects. Each manifest identifies schema version 1, split, source/scenario IDs,
target track IDs and preprocessing method. Train/validation must have disjoint
source/scenario identities, checked before loading. Keep a common `source_id`
for the same dataset release across both splits.

Let C be observed actors, P predicted actors, H history, T future steps, L lane
polylines, Q lane points, W crosswalks, R crosswalk points. One record has:

| Field | Shape | Meaning |
| --- | --- | --- |
| ego_state | H,9 | First predicted actor, not necessarily the autonomous car |
| neighbors_state | C-1,H,9 | Remaining targets first, then background context |
| actor_valid | C,H | Boolean observation validity |
| map_lanes / lane_valid | P,L,Q,17 / P,L,Q | Per-target nearby map; common focal frame |
| map_crosswalks / crosswalk_valid | P,W,R,3 / P,W,R | Polygon samples and explicit validity |
| future / future_state | P,T,2 / P,T,5 | xy and xy/heading/vx/vy labels |
| future_valid | P,T | Boolean label validity, independent of coordinates |

Histories use `x,y,heading,vx,vy,length,width,height,type_id`. Positions and sizes
are meters, angles radians, velocities m/s; sample period is .1 seconds. All
positions/velocities share the first target's current frame: +x forward, +y left.
Zero is a valid coordinate. Only masks designate missing entries. Missing targets
produce no supervised gradients and contribute no displacement diagnostics.

Released joint dimensions are C34/P2/H11/T80/L6/Q100/W4/R50. The JSON model
configuration defines the actual shape; smoke differs and cannot load a full-sized
cache. The first actor must be valid now. Values marked valid must be finite.

`prepare_waymo` lazily imports TensorFlow and the official Waymo Scenario protobuf.
It scans a bounded number of TFRecord scenarios and writes a bounded number of
pairs. Validation uses the two objects of interest; training uses each eligible
target's nearest other targets. Lane paths, boundaries and crosswalk selection
are a **practical adaptation**, named `practical_vector_map_v1`; they do not prove
released preprocessing parity. History gaps remain masked rather than imputed.
Missing lane boundaries retain zero geometry after translation and rotation:
internal presence flags follow path concatenation and resampling. Presence is
independent of zero coordinates and boundary type 0. Cache schema 1 is unchanged;
training checkpoint schema 2 additionally saves both manifest hashes and rejects
changed caches before resuming.
See [the deviation table](paper-to-code.md). Preparation rejects nonempty output
directories and currently supports the joint task only.

Install an official SDK version compatible with its TensorFlow/Python runtime in
an isolated environment, following the [official repository](https://github.com/waymo-research/waymo-open-dataset).
That combination was not available or verified here; no guessed SDK pin is
presented as tested. NumPy caches bridge such an SDK environment to the locked
model environment. Actual official evaluation requires the model and SDK in a
compatible environment and fails explicitly if the SDK is absent.

The official joint adapter produces predictions `[B,1,M,2,16,2]` with shared mode
scores `[B,1,M]` and real seven-channel history/future labels. It samples future
indices `4::5`, matching 2 Hz measurements at 3/5/8 seconds. It calls the actual
motion metric operation for minADE/minFDE/miss/overlap/mAP. Adapter layout tests
are not SDK execution evidence. Official evaluation currently collects the selected
validation inputs in memory; use an explicitly bounded subset until streaming
aggregation has been implemented and validated for full-scale runs.
