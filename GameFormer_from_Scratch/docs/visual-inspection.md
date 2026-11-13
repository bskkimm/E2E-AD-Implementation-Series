# Individual figure inspection

On 2026-11-13 the authoring Codex agent opened **each of the 18 final PNGs**
individually at its 1100×620 resolution after 64-color compression. This inspection
was repeated for every PNG after replacing Cairo with Pillow and wheel-bundled
fonts; the notebook is 972,542 bytes. No contact
sheet was used as a substitute. Checked text/contrast, clipping, panel arrow
endpoints, full-versus-toy dimensions, arithmetic, masks and gradient barriers
against the local source. Labels and equations are legible; no overlap or clipped
text was observed. Colors repeat a labeled input/operation/result grammar.

| Figure ID | Scientific check observed |
| --- | --- |
| overview | History/map -> scene -> level-0 and repeated responses -> all-level supervision and inference |
| scene-tensors | World (10,6), origin (10,5), heading 90° -> local (1,0); validity independent of zero |
| agent-history | .8×1 + .5×.4 = 1; .5 tanh(1) ≈ .3808; type addition -> .4808 |
| lane-encoding | Geometry/discrete fields, point position at zero adds (0,1), yielding (.5,1.6) |
| crosswalk-encoding | Selected matrix rows transform (2,-1,0) into (2,1), then ReLU |
| map-segments | Per-channel valid maxima (7,4), excluding padded 99; strict/partial policies labeled |
| scene-attention | Masked softmax (.25,.75,0), read 3.5; self residual and released no-query-residual cross block distinguished |
| initial-query | History + mode + agent vectors -> (1.5,1.5); shared joint mode index |
| gaussian-head | Current (10,2) + offset (3,1) -> (13,3); exp(0,ln2) -> std (1,2) |
| future-encoding | 1 m/.1 s -> 10 m/s; kinematics detached, future MLP remains learnable |
| future-interaction | Weighted scalar features (.5,3); released mean -> 1.75; normalized sum differs |
| reasoning-mask | Own future hidden after future self-attention; mixed other tokens may contain own information |
| full-assembly | C34 plus 80 map tokens -> 114; four released stages with [4,B,2,6,80,4] outputs |
| joint-objective | Joint errors select one mode; dx2/std1 yields NLL2 and four-term regression8; planning potential labeled |
| optimizer-update | First AdamW step with zero decay/ignored epsilon: w2 -> 1.9; actual accumulation/clip flow labeled |
| decoding-metrics | Joint score argmax selects index1; distances1/2/3 yield ADE2 and FDE3 m; 2 Hz SDK sampling distinguished |
| checkpoint-resume | Atomic temporary-file replacement; complete optimizer/RNG state; epoch boundary only |
| tracking-eta | New-session rate2/20 -> .1 epoch/s; four remaining ->40 s; active40+20 ->60 s |

The inventory and final asset/source hashes are in
[manifest.json](../notebooks/assets/architecture/manifest.json). The automatic check
regenerates SVG/PNG bytes, validates one attachment/anchor per figure, compares
24 local architecture definitions, rejects shadowing/stored outputs, and compares
package/notebook behavior in a fresh CPU process. Automatic arithmetic fixtures
cover pooling, Gaussian loss and ETA; executable lesson probes cover additional
worked examples. Manual inspection is distinct from these automatic checks.

The PNGs are schematic worked examples with editable SVG text. The overview is
conceptual; other figures explicitly label toy values. None is a measured training
curve or paper-result figure. Source changes require asset checks and reinspection
of any changed pixels or scientific meaning.
