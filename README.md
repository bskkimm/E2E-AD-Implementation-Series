# E2E-AD-Implementation-Series

This is a hands-on reimplementation series for learning how end-to-end autonomous driving evolved, from direct pixel-to-control imitation to real-time diffusion planning. The models are ordered by the architectural ideas they introduce, and each will eventually live in its own sub-repository. Nothing has been implemented yet; every model is planned.

## Why this series exists

Each selected model marks a distinct architectural turning point rather than merely a strong benchmark result, so the series follows learning value instead of strict chronology or leaderboard rank.
Reimplementation makes the hidden details concrete: data pipelines, coordinate conventions, loss design, training instability, and evaluation.

## The evolution story

```text
Pixel-to-control imitation
   |
   v
Command-conditioned policy
   |
   v
Privileged teacher-student learning (control -> waypoints)
   |
   v
Multimodal transformer sensor fusion
   |
   v
Object-centric token planning
   |
   v
Unified query-based perception-prediction-planning
   |
   v
Vectorized end-to-end planning
   |
   v
Generative (diffusion) planning on structured inputs
   |
   v
Real-time sensor-based diffusion planning
```

The progression is not a claim that every later model supersedes every earlier one.
It is a sequence of increasingly rich answers to the same question: what should an end-to-end driving model observe, represent, and predict?

## The series

| # | Model (year) | One-line lesson | Paradigm shift | Status | Sub-repo |
|---:|---|---|---|---|---|
| 1 | [PilotNet (NVIDIA, 2016)](https://arxiv.org/abs/1604.07316) | A CNN can map raw pixels directly to steering; behavioral cloning reveals saliency and covariate shift. | End-to-end learning is viable at all. | 🔜 Planned | (coming soon) |
| 2 | [CILRS (2019)](https://arxiv.org/abs/1904.08980) | Command-conditioned branched policies resolve intersection ambiguity; a speed head exposes causal confusion and the inertia problem. | Direct control -> navigation-conditioned control. | 🔜 Planned | (coming soon) |
| 3 | [Learning by Cheating (LBC) (2019/2020)](https://arxiv.org/abs/1912.12294) | Split "learning to see" from "learning to act" with a privileged BEV teacher, a camera student, and waypoint-plus-PID control; include a ChauffeurNet-style trajectory-perturbation ablation. | Expert imitation -> privileged teacher-student learning. | 🔜 Planned | (coming soon) |
| 4 | [TransFuser](https://arxiv.org/abs/2104.09224) + [TCP extension](https://arxiv.org/abs/2206.08129) (2021/2022) | Attention fuses camera and LiDAR BEV features across scales; TCP adds dual trajectory and control branches with situation-based fusion. | Single-sensor learning -> multimodal transformer fusion. | 🔜 Planned | (coming soon) |
| 5 | [PlanT (2022)](https://arxiv.org/abs/2210.14222) | Represent the scene as compact object-level tokens and plan with a transformer; attention weights provide explainability. | Dense pixel/BEV reasoning -> object-centric planning. | 🔜 Planned | (coming soon) |
| 6 | [UniAD (2023, CVPR Best Paper)](https://arxiv.org/abs/2212.10156) | Unify tracking, mapping, motion, occupancy, and planning as query-based tasks in one differentiable, planning-oriented network; first complete mini VectorNet and mini Lift-Splat warm-ups. | Independent task heads -> unified perception-prediction-planning. | 🔜 Planned | (coming soon) |
| 7 | [VAD](https://arxiv.org/abs/2303.12077) + [VADv2 ablation](https://arxiv.org/abs/2402.13243) (2023/2024) | Replace dense rasterized BEV with vectorized agent/map instances as planning constraints; test a probabilistic trajectory-vocabulary head in the VADv2 ablation. | Dense BEV reasoning -> vectorized planning. | 🔜 Planned | (coming soon) |
| 8 | [Diffusion Planner (2025, ICLR Oral)](https://arxiv.org/abs/2501.15564) | A DiT-style model over structured inputs jointly generates ego and neighbor trajectories with flexible classifier guidance. | Deterministic regression -> generative multimodal planning. | 🔜 Planned | (coming soon) |
| 9 | [DiffusionDrive (2024/2025, CVPR Highlight)](https://arxiv.org/abs/2411.15139) | Truncated, anchor-based diffusion makes generative planning real-time inside a sensor-based end-to-end stack. | Structured generative planning -> real-time sensor-based generative planning. | 🔜 Planned | (coming soon) |

All entries are plans, not completed reproductions.
The eventual sub-repositories will hold implementation notes, code, experiment configuration, and results; this repository will remain a compact index.

## Model notes

### 1. PilotNet (NVIDIA, 2016)

[PilotNet](https://arxiv.org/abs/1604.07316) is the minimal starting point: a convolutional network maps a front-facing image directly to a steering command.
Its key lesson is that behavioral cloning can learn useful road features without explicit perception labels, while also exposing covariate shift when the learned policy leaves the training distribution.
I plan a clean reimplementation of the network, training objective, augmentations, and saliency analysis using an appropriately scoped driving-image dataset inspired by NVIDIA's collected road data.
The goal is architectural understanding, not a claim of reproducing NVIDIA's private dataset or reported performance.

**Sub-repo:** (coming soon)

### 2. CILRS (2019)

[CILRS](https://arxiv.org/abs/1904.08980) extends direct control with high-level navigation commands and separate branches for maneuvers such as turning and following a lane.
The key idea is that conditioning removes action ambiguity at intersections, while the auxiliary speed prediction head makes causal confusion and the inertia problem concrete.
I plan a clean, scoped reimplementation and closed-loop evaluation in CARLA 0.9.10.1.
The implementation will emphasize branch selection, command-balanced data, and recovery behavior rather than broad benchmark coverage.

**Sub-repo:** (coming soon)

### 3. Learning by Cheating (LBC) (2019/2020)

[LBC](https://arxiv.org/abs/1912.12294) first trains a privileged BEV teacher and then distills its behavior into a camera-only student.
Its central move is to separate learning to act from learning to see, predict waypoints instead of raw controls, and use PID for actuation.
I plan a clean reimplementation in CARLA 0.9.10.1, including a focused ChauffeurNet-style trajectory-perturbation ablation for recovery supervision.
The scope will preserve the teacher-student mechanism while keeping data generation and route evaluation feasible on one workstation.

**Sub-repo:** (coming soon)

### 4. TransFuser (+TCP extension) (2021/2022)

[TransFuser](https://arxiv.org/abs/2104.09224) uses transformer attention to exchange information between camera and LiDAR BEV feature streams at multiple scales.
The [TCP extension](https://arxiv.org/abs/2206.08129) adds trajectory and direct-control branches, then chooses how to combine them from the driving situation.
I plan an official-repository-based reproduction in CARLA 0.9.10.1, adapted only where necessary for current PyTorch.
The study will isolate what attention-based fusion and the dual-output design each contribute.

**Sub-repo:** (coming soon)

### 5. PlanT (2022)

[PlanT](https://arxiv.org/abs/2210.14222) replaces dense scene features with compact tokens for vehicles, route elements, and other relevant objects.
Its key idea is that a transformer planner can reason directly over entities, with attention offering a limited but useful view into object relevance.
I plan an official-repository-based reproduction of the privileged planner in CARLA 0.9.10.1, with reduced data volume if required.
This stage focuses on token construction, masking, waypoint prediction, and attention analysis rather than rebuilding a full perception front end.

**Sub-repo:** (coming soon)

### 6. UniAD (2023, CVPR Best Paper)

[UniAD](https://arxiv.org/abs/2212.10156) connects tracking, online mapping, motion forecasting, occupancy prediction, and planning through query-based interfaces in one planning-oriented network.
Its key lesson is not merely multitask learning, but coordinated task representations optimized toward the final planning objective.
Before it, I plan two short drills: a mini VectorNet polyline encoder and a mini Lift-Splat camera-to-BEV module.
The main work will be an official-repository-based reduced reproduction on nuScenes that retains the task interfaces while using a practical training recipe.

**Sub-repo:** (coming soon)

### 7. VAD (+VADv2 ablation) (2023/2024)

[VAD](https://arxiv.org/abs/2303.12077) represents agents and map elements as vectors and uses those instances as explicit constraints for ego planning.
This removes much of the dense rasterized BEV reasoning used by earlier unified systems; the [VADv2](https://arxiv.org/abs/2402.13243) ablation adds a probabilistic trajectory-vocabulary head.
I plan an official-repository-based reduced reproduction on nuScenes with smaller backbones, shorter schedules, and carefully bounded ablations.
The target is to understand vectorized constraints and probabilistic planning, not to reproduce published benchmark numbers.

**Sub-repo:** (coming soon)

### 8. Diffusion Planner (2025, ICLR Oral)

[Diffusion Planner](https://arxiv.org/abs/2501.15564) applies a DiT-style denoiser to structured map and agent-history inputs and jointly samples ego and neighboring trajectories.
Its key contribution for this series is the transition from one regressed future to multimodal generation, with classifier guidance available at inference time.
I plan an official-repository-based reproduction on nuPlan with a focused training and evaluation scope.
The core target is faithful diffusion training, sampling, and guidance behavior rather than exhaustive closed-loop evaluation.

**Sub-repo:** (coming soon)

### 9. DiffusionDrive (2024/2025, CVPR Highlight)

[DiffusionDrive](https://arxiv.org/abs/2411.15139) places a truncated diffusion decoder behind learned sensor features and starts denoising from multimodal trajectory anchors.
Its key idea is that strong priors and a short schedule can preserve diverse outputs while meeting real-time planning constraints.
I plan an official-repository-based reproduction on NAVSIM without changing the central anchor-based diffusion mechanism.
This final stage connects generative planning back to a practical sensor-based end-to-end stack.

**Sub-repo:** (coming soon)
