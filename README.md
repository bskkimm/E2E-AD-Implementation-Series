# E2E-AD-Implementation-Series

This is a hands-on reimplementation series for learning how end-to-end autonomous driving evolved, from direct pixel-to-control imitation to real-time diffusion planning. The models are ordered by the architectural ideas they introduce, and each lives or will eventually live in its own sub-repository. PilotNet is implemented; the remaining models are planned.

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
| 1 | [PilotNet (NVIDIA, 2016)](https://arxiv.org/abs/1604.07316) | A CNN can map raw pixels directly to steering; behavioral cloning reveals saliency and covariate shift. | End-to-end learning is viable at all. | ✅ Implemented | [PilotNet-from-scratch](https://github.com/bskkimm/PilotNet-from-scratch) |
| 2 | [CILRS (2019)](https://arxiv.org/abs/1904.08980) | Command-conditioned branched policies resolve intersection ambiguity; a speed head exposes causal confusion and the inertia problem. | Direct control -> navigation-conditioned control. | 🔜 Planned | (coming soon) |
| 3 | [Learning by Cheating (LBC) (2019/2020)](https://arxiv.org/abs/1912.12294) | Split "learning to see" from "learning to act" with a privileged BEV teacher, a camera student, and waypoint-plus-PID control; include a ChauffeurNet-style trajectory-perturbation ablation. | Expert imitation -> privileged teacher-student learning. | 🔜 Planned | (coming soon) |
| 4 | [TransFuser](https://arxiv.org/abs/2104.09224) + [TCP extension](https://arxiv.org/abs/2206.08129) (2021/2022) | Attention fuses camera and LiDAR BEV features across scales; TCP adds dual trajectory and control branches with situation-based fusion. | Single-sensor learning -> multimodal transformer fusion. | 🔜 Planned | (coming soon) |
| 5 | [PlanT (2022)](https://arxiv.org/abs/2210.14222) | Represent the scene as compact object-level tokens and plan with a transformer; attention weights provide explainability. | Dense pixel/BEV reasoning -> object-centric planning. | 🔜 Planned | (coming soon) |
| 6 | [UniAD (2023, CVPR Best Paper)](https://arxiv.org/abs/2212.10156) | Unify tracking, mapping, motion, occupancy, and planning as query-based tasks in one differentiable, planning-oriented network; first complete mini VectorNet and mini Lift-Splat warm-ups. | Independent task heads -> unified perception-prediction-planning. | 🔜 Planned | (coming soon) |
| 7 | [VAD](https://arxiv.org/abs/2303.12077) + [VADv2 ablation](https://arxiv.org/abs/2402.13243) (2023/2024) | Replace dense rasterized BEV with vectorized agent/map instances as planning constraints; test a probabilistic trajectory-vocabulary head in the VADv2 ablation. | Dense BEV reasoning -> vectorized planning. | 🔜 Planned | (coming soon) |
| 8 | [Diffusion Planner (2025, ICLR Oral)](https://arxiv.org/abs/2501.15564) | A DiT-style model over structured inputs jointly generates ego and neighbor trajectories with flexible classifier guidance. | Deterministic regression -> generative multimodal planning. | 🔜 Planned | (coming soon) |
| 9 | [DiffusionDrive (2024/2025, CVPR Highlight)](https://arxiv.org/abs/2411.15139) | Truncated, anchor-based diffusion makes generative planning real-time inside a sensor-based end-to-end stack. | Structured generative planning -> real-time sensor-based generative planning. | 🔜 Planned | (coming soon) |

PilotNet is implemented; models 2 through 9 remain planned.
The sub-repositories hold implementation notes, code, experiment configuration, and results; this repository remains a compact index.

Short explanations and implementation scope: [Model notes](MODEL_NOTES.md).

## Model comparison

The columns differ by group. For sensor-based end-to-end models, compare the sensors, intermediate representations, training, and evaluation. For vision-language models (VLMs) and vision-language-action models (VLAs), compare the language backbone, what language does, and how actions are produced. Learning-based planners start from structured scene inputs rather than raw sensors.

★ marks models already selected for the implementation series above. Other entries provide context, not additional implementation commitments. Years below refer to the first preprint; conference publication can be later (for example, UniAD appeared at CVPR 2023).

BEV means bird's-eye view; ego means the vehicle being controlled. PID is a feedback controller that converts waypoints into steering and acceleration commands. Imitation learning trains on expert demonstrations; reinforcement learning (RL) trains from rewards. Open-loop evaluation compares predictions with recorded futures; closed-loop evaluation executes the policy and observes the resulting driving behavior.

### A. End-to-end (non-VLM)

| Year | Model | Novelty | Sensor input | Intermediate representation | Output | Training | Evaluation |
|---|---|---|---|---|---|---|---|
| 2016 | ★ [PilotNet](https://arxiv.org/abs/1604.07316) | Pixels map directly to steering | Front camera | No explicit scene representation | Steering | Imitation | Recorded-data simulation + real-road tests (closed-loop) |
| 2019 | ★ [CILRS](https://arxiv.org/abs/1904.08980) | Command-branched policy with an auxiliary speed head | Front camera, speed, command | No explicit scene representation | Steering, throttle, brake | Imitation | CARLA NoCrash (closed-loop) |
| 2019 | ★ [LBC](https://arxiv.org/abs/1912.12294) | Privileged teacher distilled into a camera student | Front camera, speed, command | Ground-truth BEV (teacher only) | Waypoints → PID | Imitation + distillation | CARLA NoCrash (closed-loop) |
| 2021 | [Roach](https://arxiv.org/abs/2108.08265) | RL expert replaces a rule-based expert as teacher | Camera, speed, command (student) | Privileged BEV (teacher only) | Steering, throttle, brake | RL teacher → imitation student | CARLA NoCrash / Leaderboard (closed-loop) |
| 2021/2022 | ★ [TransFuser](https://arxiv.org/abs/2104.09224) + [TCP](https://arxiv.org/abs/2206.08129) | TransFuser fuses camera and LiDAR with attention; TCP combines trajectory and control branches | TransFuser: camera + LiDAR; TCP: front camera; both use ego/navigation inputs | TransFuser: fused image/BEV features; TCP: image features + trajectory/control hidden states | TransFuser: waypoints → PID; TCP: waypoints + control | Imitation | CARLA Longest6 / Leaderboard (closed-loop; varies by paper/version) |
| 2022 | ★ [UniAD](https://arxiv.org/abs/2212.10156) | Query-based tracking, mapping, motion, and occupancy feed planning | Multi-camera + ego/navigation inputs | Track, map, motion, and occupancy queries | Ego trajectory (3 s) | Multi-task imitation | nuScenes (open-loop) |
| 2023 | ★ [VAD](https://arxiv.org/abs/2303.12077) | Vectorized agents and map act as planning constraints | Multi-camera + ego/navigation inputs | Vectorized agents and map elements | Ego trajectory (3 s) | Imitation with auxiliary perception/prediction supervision | nuScenes (open-loop) + CARLA (closed-loop) |
| 2023 | [AD-MLP](https://arxiv.org/abs/2305.10430) | Ego-state-only baseline exposes weaknesses in open-loop evaluation | Ego status only; no scene sensors | No explicit scene representation | Ego trajectory | Imitation | nuScenes (open-loop) |
| 2024 | [SparseDrive](https://arxiv.org/abs/2405.19620) | Fully sparse scene representation; motion and planning predicted in parallel | Multi-camera + ego/navigation inputs | Sparse agent and map instances | Candidate ego trajectories + collision-aware selection | Multi-task imitation | nuScenes (open-loop) |
| 2024 | ★ [DiffusionDrive](https://arxiv.org/abs/2411.15139) | Truncated diffusion starts from trajectory anchors for real-time planning | Camera + LiDAR + ego state | TransFuser BEV features | Candidate ego trajectories + scores; selected ego trajectory | Imitation (diffusion) | NAVSIM (non-reactive simulation-based evaluation) |
| 2025 | [RAD](https://arxiv.org/abs/2502.13144) | RL in 3D Gaussian Splatting reconstructions of real scenes | Multi-camera sequences | BEV features, map tokens, and agent tokens | Discrete lateral and longitudinal actions | Perception + imitation pre-training → RL | Reconstructed 3DGS scenes (closed-loop) |

AD-MLP is included as an evaluation diagnostic: low trajectory error alone does not establish that a model understands the scene or can drive successfully. In RAD, 3D Gaussian Splatting (3DGS) reconstructs a scene so cameras can be rendered from new viewpoints.

### B. VLM/VLA end-to-end

| Year | Model | Novelty | Input | VLM backbone | Language role | Action output | Evaluation |
|---|---|---|---|---|---|---|---|
| 2023 | [DriveLM](https://arxiv.org/abs/2312.14150) | Connects perception, prediction, and planning through a graph of questions and answers | Camera views + questions | BLIP-2 (Flan-T5-XL) | Graph-structured visual question answering | Trajectory coordinates encoded as discrete language tokens | DriveLM-nuScenes / DriveLM-CARLA (open-loop) |
| 2023 | [LMDrive](https://arxiv.org/abs/2312.07488) | Language-guided end-to-end driving evaluated in closed loop | Multi-camera + LiDAR + instructions | LLaVA-v1.5; LLaMA/Vicuna alternatives studied | Navigation instructions and notices | Waypoints from an adapter head → PID | CARLA LangAuto (closed-loop) |
| 2024 | [DriveVLM / DriveVLM-Dual](https://arxiv.org/abs/2402.12289) | Scene reasoning and hierarchical planning; Dual pairs the slow VLM with a fast conventional driving stack | Multi-frame camera; Dual also uses the conventional stack's inputs | Qwen-VL | Chain-of-thought: describe → analyze → plan | Coarse trajectory; Dual refines it with a conventional planner | nuScenes planning + SUP-AD scene understanding (open-loop) |
| 2024 | [EMMA (Waymo)](https://arxiv.org/abs/2410.23262) | One multimodal model represents driving tasks through text inputs and outputs | Surround cameras + navigation and ego history as text | Gemini | Task prompts, text representations, and chain-of-thought rationale | Trajectory as text tokens | nuScenes, WOMD (open-loop) |
| 2025 | [SimLingo](https://arxiv.org/abs/2503.09594) | Aligns language and driving actions through action dreaming | Front camera + speed + navigation (target points or language) | InternVL2-1B (Qwen2-0.5B-Instruct language model) | Visual question answering, commentary, and instructions | Path waypoints + speed waypoints | CARLA Leaderboard 2.0, Bench2Drive (closed-loop) |
| 2025 | [Alpamayo-R1 (NVIDIA)](https://arxiv.org/abs/2511.00088) | Reasoning VLA with causal reasoning traces and RL post-training | Multi-camera video + ego-motion history; optional navigation | Cosmos-Reason | Causal reasoning traces linked to driving decisions | Continuous trajectory from a flow-matching decoder | Closed-loop simulation + real-road demonstration |

### C. Learning-based planning (structured input → learned model → trajectory)

| Year | Model | Novelty | Input | Output |
|---|---|---|---|---|
| 2018 | [ChauffeurNet](https://arxiv.org/abs/1812.03079) | Mid-level BEV representation; perturbed training examples teach recovery | Raster BEV: map, traffic lights, route, agent boxes/history, and ego past | Ego waypoints, heading, and speed + auxiliary heads |
| 2021 | [Urban Driver](https://arxiv.org/abs/2109.13333) | Closed-loop imitation training with policy gradients in a differentiable simulator | Vectorized agents, map, and ego history | Ego trajectory |
| 2022 | ★ [PlanT](https://arxiv.org/abs/2210.14222) | Transformer over object tokens; attention helps inspect object relevance | Vehicle and route tokens from privileged CARLA state | Ego waypoints + auxiliary vehicle forecasts |
| 2023 | [GameFormer](https://arxiv.org/abs/2303.05760) | Iterative interaction prediction and planning using level-k reasoning (each level responds to the previous level's predictions) | Vectorized agents, map, and ego state | Ego + agent trajectories |
| 2023 | [PlanTF](https://arxiv.org/abs/2309.10443) | Diagnoses ego-state shortcuts; combines state dropout and data augmentation | Vectorized agents, map, and current ego state (no ego history) | Ego trajectory + agent predictions |
| 2024 | [PLUTO](https://arxiv.org/abs/2404.14327) | Contrastive imitation learning with lateral/longitudinal candidate queries | Vectorized agents, map, static obstacles, and ego state | Candidate ego trajectories + scores |
| 2025 | ★ [Diffusion Planner](https://arxiv.org/abs/2501.15564) | Diffusion transformer jointly denoises ego and neighbor trajectories; guidance applied at inference | Vectorized agent histories, lanes, route, and ego state | Ego + neighbor trajectories |
