# Learning-Based-AD-Implementation-Series

This is a hands-on implementation series for learning-based autonomous driving, covering sensor-to-action policies, vision-language-action models, and planners with structured scene inputs. The models are grouped by approach and the architectural ideas they introduce. PilotNet is implemented; each table tracks implementation status and repository availability.

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

Short explanations and implementation scope: [Model notes](MODEL_NOTES.md).

## Model comparison

The columns differ by group. For sensor-based end-to-end models, compare the sensors, intermediate representations, training, and evaluation. For vision-language models (VLMs) and vision-language-action models (VLAs), compare the language backbone, what language does, and how actions are produced. Learning-based planners start from structured scene inputs rather than raw sensors.

Status / Repository refers to implementations in this series: Implemented links to the available repository, Planned identifies scheduled work with a repository to come, and Not scheduled identifies reference models without an implementation commitment. Years refer to the first preprint; conference publication can be later (for example, UniAD appeared at CVPR 2023).

BEV means bird's-eye view; ego means the vehicle being controlled. PID is a feedback controller that converts waypoints into steering and acceleration commands. Imitation learning trains on expert demonstrations; reinforcement learning (RL) trains from rewards. Open-loop evaluation compares predictions with recorded futures; closed-loop evaluation executes the policy and observes the resulting driving behavior.

### A. Sensor-to-action end-to-end

| Model (year) | Novelty | Sensor input | Intermediate representation | Output | Training | Evaluation | Status / Repository |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [PilotNet](https://arxiv.org/abs/1604.07316) (2016) | Pixels map directly to steering | Front camera | No explicit scene representation | Steering | Imitation | Recorded-data simulation + real-road tests (closed-loop) | Implemented · [PilotNet-from-scratch](https://github.com/bskkimm/PilotNet-from-scratch) |
| [CILRS](https://arxiv.org/abs/1904.08980) (2019) | Command-branched policy with an auxiliary speed head | Front camera, speed, command | No explicit scene representation | Steering, throttle, brake | Imitation | CARLA NoCrash (closed-loop) | Planned · Repository coming soon |
| [LBC](https://arxiv.org/abs/1912.12294) (2019) | Privileged teacher distilled into a camera student | Front camera, speed, command | Ground-truth BEV (teacher only) | Waypoints → PID | Imitation + distillation | CARLA NoCrash (closed-loop) | Planned · Repository coming soon |
| [Roach](https://arxiv.org/abs/2108.08265) (2021) | RL expert replaces a rule-based expert as teacher | Camera, speed, command (student) | Privileged BEV (teacher only) | Steering, throttle, brake | RL teacher → imitation student | CARLA NoCrash / Leaderboard (closed-loop) | Not scheduled · — |
| [TransFuser](https://arxiv.org/abs/2104.09224) + [TCP](https://arxiv.org/abs/2206.08129) (2021/2022) | TransFuser fuses camera and LiDAR with attention; TCP combines trajectory and control branches | TransFuser: camera + LiDAR; TCP: front camera; both use ego/navigation inputs | TransFuser: fused image/BEV features; TCP: image features + trajectory/control hidden states | TransFuser: waypoints → PID; TCP: waypoints + control | Imitation | CARLA Longest6 / Leaderboard (closed-loop; varies by paper/version) | Planned · Repository coming soon |
| [UniAD](https://arxiv.org/abs/2212.10156) (2022) | Query-based tracking, mapping, motion, and occupancy feed planning | Multi-camera + ego/navigation inputs | Track, map, motion, and occupancy queries | Ego trajectory (3 s) | Multi-task imitation | nuScenes (open-loop) | Planned · Repository coming soon |
| [VAD](https://arxiv.org/abs/2303.12077) (2023) | Vectorized agents and map act as planning constraints | Multi-camera + ego/navigation inputs | Vectorized agents and map elements | Ego trajectory (3 s) | Imitation with auxiliary perception/prediction supervision | nuScenes (open-loop) + CARLA (closed-loop) | Planned · Repository coming soon |
| [AD-MLP](https://arxiv.org/abs/2305.10430) (2023) | Ego-state-only baseline exposes weaknesses in open-loop evaluation | Ego status only; no scene sensors | No explicit scene representation | Ego trajectory | Imitation | nuScenes (open-loop) | Not scheduled · — |
| [SparseDrive](https://arxiv.org/abs/2405.19620) (2024) | Fully sparse scene representation; motion and planning predicted in parallel | Multi-camera + ego/navigation inputs | Sparse agent and map instances | Candidate ego trajectories + collision-aware selection | Multi-task imitation | nuScenes (open-loop) | Not scheduled · — |
| [DiffusionDrive](https://arxiv.org/abs/2411.15139) (2024) | Truncated diffusion starts from trajectory anchors for real-time planning | Camera + LiDAR + ego state | TransFuser BEV features | Candidate ego trajectories + scores; selected ego trajectory | Imitation (diffusion) | NAVSIM (non-reactive simulation-based evaluation) | Planned · Repository coming soon |
| [RAD](https://arxiv.org/abs/2502.13144) (2025) | RL in 3D Gaussian Splatting reconstructions of real scenes | Multi-camera sequences | BEV features, map tokens, and agent tokens | Discrete lateral and longitudinal actions | Perception + imitation pre-training → RL | Reconstructed 3DGS scenes (closed-loop) | Not scheduled · — |

AD-MLP is included as an evaluation diagnostic: low trajectory error alone does not establish that a model understands the scene or can drive successfully. In RAD, 3D Gaussian Splatting (3DGS) reconstructs a scene so cameras can be rendered from new viewpoints.

### B. VLM/VLA end-to-end

| Model (year) | Novelty | Input | VLM backbone | Language role | Action output | Evaluation | Status / Repository |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DriveLM](https://arxiv.org/abs/2312.14150) (2023) | Connects perception, prediction, and planning through a graph of questions and answers | Camera views + questions | BLIP-2 (Flan-T5-XL) | Graph-structured visual question answering | Trajectory coordinates encoded as discrete language tokens | DriveLM-nuScenes / DriveLM-CARLA (open-loop) | Not scheduled · — |
| [LMDrive](https://arxiv.org/abs/2312.07488) (2023) | Language-guided end-to-end driving evaluated in closed loop | Multi-camera + LiDAR + instructions | LLaVA-v1.5; LLaMA/Vicuna alternatives studied | Navigation instructions and notices | Waypoints from an adapter head → PID | CARLA LangAuto (closed-loop) | Not scheduled · — |
| [DriveVLM / DriveVLM-Dual](https://arxiv.org/abs/2402.12289) (2024) | Scene reasoning and hierarchical planning; Dual pairs the slow VLM with a fast conventional driving stack | Multi-frame camera; Dual also uses the conventional stack's inputs | Qwen-VL | Chain-of-thought: describe → analyze → plan | Coarse trajectory; Dual refines it with a conventional planner | nuScenes planning + SUP-AD scene understanding (open-loop) | Not scheduled · — |
| [EMMA (Waymo)](https://arxiv.org/abs/2410.23262) (2024) | One multimodal model represents driving tasks through text inputs and outputs | Surround cameras + navigation and ego history as text | Gemini | Task prompts, text representations, and chain-of-thought rationale | Trajectory as text tokens | nuScenes, WOMD (open-loop) | Not scheduled · — |
| [SimLingo](https://arxiv.org/abs/2503.09594) (2025) | Aligns language and driving actions through action dreaming | Front camera + speed + navigation (target points or language) | InternVL2-1B (Qwen2-0.5B-Instruct language model) | Visual question answering, commentary, and instructions | Path waypoints + speed waypoints | CARLA Leaderboard 2.0, Bench2Drive (closed-loop) | Not scheduled · — |
| [Alpamayo-R1 (NVIDIA)](https://arxiv.org/abs/2511.00088) (2025) | Reasoning VLA with causal reasoning traces and RL post-training | Multi-camera video + ego-motion history; optional navigation | Cosmos-Reason | Causal reasoning traces linked to driving decisions | Continuous trajectory from a flow-matching decoder | Closed-loop simulation + real-road demonstration | Not scheduled · — |

### C. Learning-based planning (structured input → learned model → trajectory)

| Model (year) | Novelty | Input | Output | Status / Repository |
| --- | --- | --- | --- | --- |
| [ChauffeurNet](https://arxiv.org/abs/1812.03079) (2018) | Mid-level BEV representation; perturbed training examples teach recovery | Raster BEV: map, traffic lights, route, agent boxes/history, and ego past | Ego waypoints, heading, and speed + auxiliary heads | Not scheduled · — |
| [Urban Driver](https://arxiv.org/abs/2109.13333) (2021) | Closed-loop imitation training with policy gradients in a differentiable simulator | Vectorized agents, map, and ego history | Ego trajectory | Not scheduled · — |
| [PlanT](https://arxiv.org/abs/2210.14222) (2022) | Transformer over object tokens; attention helps inspect object relevance | Vehicle and route tokens from privileged CARLA state | Ego waypoints + auxiliary vehicle forecasts | Planned · Repository coming soon |
| [GameFormer](https://arxiv.org/abs/2303.05760) (2023) | Iterative interaction prediction and planning using level-k reasoning (each level responds to the previous level's predictions) | Vectorized agents, map, and ego state | Ego + agent trajectories | Not scheduled · — |
| [PlanTF](https://arxiv.org/abs/2309.10443) (2023) | Diagnoses ego-state shortcuts; combines state dropout and data augmentation | Vectorized agents, map, and current ego state (no ego history) | Ego trajectory + agent predictions | Not scheduled · — |
| [PLUTO](https://arxiv.org/abs/2404.14327) (2024) | Contrastive imitation learning with lateral/longitudinal candidate queries | Vectorized agents, map, static obstacles, and ego state | Candidate ego trajectories + scores | Not scheduled · — |
| [Diffusion Planner](https://arxiv.org/abs/2501.15564) (2025) | Diffusion transformer jointly denoises ego and neighbor trajectories; guidance applied at inference | Vectorized agent histories, lanes, route, and ego state | Ego + neighbor trajectories | Planned · Repository coming soon |
