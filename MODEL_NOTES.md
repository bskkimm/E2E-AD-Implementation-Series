# Model notes

Short explanations and implementation scope for the nine models in the [implementation series](README.md#the-series). PilotNet is implemented; the remaining models are planned.

## 1. PilotNet (NVIDIA, 2016)

[PilotNet](https://arxiv.org/abs/1604.07316) maps a front-camera image directly to steering. It shows how imitation can learn road features without explicit perception labels, while exposing failures when the vehicle leaves familiar training situations. The implementation includes the architecture, CSV training workflow, validation, and public dataset documentation; it uses substitute data rather than reproducing NVIDIA's private dataset or driving results.

**Sub-repo:** [PilotNet-from-scratch](https://github.com/bskkimm/PilotNet-from-scratch)

## 2. CILRS (2019)

[CILRS](https://arxiv.org/abs/1904.08980) selects control branches using navigation commands and adds an auxiliary speed prediction head. The planned CARLA 0.9.10.1 implementation focuses on command-balanced training, branch selection, and recovery behavior.

## 3. Learning by Cheating (LBC) (2019/2020)

[LBC](https://arxiv.org/abs/1912.12294) trains a teacher with ground-truth scene information, then transfers its behavior to a camera student. The planned CARLA 0.9.10.1 implementation covers teacher-student training, waypoint prediction, feedback control, and a focused recovery-training experiment inspired by ChauffeurNet.

## 4. TransFuser (+TCP extension) (2021/2022)

[TransFuser](https://arxiv.org/abs/2104.09224) uses attention to combine camera and LiDAR features; [TCP](https://arxiv.org/abs/2206.08129) combines trajectory and direct-control branches. The planned study builds on official repositories in CARLA 0.9.10.1 and isolates the contributions of sensor fusion and the dual-output design.

## 5. PlanT (2022)

[PlanT](https://arxiv.org/abs/2210.14222) plans over compact vehicle and route tokens, with attention helping inspect which objects influence decisions. The planned CARLA implementation focuses on token construction, masking, waypoints, and attention analysis using privileged scene inputs.

## 6. UniAD (2023, CVPR Best Paper)

[UniAD](https://arxiv.org/abs/2212.10156) connects tracking, mapping, motion prediction, occupancy, and planning through learned queries. The planned reduced nuScenes reproduction preserves those task interfaces, preceded by small VectorNet and Lift-Splat exercises for structured map encoding and camera-to-bird's-eye-view projection.

## 7. VAD (+VADv2 ablation) (2023/2024)

[VAD](https://arxiv.org/abs/2303.12077) uses vectorized agents and map elements to constrain planning. The planned reduced nuScenes reproduction studies these constraints, with a [VADv2](https://arxiv.org/abs/2402.13243) experiment comparing a probabilistic trajectory-vocabulary output head.

## 8. Diffusion Planner (2025, ICLR Oral)

[Diffusion Planner](https://arxiv.org/abs/2501.15564) jointly generates ego and neighboring trajectories by progressively removing noise, with guidance available during inference. The planned nuPlan reproduction focuses on diffusion training, sampling, and guidance using structured map and agent-history inputs.

## 9. DiffusionDrive (2024/2025, CVPR Highlight)

[DiffusionDrive](https://arxiv.org/abs/2411.15139) starts from trajectory anchors and uses a short denoising schedule to make sensor-based generative planning fast. The planned NAVSIM reproduction preserves the anchor-based diffusion mechanism and studies its planning quality and runtime.
