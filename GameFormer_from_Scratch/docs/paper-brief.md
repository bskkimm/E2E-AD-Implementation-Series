# GameFormer implementation brief

- **Paper:** Huang, Liu and Lv, *GameFormer: Game-theoretic Modeling and Learning
  of Transformer-based Interactive Prediction and Planning for Autonomous
  Driving*, ICCV 2023, pp. 3903–3913;
  [arXiv 2303.05760](https://arxiv.org/abs/2303.05760).
- **Supplement:** [official supplementary material](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Huang_GameFormer_Game-theoretic_Modeling_ICCV_2023_supplemental.pdf),
  particularly A.1 (prediction), A.2 (planning), A.5 (nuPlan).
- **Official source:** [MCZhi/GameFormer](https://github.com/MCZhi/GameFormer/tree/fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128),
  pinned to `fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128`.
- **Implementation:** current `GameFormer_from_Scratch` directory in
  `bskkimm/E2E-AD-Implementation-Series`, base `f3d5512`; no separate repository
  has been created. Publishing/review settings must be verified, not assumed.
- **Playbook:** `707c64b138d5e6c534af3d197fd4f278aab03d45`, reviewed main.
- **Audience:** learners who know Python and basic PyTorch; explain geometry,
  attention, multimodal futures and every loss in a standalone walkthrough.
- **Target:** released WOMD six-mode joint interaction predictor first; shared
  architecture also supports the released WOMD open-loop planning variant.
  The user asked to keep working after scope options were presented; this is the
  stated default, not a claim that they selected a specific benchmark score.
- **Fidelity:** release-compatible operators and named recipe differences; correct
  explicit validity masks are a documented deviation from zero-coordinate tests.
  The 64-mode EM ensemble and nuPlan refinement/simulation are separate work.
- **Data:** Waymo Open Motion Dataset v1.1, scenario/training and
  scenario/validation_interactive for interaction prediction. No local path or
  dataset access was supplied. SDK/preprocessing/evaluation stay optional.
- **Weights:** random initialization throughout; no pretrained weights supplied.
- **Environment inspected:** Python 3.10.12/system Torch 1.8 are unsuitable as
  the supported runtime. Prepare an isolated, locked Python >=3.11 environment.
  32 logical CPUs and approximately 62.6 GiB host RAM are visible; these are not
  guaranteed allocations. `nvidia-smi` cannot communicate with the driver.
- **Budget:** CPU synthetic checks and small correctness/benchmark runs only.
  Full training, dataset acquisition and paid GPU use are not authorized.
- **Released joint recipe:** D256, 8 heads, 6 encoder layers, K3 interaction
  levels plus level 0, 6 modes, 11 history samples, 80 future steps at 10 Hz,
  32 background agents plus two targets; AdamW 1e-4, decay .01, 30 epochs,
  clip norm 5; epoch milestones 20/22/24/26/28, gamma .5; 16/GPU in released
  trainer. Paper A.1 instead states 64/GPU on four V100s and a different schedule.
- **Planning recipe:** ego plus 10 predicted neighbors, 50 future steps,
  K4, 6 encoder layers, 6 modes; imitation plus .1 interaction loss after level 0;
  20 epochs, batch 32, AdamW 1e-4, milestones 10/12/14/16/18.
- **Evaluation:** joint WOMD official SDK minADE/minFDE/miss/overlap/mAP,
  2 Hz samples with 3/5/8-second measurements. Diagnostics have separate names.
  Planning displacement diagnostics do not establish closed-loop driving quality.
- **CPU criteria:** analytical geometry, partial/empty masks, finite gradients,
  joint mode selection, tiny-batch learning, resume equivalence, tracking/ETA,
  offline notebook execution/parity and reproducible figure attachments.
- **Real/GPU gates:** actual cache provenance, official SDK results and GPU
  precision/memory checks need their assets; remain unmet until run.
- **Notebook:** local component definitions, equations/shapes, worked figures,
  full optimization step, joint decoding, metrics, tracking and resume.
- **Review:** existing pra preferred; installation, private-code provider policy
  and current-head review remain to be checked. No human approval is fabricated.
- **Delivery:** source/data/model; training/tracking; notebook; evaluation and
  performance, with independent checks and final current-head review.
