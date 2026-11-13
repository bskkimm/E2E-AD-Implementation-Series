# GameFormer from scratch

A readable, local PyTorch implementation of **GameFormer** (ICCV 2023):
predict several possible futures, then refine them as agents respond to each other.
The primary task is Waymo joint interaction prediction. The same model supports
the released Waymo open-loop planning objective.

**Status:** CPU synthetic execution, scientific fixtures, notebook/package parity,
and a bounded comparison with the official source are tested. **Paper results
have not been reproduced.** Real Waymo data, the official evaluation SDK and a
working CUDA device were unavailable. See [results and limits](docs/results.md).

```bash
python3.11 -m pip install uv==0.11.16
uv sync --frozen --extra dev --extra tracking
OMP_NUM_THREADS=1 .venv/bin/gameformer train \
  --config configs/smoke.json --synthetic --output outputs/demo
.venv/bin/gameformer eval --config configs/smoke.json --synthetic \
  --checkpoint outputs/demo/latest.pt --output outputs/demo-eval.json
```

Run these commands inside `GameFormer_from_Scratch`. The smoke recipe keeps the
80-step prediction horizon and actual loss while reducing model width and context.
Choose a fresh output directory or explicitly resume. No dataset is downloaded.

```text
Observed histories + local map
              |
        Scene Transformer
              |
       Level 0: initial futures
              |
       Level 1 ... K: responses to previous futures
              |
      Joint modes, uncertainty and scores
              |
       Loss at every level / inference by scores
```

The [standalone walkthrough](notebooks/gameformer_e2e_walkthrough.ipynb) defines
the entire architecture locally, explains tensor shapes and gradient flow, and
performs a real optimizer update. It has [18 worked illustrations](notebooks/assets/architecture/README.md)
with editable SVGs and offline image attachments. Predictions in that lesson are
random synthetic diagnostics.

Use the [command guide](COMMAND_GUIDE.md) for training, recovery, MLflow tracking,
preparation, evaluation, benchmarks and bounded tuning. Supporting references:

- [Setup and tested environment](docs/setup.md)
- [Data contract and practical preprocessing](docs/data.md)
- [Paper-to-code mapping and deliberate deviations](docs/paper-to-code.md)
- [Performance measurement](docs/benchmark.md)
- [Results manifest](docs/results-manifest.json)
- [CI and review status](docs/ci-review.md)

All architecture, loss and training code lives in `gameformer/`; no model or
training framework is imported. Initialization is random. `released_joint`,
`released_planning`, `paper_joint_reconstruction`, `practical_joint` and `smoke`
are separate named recipes. In particular, published K6 reasoning and released
K3 reasoning are distinguished. Explicit validity masks preserve coordinates at
zero; the preprocessing path is a documented practical adaptation.

The containing series repository installs the workflow at
`../.github/workflows/gameformer.yml`. Run the same checks locally with:

```bash
UV_CACHE_DIR=/tmp/gameformer-uv-cache .venv/bin/python scripts/check_all.py
```

Dependency auditing and the first actionlint download require network access;
scientific and notebook tests run offline. Full training needs an explicit data
and compute budget. Displacement diagnostics do not establish closed-loop driving
quality; nuPlan refinement, the marginal EM ensemble and submission packaging are
outside this implementation.

Sources: [paper](https://arxiv.org/abs/2303.05760),
[supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Huang_GameFormer_Game-theoretic_Modeling_ICCV_2023_supplemental.pdf),
[official code at fcb0d4a](https://github.com/MCZhi/GameFormer/tree/fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128).
This project follows the user's private playbook at reviewed commit
`707c64b138d5e6c534af3d197fd4f278aab03d45`; public CI needs no private access.
