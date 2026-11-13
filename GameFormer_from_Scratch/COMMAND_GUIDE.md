# Commands

Run inside `GameFormer_from_Scratch`, after [setup](docs/setup.md). Commands below
use the installed `.venv/bin/gameformer` entry point. The default configuration
path is relative to this directory; pass an absolute `--config` when elsewhere.
Synthetic execution must be requested explicitly. An absent real asset never
falls back to synthetic input.

## CPU training and evaluation

```bash
OMP_NUM_THREADS=1 .venv/bin/gameformer train --config configs/smoke.json \
  --synthetic --output outputs/smoke
.venv/bin/gameformer eval --config configs/smoke.json --synthetic \
  --checkpoint outputs/smoke/latest.pt --output outputs/smoke-eval.json
```

The default smoke run is two epochs with eight training and four validation
scenes, seed 3407, microbatch two, FP32 and one process. Inspect `config.json`,
`environment.json`, `metrics.jsonl`, `progress.json`, `latest.pt`, and the trajectory
figures under the output directory. Metrics prefixed `diagnostic/` are selected
using inference scores; they are not official Waymo benchmark results.

## Resume and MLflow

```bash
OMP_NUM_THREADS=1 .venv/bin/gameformer train --config configs/smoke.json \
  --synthetic --resume outputs/smoke/latest.pt --output outputs/smoke
mkdir -p outputs
OMP_NUM_THREADS=1 .venv/bin/gameformer train --config configs/smoke.json \
  --synthetic --output outputs/tracked \
  --tracking-uri "sqlite:///$PWD/outputs/tracking.db"
.venv/bin/mlflow ui --backend-store-uri "sqlite:///$PWD/outputs/tracking.db" \
  --host 127.0.0.1 --port 5000
```

Recovery is at a **completed epoch boundary**. A crash during an epoch replays it.
A fully finished run has no remaining epochs. Checkpoints save model, AdamW,
scheduler, scaler, RNG/shuffle state, run ID and timing. Recipe changes are rejected;
device and loader settings can change. Schema-2 checkpoints also compare both cache
manifest hashes before restoring state or writing session artifacts. Schema-1
checkpoints cannot provide this recovery guarantee and are rejected. The sampler's
generator is independent of worker startup; exact recovery covers deterministic
datasets, including persistent workers, without random worker augmentation.
Original `config.json` and `environment.json` remain intact. Every start adds unique
`sessions/<session>/config.json` and `environment.json` snapshots locally and in MLflow.
Tracking displays the current runtime configuration separately. Only load trusted
project checkpoints: recovery includes Python/NumPy state.

Edit a copy of the JSON to change a run's settings. MLflow remains optional, records
progress and finish time in `Asia/Tokyo` by default, and includes observed validation
and checkpoint overhead in the new session rate. Estimates are provisional before
two completed session epochs. Resume excludes downtime from active time; the saved
active boundary is immediately before the checkpoint write. Current progress
includes that write; recovery can undercount that last write and post-save logging.
Checkpoint uploads are opt-in through `upload_checkpoint`; local atomic saves always
occur.

## Prepare real data and evaluate officially (opt-in)

Obtain authorized Waymo v1.1 data and a compatible SDK separately, as described in
[the data guide](docs/data.md). The local preparation method is explicitly practical.
These examples require those assets; they have not been run against real shards.

```bash
.venv/bin/gameformer prepare --config configs/practical_joint.json \
  --source /path/to/scenario/training --split train \
  --max-scenarios 100 --max-records 200 --output data/train
.venv/bin/gameformer prepare --config configs/practical_joint.json \
  --source /path/to/scenario/validation_interactive --split validation \
  --max-scenarios 100 --max-records 100 --output data/validation
```

In a copied config set `train_data` and `validation_data` to these cache directories,
set `synthetic` false, and choose an authorized budget. Cache training uses all records
in the supplied manifests; `train_size`/`validation_size` apply only to synthetic data.

```bash
.venv/bin/gameformer train --config outputs/my-real-config.json --output outputs/real
.venv/bin/gameformer eval --config outputs/my-real-config.json \
  --checkpoint outputs/real/latest.pt --official --output outputs/official-eval.json
```

The current official adapter is for two-agent, 80-step joint prediction. It preserves
joint modes, resamples to 2 Hz, and calls the actual SDK. A subset remains a subset;
the split and provenance determine what can be claimed. Planning requires supplied
conforming caches or synthetic fixtures; raw planning preparation and simulator
evaluation are not implemented.

## Benchmarks and bounded tuning

```bash
OMP_NUM_THREADS=1 .venv/bin/gameformer benchmark --config configs/smoke.json \
  --synthetic --mode loader --steps 10 --warmup 2 --output outputs/loader.json
OMP_NUM_THREADS=1 .venv/bin/gameformer benchmark --config configs/smoke.json \
  --synthetic --mode step --steps 10 --warmup 2 --output outputs/step.json
OMP_NUM_THREADS=1 .venv/bin/gameformer benchmark --config configs/smoke.json \
  --synthetic --mode end_to_end --steps 10 --warmup 2 --output outputs/end-to-end.json
OMP_NUM_THREADS=1 .venv/bin/gameformer tune --config configs/smoke.json \
  --synthetic --max-trials 3 --time-limit 20 --steps 3 --output outputs/tuning.json
```

Startup is separate. Step timing caches a CPU batch and includes any device transfer;
end-to-end timing loads each group and includes loss/backward/AdamW. Epoch validation,
checkpointing and MLflow are outside those training-path benchmark scopes.
Worker loading has a configurable `loader_timeout` in seconds (default 30);
tuning caps a worker trial's timeout at its remaining budget.
The tuner keeps data/model/effective batch/objective/schedule fixed, compares against
FP32, and saves every attempted candidate. Its output/gradient probe forwards one
microbatch at a time and combines only detached comparison outputs on CPU. Its
deadline is checked between candidates
and timed steps; initialization, correctness probes and an atomic update may overrun.
See [measurement limits](docs/benchmark.md).

## Notebook, source parity and all checks

```bash
.venv/bin/python -m gameformer.teaching build
OMP_NUM_THREADS=1 .venv/bin/python -m gameformer.teaching check
UV_CACHE_DIR=/tmp/gameformer-uv-cache .venv/bin/python scripts/check_all.py
.venv/bin/python scripts/check_all.py --stage integration
```

Rebuild teaching assets after changing a referenced source. Keep the notebook's
outputs empty and counts null. Open it with the `.venv` Python kernel to follow the
lesson interactively; default execution is CPU, offline and synthetic.

The reference parity script loads only an explicitly supplied official checkout and
requires the pinned revision. It does not download code or pretrained weights:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/check_upstream.py \
  --source /path/to/MCZhi-GameFormer --output outputs/upstream-parity.json
```

All subcommands support `--help`. `--device cuda` requires working hardware and a
separately validated CUDA build; the committed lock installs the CPU build.
