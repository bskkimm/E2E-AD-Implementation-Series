"""Re-run bounded CPU evidence; checkpoints remain under ignored outputs/."""

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from gameformer.config import recipe
from gameformer.engine import build, environment_record, evaluate, make_loaders, train, update
from gameformer.losses import objective
from gameformer.performance import benchmark, tune

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/evidence")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = replace(recipe("smoke"), synthetic=True)
    result = train(config, output / "smoke")
    raw, model, _, _, _ = build(config)
    from gameformer.engine import load_checkpoint

    raw.load_state_dict(load_checkpoint(output / "smoke/latest.pt")["model"])
    _, validation = make_loaders(config, torch.Generator().manual_seed(config.seed))
    metrics, _ = evaluate(model, validation, config)
    write(ROOT / "docs/evidence/cpu-smoke.json", {"result": result, "metrics": metrics})
    measurements = {
        scope: benchmark(config, scope, steps=10, warmup=2)
        for scope in ("loader", "step", "end_to_end")
    }
    write(ROOT / "docs/evidence/cpu-benchmark.json", measurements)
    write(
        ROOT / "docs/evidence/cpu-tuning.json", tune(config, max_trials=4, time_limit=20, steps=3)
    )
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    group = [next(iter(loader))]
    _, model, optimizer, _, scaler = build(config)
    model.eval()
    with torch.no_grad():
        before = float(objective(model(group[0]), group[0], "joint")["loss"])
    for _ in range(40):
        update(model, optimizer, scaler, group, config)
    model.eval()
    with torch.no_grad():
        after = float(objective(model(group[0]), group[0], "joint")["loss"])
    write(
        ROOT / "docs/evidence/tiny-learning.json",
        {
            "updates": 40,
            "before_loss": before,
            "after_loss": after,
            "protocol": "Repeated fixed two-scene training batch; actual all-level joint objective; synthetic overfit, not validation",
        },
    )
    environment = environment_record(config)
    source_files = [
        *sorted((ROOT / "gameformer").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.json")),
    ]
    source_hash = hashlib.sha256(
        "".join(f"{path.relative_to(ROOT)}:{sha(path)}\n" for path in source_files).encode()
    ).hexdigest()
    artifacts = [
        *sorted((ROOT / "docs/evidence").glob("*.json")),
        ROOT / "docs/upstream-parity.json",
        ROOT / "docs/dependency-audit.json",
        ROOT / "notebooks/assets/architecture/manifest.json",
    ]
    write(
        ROOT / "docs/results-manifest.json",
        {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "paper_results_reproduced": False,
            "smoke_run": {
                "code_commit": environment["code_commit"],
                "dirty_state": environment["dirty_state"],
                "source_sha256": source_hash,
                "source_hash_protocol": "SHA256 of ordered path:SHA256 lines; gameformer/*.py then configs/*.json",
                "config_sha256": hashlib.sha256(
                    json.dumps(config.to_dict(), sort_keys=True).encode()
                ).hexdigest(),
                "checkpoint_sha256": sha(output / "smoke/latest.pt"),
                "checkpoint_path": str(output / "smoke/latest.pt"),
                "seed": config.seed,
                "dataset": "seeded constant-velocity synthetic scenes",
                "split": {
                    "train": {"size": 8, "seed": config.seed},
                    "validation": {"size": 4, "seed": config.seed + 100000},
                },
                "metric_protocol": "Inference-selected shared final-level mode; ADE per valid timestep, FDE per valid agent at last valid label; meters",
                "metrics": metrics,
                "config": config.to_dict(),
                "environment": environment,
            },
            "artifacts": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in artifacts
            ],
            "unmet_scientific_gates": [
                "real Waymo cache validation",
                "actual official SDK evaluation",
                "CUDA precision/gradient/memory checks",
                "budgeted full training and official split results",
            ],
        },
    )
    print("Recorded bounded evidence and results manifest")


if __name__ == "__main__":
    main()
