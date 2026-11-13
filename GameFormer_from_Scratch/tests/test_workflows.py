"""User-visible commands, measured scopes and missing-asset error paths."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

import torch

from gameformer.config import ModelConfig, recipe
from gameformer.data import validate_sample
from gameformer.performance import benchmark, tune
from gameformer.prepare import scenario_sample


def test_waymo_boundary_preserves_true_zero_coordinates():
    config = ModelConfig.smoke()
    tracks = []
    for i in range(4):
        states = [
            NS(
                center_x=float(j - 10 + i),
                center_y=0.0,
                heading=0.0,
                velocity_x=10.0,
                velocity_y=0.0,
                length=4.0,
                width=2.0,
                height=1.5,
                valid=True,
            )
            for j in range(91)
        ]
        tracks.append(NS(id=i + 1, object_type=1, states=states))
    scene = NS(tracks=tracks, map_features=[], dynamic_map_states=[], current_time_index=10)
    sample = scenario_sample(scene, (0, 1), config)
    validate_sample(sample, config)
    torch.testing.assert_close(sample["ego_state"][-1, :2], torch.tensor([0.0, 0.0]))
    assert sample["actor_valid"][0].all() and sample["future_valid"].all()
    assert not sample["lane_valid"].any()


def test_benchmark_reports_actual_training_scope_and_tuner_keeps_recipe():
    config = replace(recipe("smoke"), synthetic=True)
    result = benchmark(config, mode="end_to_end", steps=2, warmup=1)
    assert result["samples"] == 4 and result["optimizer_updates"] == 2
    assert result["samples_per_second"] > 0 and result["loader_wait_seconds"] >= 0
    assert result["scope"] == "end_to_end"
    report = tune(config, max_trials=1, time_limit=5.0, steps=1)
    assert report["selected"]["model"] == config.to_dict()["model"]
    assert (
        report["selected"]["microbatch"] * report["selected"]["accumulation"]
        == config.effective_batch
    )
    assert report["selection_claim"] == "best measured among tested candidates"


def test_train_cli_and_missing_real_data(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gameformer.cli",
            "train",
            "--config",
            "configs/smoke.json",
            "--synthetic",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "result.json").read_text())["completed_epochs"] == 2
    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "gameformer.cli",
            "train",
            "--config",
            "configs/smoke.json",
            "--output",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0 and "no synthetic fallback" in missing.stderr
