"""User-visible commands, measured scopes and missing-asset error paths."""

import json
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from gameformer.config import ModelConfig, recipe
from gameformer.data import validate_sample
from gameformer.engine import make_loaders
from gameformer.performance import benchmark, tune
from gameformer.prepare import scenario_sample


class SlowSceneDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        time.sleep(0.5)
        return index


def test_loader_timeout_bounds_a_stalled_worker(monkeypatch):
    monkeypatch.setattr("gameformer.engine.SyntheticDataset", SlowSceneDataset)
    config = replace(recipe("smoke"), synthetic=True, workers=1, loader_timeout=0.05)
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    with pytest.raises(RuntimeError, match="timed out"):
        next(iter(loader))


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


def test_practical_map_rotates_geometry_and_preserves_signal_semantics():
    import math

    config = replace(ModelConfig.smoke(), legacy_lane_columns=False)
    tracks = []
    for actor in range(2):
        states = [
            NS(
                center_x=10.0 + actor,
                center_y=5.0 + (j - 10) * 0.1,
                heading=math.pi / 2,
                velocity_x=0.0,
                velocity_y=1.0,
                length=4.0,
                width=2.0,
                height=1.5,
                valid=j != 0,
            )
            for j in range(91)
        ]
        tracks.append(NS(id=actor + 1, object_type=1, states=states))

    def feature(identifier, kind, value):
        return NS(id=identifier, WhichOneof=lambda _: kind, **{kind: value})

    def points(x):
        return [NS(x=x, y=float(y)) for y in (5, 6, 7)]

    def boundary(identifier):
        return NS(boundary_feature_id=identifier, boundary_type=2)

    lane = NS(
        polyline=points(10),
        speed_limit_mph=22.37,
        type=1,
        interpolating=True,
        exit_lanes=[],
        left_boundaries=[boundary(2)],
        right_boundaries=[boundary(3)],
    )
    scene = NS(
        tracks=tracks,
        current_time_index=10,
        map_features=[
            feature(1, "lane", lane),
            feature(2, "road_line", NS(polyline=points(8.5))),
            feature(3, "road_edge", NS(polyline=points(11.5))),
            feature(4, "stop_sign", NS(lane=[1])),
            feature(
                5,
                "crosswalk",
                NS(
                    polygon=[
                        NS(x=9.0, y=6.0),
                        NS(x=11.0, y=6.0),
                        NS(x=11.0, y=7.0),
                        NS(x=9.0, y=7.0),
                    ]
                ),
            ),
        ],
        dynamic_map_states=[
            NS(lane_states=[NS(lane=1, state=3, stop_point=NS(x=10.0, y=7.0))]) for _ in range(11)
        ],
    )
    sample = scenario_sample(scene, (0, 1), config)
    torch.testing.assert_close(
        sample["ego_state"][-1, :5], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]), atol=1e-6, rtol=0
    )
    assert not sample["actor_valid"][0, 0]
    torch.testing.assert_close(sample["ego_state"][0], torch.zeros(9))
    vectors = sample["map_lanes"][0, 0]
    torch.testing.assert_close(vectors[:, 1], torch.zeros(config.lane_points), atol=1e-6, rtol=0)
    torch.testing.assert_close(vectors[:, 4], torch.full((config.lane_points,), 1.5))
    torch.testing.assert_close(vectors[:, 7], torch.full((config.lane_points,), -1.5))
    torch.testing.assert_close(vectors[:, 9], torch.full((config.lane_points,), 10.0))
    assert (vectors[:, 13] == 3).all() and (vectors[:, 14:17] == 1).all()
    assert sample["lane_valid"][0, 0].all() and sample["crosswalk_valid"].all()


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
