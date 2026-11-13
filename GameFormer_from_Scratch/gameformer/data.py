"""Explicit scene contracts, deterministic diagnostics and safe processed NPZ caches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import ModelConfig


def shapes(c: ModelConfig) -> dict[str, tuple[int, ...]]:
    return {
        "ego_state": (c.history, 9),
        "neighbors_state": (c.actors - 1, c.history, 9),
        "actor_valid": (c.actors, c.history),
        "map_lanes": (c.predicted, c.lanes, c.lane_points, 17),
        "lane_valid": (c.predicted, c.lanes, c.lane_points),
        "map_crosswalks": (c.predicted, c.crosswalks, c.crosswalk_points, 3),
        "crosswalk_valid": (c.predicted, c.crosswalks, c.crosswalk_points),
        "future": (c.predicted, c.future, 2),
        "future_state": (c.predicted, c.future, 5),
        "future_valid": (c.predicted, c.future),
    }


def validate_sample(sample: dict[str, Tensor], config: ModelConfig) -> None:
    for name, shape in shapes(config).items():
        if name not in sample or tuple(sample[name].shape) != shape:
            raise ValueError(f"{name}: expected {shape}; explicit masks and future_state required")
        if name.endswith("valid") and sample[name].dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    actors = torch.cat((sample["ego_state"][None], sample["neighbors_state"]))
    for values, valid in (
        (actors, sample["actor_valid"]),
        (sample["map_lanes"], sample["lane_valid"]),
        (sample["map_crosswalks"], sample["crosswalk_valid"]),
        (sample["future_state"], sample["future_valid"]),
        (sample["future"], sample["future_valid"]),
    ):
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Non-finite value in a valid scene entry")
    if not sample["actor_valid"][0, -1]:
        raise ValueError("Focal actor must be valid at the current time")
    ids = actors[..., 8][sample["actor_valid"]]
    if ((ids < 0) | (ids > 3) | (ids != ids.round())).any():
        raise ValueError("Agent type IDs must be integers in [0,3]")


def synthetic_sample(c: ModelConfig, index: int, seed: int = 3407) -> dict[str, Tensor]:
    """Illustrative constant-velocity scenes; this is not a WOMD sample."""
    g = torch.Generator().manual_seed(seed + index)
    actors = torch.zeros(c.actors, c.history, 9)
    positions = torch.randn(c.actors, 2, generator=g) * 4
    positions[0] = 0
    velocity = torch.randn(c.actors, 2, generator=g) * 0.2
    velocity[:, 0] += 2
    times = torch.arange(1 - c.history, 1) * 0.1
    actors[..., :2] = positions[:, None] + velocity[:, None] * times[None, :, None]
    actors[..., 2] = torch.atan2(velocity[:, 1], velocity[:, 0])[:, None]
    actors[..., 3:5] = velocity[:, None]
    actors[..., 5:8] = torch.tensor([4.0, 2.0, 1.5])
    actors[..., 8] = torch.arange(c.actors).remainder(3)[:, None] + 1
    lanes = torch.zeros(c.predicted, c.lanes, c.lane_points, 17)
    x = torch.linspace(-5, 40, c.lane_points)
    for lane in range(c.lanes):
        for offset, y in ((0, lane * 3.0), (3, lane * 3.0 + 1.5), (6, lane * 3.0 - 1.5)):
            lanes[:, lane, :, offset] = x
            lanes[:, lane, :, offset + 1] = y
    lanes[..., 9] = 13.4
    lanes[..., 10:13] = 1
    crosswalks = torch.randn(c.predicted, c.crosswalks, c.crosswalk_points, 3, generator=g)
    future_times = torch.arange(1, c.future + 1) * 0.1
    future = (
        positions[: c.predicted, None] + velocity[: c.predicted, None] * future_times[None, :, None]
    )
    state = torch.cat((future, actors[: c.predicted, -1, None, 2:5].expand(-1, c.future, -1)), -1)
    sample = {
        "ego_state": actors[0],
        "neighbors_state": actors[1:],
        "actor_valid": torch.ones(c.actors, c.history, dtype=torch.bool),
        "map_lanes": lanes,
        "lane_valid": torch.ones(lanes.shape[:-1], dtype=torch.bool),
        "map_crosswalks": crosswalks,
        "crosswalk_valid": torch.ones(crosswalks.shape[:-1], dtype=torch.bool),
        "future": future,
        "future_state": state,
        "future_valid": torch.ones(c.predicted, c.future, dtype=torch.bool),
    }
    validate_sample(sample, c)
    return sample


class SyntheticDataset(Dataset):
    def __init__(self, config: ModelConfig, size: int, seed: int):
        self.config, self.size, self.seed = config, size, seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return synthetic_sample(self.config, index, self.seed)


class SceneDataset(Dataset):
    """Versioned cache with explicit validity and source/split provenance."""

    def __init__(self, directory: str | Path, config: ModelConfig, expected_split: str = ""):
        self.directory, self.config = Path(directory), config
        path = self.directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache manifest: {path}; see docs/data.md")
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("schema_version") != 1 or not self.manifest.get("records"):
            raise ValueError("Cache needs schema_version=1 and nonempty records")
        if expected_split and self.manifest.get("split") != expected_split:
            raise ValueError(f"Cache split is not {expected_split}")
        self.records = self.manifest["records"]
        for record in self.records:
            if not all(record.get(key) for key in ("file", "scenario_id", "source_id", "sha256")):
                raise ValueError("Each cache record needs file/scenario_id/source_id/sha256")
            file = (self.directory / record["file"]).resolve()
            if not file.is_relative_to(self.directory.resolve()) or file.suffix != ".npz":
                raise ValueError("Cache file must be a local NPZ within its directory")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.directory / record["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            sample = {name: torch.from_numpy(archive[name].copy()) for name in shapes(self.config)}
        validate_sample(sample, self.config)
        return sample


def assert_disjoint(train: SceneDataset, validation: SceneDataset) -> None:
    """Compare source scenario identity before selecting individual interaction pairs."""
    identities = [
        {(r["source_id"], r["scenario_id"]) for r in d.records} for d in (train, validation)
    ]
    if identities[0] & identities[1]:
        raise ValueError("Training and validation contain the same source scenario")
