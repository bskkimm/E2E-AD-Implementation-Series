"""Resolved configuration; names distinguish paper, release and diagnostic recipes."""

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    heads: int = 8
    encoder_layers: int = 6
    levels: int = 3
    modes: int = 6
    actors: int = 34
    predicted: int = 2
    history: int = 11
    future: int = 80
    lanes: int = 6
    lane_points: int = 100
    crosswalks: int = 4
    crosswalk_points: int = 50
    lane_stride: int = 10
    crosswalk_stride: int = 10
    dropout: float = 0.1
    partial_segments: bool = False
    legacy_lane_columns: bool = True
    legacy_heading: bool = True
    normalized_future_pool: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.width,
            self.heads,
            self.encoder_layers,
            self.modes,
            self.actors,
            self.predicted,
            self.history,
            self.future,
            self.lanes,
            self.lane_points,
            self.crosswalks,
            self.crosswalk_points,
            self.lane_stride,
            self.crosswalk_stride,
        )
        if any(x < 1 for x in integers) or self.levels < 0:
            raise ValueError("Model dimensions must be positive; levels may be zero")
        if self.width % self.heads or self.width % 2 or self.predicted > self.actors:
            raise ValueError("Width must be even/divisible by heads; predicted <= actors")
        if self.lane_points % self.lane_stride or self.crosswalk_points % self.crosswalk_stride:
            raise ValueError("Map point counts must be divisible by segment strides")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")

    @classmethod
    def smoke(cls) -> "ModelConfig":
        return cls(
            width=32,
            heads=4,
            encoder_layers=1,
            levels=1,
            actors=4,
            lanes=2,
            lane_points=20,
            crosswalks=1,
            crosswalk_points=10,
            dropout=0,
        )


@dataclass(frozen=True)
class RunConfig:
    recipe: str = "released_joint"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: str = "joint"
    seed: int = 3407
    epochs: int = 30
    microbatch: int = 16
    accumulation: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    milestones: tuple[int, ...] = (20, 22, 24, 26, 28)
    gamma: float = 0.5
    clip_norm: float = 5
    device: str = "cpu"
    precision: str = "fp32"
    workers: int = 0
    prefetch: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    fused: bool = False
    compile_model: bool = False
    synthetic: bool = False
    train_size: int = 8
    validation_size: int = 4
    train_data: str = ""
    validation_data: str = ""
    tracking_uri: str = ""
    timezone: str = "Asia/Tokyo"
    upload_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.task not in ("joint", "planning") or self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("Unknown task or precision")
        if self.task == "joint" and (self.model.future != 80 or self.model.predicted != 2):
            raise ValueError("Joint WOMD loss requires two agents and 80 future samples")
        if (
            min(
                self.epochs,
                self.microbatch,
                self.accumulation,
                self.train_size,
                self.validation_size,
                self.prefetch,
            )
            < 1
            or self.workers < 0
        ):
            raise ValueError("Counts must be positive; workers may be zero")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.clip_norm <= 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 < self.gamma <= 1 or tuple(sorted(set(self.milestones))) != self.milestones:
            raise ValueError("Milestones must be unique/increasing and 0 < gamma <= 1")
        if self.precision == "fp16" and not self.device.startswith("cuda"):
            raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")
        ZoneInfo(self.timezone)

    @property
    def effective_batch(self) -> int:
        return self.microbatch * self.accumulation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunConfig":
        value = dict(value)
        value["model"] = ModelConfig(**value.get("model", {}))
        if "milestones" in value:
            value["milestones"] = tuple(value["milestones"])
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def recipe(name: str) -> RunConfig:
    if name == "released_joint":
        return RunConfig()
    if name == "smoke":
        return RunConfig(
            recipe=name,
            model=ModelConfig.smoke(),
            epochs=2,
            microbatch=2,
            learning_rate=1e-3,
            milestones=(1,),
        )
    if name == "practical_joint":
        return RunConfig(
            recipe=name,
            model=replace(
                ModelConfig(),
                legacy_lane_columns=False,
                legacy_heading=False,
                partial_segments=True,
            ),
        )
    if name == "paper_joint_reconstruction":
        return RunConfig(
            recipe=name,
            model=replace(
                ModelConfig(), levels=6, actors=22, crosswalk_points=100, crosswalk_stride=20
            ),
            microbatch=64,
            milestones=(18, 21, 24, 27),
        )
    if name == "released_planning":
        return RunConfig(
            recipe=name,
            task="planning",
            model=replace(ModelConfig(), actors=11, predicted=11, history=20, future=50, levels=4),
            epochs=20,
            microbatch=32,
            milestones=(10, 12, 14, 16, 18),
        )
    raise ValueError(f"Unknown recipe: {name}")
