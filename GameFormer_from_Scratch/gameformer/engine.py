"""Plain training, sample-weighted accumulation and atomic epoch-boundary recovery."""

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import RunConfig
from .data import SceneDataset, SyntheticDataset, assert_disjoint
from .evaluation import diagnostic_totals
from .losses import objective
from .model import GameFormer
from .tracking import Tracker, progress_snapshot


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def make_loaders(config: RunConfig, generator: torch.Generator):
    train: Dataset
    validation: Dataset
    if config.synthetic:
        train = SyntheticDataset(config.model, config.train_size, config.seed)
        validation = SyntheticDataset(config.model, config.validation_size, config.seed + 100000)
    else:
        if not config.train_data or not config.validation_data:
            raise ValueError(
                "Real training requires train_data and validation_data; no synthetic fallback"
            )
        train_cache = SceneDataset(config.train_data, config.model, "train")
        validation_cache = SceneDataset(config.validation_data, config.model, "validation")
        assert_disjoint(train_cache, validation_cache)
        train, validation = train_cache, validation_cache
    options: dict[str, Any] = {
        "batch_size": config.microbatch,
        "num_workers": config.workers,
        "pin_memory": config.pin_memory,
    }
    if config.workers:
        options.update(
            prefetch_factor=config.prefetch,
            persistent_workers=config.persistent_workers,
            timeout=config.loader_timeout,
        )
    # Each loader owns a generator; validation must not advance the shuffle RNG.
    return (
        DataLoader(train, shuffle=True, generator=generator, **options),
        DataLoader(
            validation,
            shuffle=False,
            generator=torch.Generator().manual_seed(config.seed + 1),
            **options,
        ),
    )


def build(config: RunConfig):
    seed_everything(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    raw_model = GameFormer(config.model).to(config.device)
    model = torch.compile(raw_model) if config.compile_model else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True if config.fused else None,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, list(config.milestones), gamma=config.gamma
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    return raw_model, model, optimizer, scheduler, scaler


def precision_context(config: RunConfig):
    if config.precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=torch.device(config.device).type,
        dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
    )


def update(model, optimizer, scaler, batches, config: RunConfig):
    """One optimizer update; a partial final group is weighted by actual samples."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    size = sum(batch["ego_state"].shape[0] for batch in batches)
    totals: dict[str, float] = {}
    for batch in batches:
        batch = move(batch, config.device)
        weight = batch["ego_state"].shape[0] / size
        with precision_context(config):
            components = objective(model(batch), batch, config.task)
        scaler.scale(components["loss"] * weight).backward()
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * weight
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.clip_norm, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    return {**totals, "gradient_norm": float(norm), "samples": size}


@torch.no_grad()
def evaluate(model, loader, config: RunConfig):
    model.eval()
    totals = {
        name: 0.0 for name in ("distance_sum", "valid_steps", "final_distance_sum", "valid_agents")
    }
    loss_sum, count = 0.0, 0
    last = None
    for batch in loader:
        batch = move(batch, config.device)
        with precision_context(config):
            output = model(batch)
            values = objective(output, batch, config.task)
        batch_size = batch["ego_state"].shape[0]
        loss_sum += float(values["loss"]) * batch_size
        count += batch_size
        for name, value in diagnostic_totals(output, batch, config.task).items():
            totals[name] += value
        last = batch, output
    metrics = {
        "diagnostic/validation_loss": loss_sum / max(count, 1),
        "diagnostic/valid_steps": totals["valid_steps"],
        "diagnostic/valid_agents": totals["valid_agents"],
    }
    if totals["valid_steps"]:
        metrics["diagnostic/validation_ADE_m"] = totals["distance_sum"] / totals["valid_steps"]
    if totals["valid_agents"]:
        metrics["diagnostic/validation_FDE_m"] = (
            totals["final_distance_sum"] / totals["valid_agents"]
        )
    return metrics, last


def atomic_checkpoint(path: Path, payload: dict) -> float:
    started = time.perf_counter()
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return time.perf_counter() - started


def load_checkpoint(path: str | Path) -> dict:
    # Only load checkpoints produced by this project from a trusted local source.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return checkpoint


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def config_fingerprint(config: RunConfig):
    value = asdict(config)
    for name in (
        "device",
        "workers",
        "loader_timeout",
        "prefetch",
        "pin_memory",
        "persistent_workers",
        "tracking_uri",
        "upload_checkpoint",
    ):
        value.pop(name)
    return value


def environment_record(config: RunConfig):
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()

    manifests = {}
    for split, directory in (("train", config.train_data), ("validation", config.validation_data)):
        path = Path(directory) / "manifest.json"
        if directory and path.is_file():
            manifests[split] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": config.device,
        "gpu": torch.cuda.get_device_name() if config.device.startswith("cuda") else None,
        "cpu_count": os.cpu_count(),
        "code_commit": git("rev-parse", "HEAD"),
        "dirty_state": git("status", "--porcelain", "--", "."),
        "data_manifests": manifests,
        "playbook_commit": "707c64b138d5e6c534af3d197fd4f278aab03d45",
        "official_commit": "fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128",
        "initialization": "random; no pretrained weights",
        "distributed_workers": 1,
    }


def trajectory_figure(batch, outputs, path, task="joint"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .evaluation import decode

    selected, _ = decode(outputs, task)
    figure, axis = plt.subplots(figsize=(6, 4))
    for i in range(min(selected.shape[1], 4)):
        prediction = selected[0, i].detach().float().cpu()
        valid = batch["future_valid"][0, i].cpu()
        truth = batch["future"][0, i].detach().float().cpu()[valid]
        axis.plot(prediction[:, 0], prediction[:, 1], label=f"predicted agent {i}")
        axis.plot(truth[:, 0], truth[:, 1], "--", label=f"target {i}")
    axis.set(
        xlabel="local x (m)",
        ylabel="local y (m)",
        title="Inference-selected trajectory diagnostic",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def train(
    config: RunConfig,
    output: str | Path,
    resume: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> dict:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and (directory / "latest.pt").exists():
        raise ValueError(
            "Output already has a checkpoint; explicitly resume or choose a new output"
        )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader, valid_loader = make_loaders(config, generator)
    raw, model, optimizer, scheduler, scaler = build(config)
    checkpoint = load_checkpoint(resume) if resume else None
    start_epoch, global_step, prior_active, original_started = 0, 0, 0.0, time.time()
    if checkpoint:
        if config_fingerprint(config) != config_fingerprint(
            RunConfig.from_dict(checkpoint["config"])
        ):
            raise ValueError("Resume configuration changes the experiment recipe")
        raw.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start_epoch, global_step = checkpoint["completed_epochs"], checkpoint["global_step"]
        prior_active, original_started = (
            checkpoint["active_seconds"],
            checkpoint["original_started"],
        )
    tracker = Tracker(
        config.tracking_uri,
        directory,
        config.to_dict(),
        checkpoint.get("run_id") if checkpoint else None,
    )
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (directory / "environment.json").write_text(
        json.dumps(environment_record(config), indent=2) + "\n"
    )
    tracker.artifact(directory / "config.json")
    tracker.artifact(directory / "environment.json")
    session_start = time.perf_counter()
    completed = start_epoch
    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.perf_counter()
            group = []
            train_totals: dict[str, float] = {}
            trained = 0
            for i, batch in enumerate(train_loader):
                group.append(batch)
                if len(group) < config.accumulation and i + 1 < len(train_loader):
                    continue
                values = update(model, optimizer, scaler, group, config)
                global_step += 1
                tracker.log({f"train/{name}": value for name, value in values.items()}, global_step)
                for name, value in values.items():
                    if name != "samples":
                        train_totals[name] = train_totals.get(name, 0.0) + value * values["samples"]
                trained += values["samples"]
                group = []
            metrics, last = evaluate(model, valid_loader, config)
            metrics.update(
                {f"train/epoch_{name}": value / trained for name, value in train_totals.items()}
            )
            metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
            if last is not None:
                image_path = directory / f"epoch-{epoch + 1:03d}.png"
                last_batch, last_output = last
                trajectory_figure(last_batch, last_output, image_path, config.task)
                tracker.artifact(image_path)
            scheduler.step()
            completed = epoch + 1
            active = prior_active + time.perf_counter() - session_start
            payload = {
                "schema_version": 1,
                "config": config.to_dict(),
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(generator),
                "completed_epochs": completed,
                "global_step": global_step,
                "active_seconds": active,
                "original_started": original_started,
                "run_id": tracker.run_id,
            }
            metrics["timing/checkpoint_seconds"] = atomic_checkpoint(
                directory / "latest.pt", payload
            )
            metrics["timing/epoch_seconds"] = time.perf_counter() - epoch_start
            tracker.log(metrics, global_step)
            snapshot = progress_snapshot(
                completed,
                config.epochs,
                completed - start_epoch,
                time.perf_counter() - session_start,
                prior_active,
                original_started,
                time.time(),
                config.timezone,
            )
            tracker.progress(snapshot, global_step)
            (directory / "progress.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"epoch": completed, "global_step": global_step, **metrics}) + "\n"
                )
            print(
                f"Epoch {completed}/{config.epochs} loss={metrics['train/epoch_loss']:.4f} remaining={snapshot['remaining_seconds']}s finish={snapshot['finish_time']}",
                flush=True,
            )
            if config.upload_checkpoint:
                tracker.artifact(directory / "latest.pt")
            if stop_after_epoch and completed >= stop_after_epoch:
                break
        tracker.finish("FINISHED" if completed == config.epochs else "KILLED")
    except BaseException:
        tracker.finish("FAILED")
        raise
    result = {
        "completed_epochs": completed,
        "global_step": global_step,
        "run_id": tracker.run_id,
        "checkpoint": str(directory / "latest.pt"),
        "data_kind": "synthetic" if config.synthetic else "real_subset",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
