"""Bounded measurements; cached-step throughput is never called end-to-end."""

import gc
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from .config import RunConfig
from .engine import build, environment_record, make_loaders, move, precision_context, update
from .losses import objective


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def hardware_limits():
    """Visible hardware is not necessarily the allocation granted to this process."""
    result = {"logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads()}
    for name, path in (
        ("cpu_quota", "/sys/fs/cgroup/cpu.max"),
        ("memory_limit_bytes", "/sys/fs/cgroup/memory.max"),
    ):
        file = Path(path)
        result[name] = file.read_text().strip() if file.is_file() else None
    result["host_ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return result


def grouped_batches(loader, accumulation):
    while True:
        group = []
        for batch in loader:
            group.append(batch)
            if len(group) == accumulation:
                yield group
                group = []
        if group:
            yield group


def benchmark(
    config: RunConfig,
    mode: str = "end_to_end",
    steps: int = 5,
    warmup: int = 2,
    max_seconds: float | None = None,
) -> dict:
    if mode not in ("loader", "step", "end_to_end") or steps < 1 or warmup < 0:
        raise ValueError("Invalid benchmark scope/count")
    start = time.perf_counter()
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    iterator = grouped_batches(loader, config.accumulation)
    if mode != "loader":
        _, model, optimizer, _, scaler = build(config)
    cached = next(iterator) if mode == "step" else None
    for _ in range(warmup):
        group = cached if cached is not None else next(iterator)
        if mode != "loader":
            update(model, optimizer, scaler, group, config)
    synchronize(config.device)
    startup = time.perf_counter() - start
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
    start = time.perf_counter()
    count, updates, wait, completed = 0, 0, 0.0, 0
    for _ in range(steps):
        before = time.perf_counter()
        group = cached if cached is not None else next(iterator)
        wait += time.perf_counter() - before if cached is None else 0
        count += sum(b["ego_state"].shape[0] for b in group)
        if mode != "loader":
            update(model, optimizer, scaler, group, config)
            updates += 1
        synchronize(config.device)
        completed += 1
        if max_seconds is not None and time.perf_counter() - start >= max_seconds:
            break
    elapsed = time.perf_counter() - start
    return {
        "scope": mode,
        "samples": count,
        "optimizer_updates": updates,
        "measured_groups": completed,
        "warmup_groups": warmup,
        "seconds": elapsed,
        "startup_seconds": startup,
        "samples_per_second": count / elapsed,
        "group_seconds": elapsed / completed,
        "loader_wait_seconds": wait,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(config.device)
        if config.device.startswith("cuda")
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(config.device)
        if config.device.startswith("cuda")
        else None,
        "cache_conditions": "synthetic generation"
        if config.synthetic
        else "filesystem cache uncontrolled; NPZ read/checksum included",
        "config": config.to_dict(),
        "environment": environment_record(config),
        "cpu_affinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "hardware_limits": hardware_limits(),
    }


def correctness_probe(config: RunConfig):
    _, model, optimizer, _, scaler = build(config)
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    group = next(grouped_batches(loader, config.accumulation))
    batch = move({key: torch.cat([item[key] for item in group]) for key in group[0]}, config.device)
    model.eval()
    with precision_context(config):
        outputs = model(batch)
        loss = objective(outputs, batch, config.task)["loss"]
    loss.backward()
    gradient = torch.cat(
        [p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None]
    )
    result = {
        "output": outputs["trajectories"].detach().float().cpu(),
        "scores": outputs["scores"].detach().float().cpu(),
        "loss": loss.detach().float().cpu(),
        "gradient": gradient,
    }
    # Same real optimizer updates as training; compare two-step learning as well.
    result["learning"] = torch.tensor(
        [update(model, optimizer, scaler, group, config)["loss"] for _ in range(2)]
    )
    return result


def tune(config: RunConfig, max_trials: int = 3, time_limit: float = 30.0, steps: int = 3) -> dict:
    if max_trials < 1 or time_limit <= 0 or steps < 1:
        raise ValueError("Tuning needs positive trial/time/step limits")
    started = time.perf_counter()
    baseline = replace(config, precision="fp32", fused=False, compile_model=False)
    reference = correctness_probe(baseline)
    candidates = [
        baseline,
        replace(baseline, workers=2, persistent_workers=True),
        replace(baseline, precision="bf16"),
    ]
    if baseline.microbatch % 2 == 0:
        candidates.append(
            replace(
                baseline,
                microbatch=baseline.microbatch // 2,
                accumulation=baseline.accumulation * 2,
            )
        )
    if baseline.accumulation % 2 == 0:
        candidates.append(
            replace(
                baseline,
                microbatch=baseline.microbatch * 2,
                accumulation=baseline.accumulation // 2,
            )
        )
    if baseline.device.startswith("cuda"):
        candidates.extend(
            (
                replace(baseline, pin_memory=True, workers=2, persistent_workers=True, prefetch=4),
                replace(baseline, fused=True),
            )
        )
    candidates.append(replace(baseline, compile_model=True))
    trials, selected, best_upper = [], baseline, 0.0
    for candidate in candidates[:max_trials]:
        if trials and time.perf_counter() - started >= time_limit:
            break
        if candidate.workers:
            candidate = replace(
                candidate,
                loader_timeout=min(
                    candidate.loader_timeout,
                    max(0.05, time_limit - (time.perf_counter() - started)),
                ),
            )
        trial = {"config": candidate.to_dict()}
        try:
            probe = reference if candidate == baseline else correctness_probe(candidate)
            for name in ("output", "scores", "loss", "gradient", "learning"):
                if not torch.isfinite(probe[name]).all():
                    raise ValueError(f"Nonfinite {name}")
                torch.testing.assert_close(
                    probe[name],
                    reference[name],
                    atol=1e-4 if candidate.precision == "fp32" else 0.03,
                    rtol=1e-4 if candidate.precision == "fp32" else 0.01,
                )
            measurements = [
                benchmark(
                    candidate,
                    steps=steps,
                    warmup=1,
                    max_seconds=max(0.01, time_limit - (time.perf_counter() - started)),
                )
                for _ in range(2)
            ]
            speed = statistics.median(item["samples_per_second"] for item in measurements)
            trial.update(
                status="accepted", measurements=measurements, median_samples_per_second=speed
            )
            memory_ok = True
            if candidate.device.startswith("cuda"):
                total = torch.cuda.get_device_properties(candidate.device).total_memory
                memory_ok = max(item["peak_reserved_bytes"] for item in measurements) < total * 0.9
            # Require >3% measured improvement and memory headroom, or retain baseline.
            lower = min(item["samples_per_second"] for item in measurements)
            upper = max(item["samples_per_second"] for item in measurements)
            if memory_ok and (not best_upper or lower > best_upper * 1.03):
                selected, best_upper = candidate, upper
            trial["memory_headroom_ok"] = memory_ok
        except (AssertionError, ValueError, RuntimeError) as error:
            trial.update(status="rejected", reason=f"{type(error).__name__}: {error}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        trials.append(trial)
    return {
        "selected": selected.to_dict(),
        "trials": trials,
        "selection_claim": "best measured among tested candidates",
        "selection_rule": "correctness + two repeats + candidate worst > incumbent best by 3% + >=10% GPU memory headroom; otherwise FP32 baseline",
        "elapsed_seconds": time.perf_counter() - started,
        "time_limit_seconds": time_limit,
        "budget_semantics": "deadline checked between candidates and timed steps; initialization/probes and one atomic step may overrun",
        "max_trials": max_trials,
        "hardware_limits": hardware_limits(),
        "candidate_count_available": len(candidates),
    }
