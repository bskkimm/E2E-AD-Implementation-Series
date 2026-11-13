"""Small train/evaluate/prepare/benchmark/tune commands with explicit asset modes."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import RunConfig


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "benchmark", "tune", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/smoke.json")
        p.add_argument("--synthetic", action="store_true")
        p.add_argument("--device")
        p.add_argument("--output", required=True)
        if name == "train":
            p.add_argument("--resume")
            p.add_argument("--tracking-uri")
        elif name == "eval":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--official", action="store_true")
        elif name == "benchmark":
            p.add_argument("--mode", choices=("loader", "step", "end_to_end"), default="end_to_end")
            p.add_argument("--steps", type=int, default=5)
            p.add_argument("--warmup", type=int, default=2)
        elif name == "tune":
            p.add_argument("--max-trials", type=int, default=3)
            p.add_argument("--time-limit", type=float, default=30.0)
            p.add_argument("--steps", type=int, default=3)
        elif name == "prepare":
            p.add_argument("--source", required=True)
            p.add_argument("--split", choices=("train", "validation"), required=True)
            p.add_argument("--max-scenarios", type=int, default=100)
            p.add_argument("--max-records", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        config = RunConfig.load(args.config)
        if args.synthetic:
            config = replace(config, synthetic=True)
        if args.device:
            config = replace(config, device=args.device)
        if args.command == "train":
            from .engine import train

            if args.tracking_uri:
                config = replace(config, tracking_uri=args.tracking_uri)
            result = train(config, args.output, resume=args.resume)
        elif args.command == "prepare":
            from .prepare import prepare_waymo

            if args.synthetic:
                raise ValueError(
                    "Preparation requires actual Scenario files; synthetic is not a replacement"
                )
            result = prepare_waymo(
                args.source,
                args.output,
                config.model,
                args.split,
                args.max_scenarios,
                args.max_records,
            )
        elif args.command in ("benchmark", "tune"):
            from .performance import benchmark, tune

            result = (
                benchmark(config, args.mode, args.steps, args.warmup)
                if args.command == "benchmark"
                else tune(config, args.max_trials, args.time_limit, args.steps)
            )
            write_json(args.output, result)
        else:
            from .engine import build, evaluate, load_checkpoint, make_loaders, move
            from .evaluation import official_metric_inputs, official_metrics

            checkpoint = load_checkpoint(args.checkpoint)
            stored = RunConfig.from_dict(checkpoint["config"])
            if config.model != stored.model or config.task != stored.task:
                raise ValueError("Evaluation config must match checkpoint model/task")
            if args.official and config.synthetic:
                raise ValueError("Official evaluation requires real labeled data")
            raw, model, _, _, _ = build(config)
            raw.load_state_dict(checkpoint["model"])
            _, validation = make_loaders(config, torch.Generator().manual_seed(config.seed))
            metrics, _ = evaluate(model, validation, config)
            if args.official:
                chunks = []
                with torch.no_grad():
                    for batch in validation:
                        batch = move(batch, config.device)
                        chunks.append(
                            {
                                key: value.cpu()
                                for key, value in official_metric_inputs(
                                    model(batch), batch
                                ).items()
                            }
                        )
                metrics.update(
                    official_metrics(
                        {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
                    )
                )
            result = {
                "metrics": metrics,
                "data_kind": "synthetic_diagnostic" if config.synthetic else "subset_validation",
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(
                    json.dumps(config.to_dict(), sort_keys=True).encode()
                ).hexdigest(),
                "seed": config.seed,
                "config": config.to_dict(),
            }
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
