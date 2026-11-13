"""Local optional MLflow and pure, timezone-aware progress arithmetic."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def progress_snapshot(
    completed: int,
    total: int,
    session_completed: int,
    session_seconds: float,
    prior_active: float,
    original_started: float,
    now: float,
    timezone: str = "Asia/Tokyo",
) -> dict[str, Any]:
    if total < 1 or not 0 <= completed <= total or session_seconds < 0:
        raise ValueError("Invalid progress counters")
    rate = session_completed / session_seconds if session_completed and session_seconds else None
    remaining = (total - completed) / rate if rate else (0.0 if completed == total else None)
    return {
        "percent": completed / total * 100,
        "completed_epochs": completed,
        "active_seconds": prior_active + session_seconds,
        "wall_seconds": max(0.0, now - original_started),
        "session_seconds": session_seconds,
        "remaining_seconds": remaining,
        "finish_time": datetime.fromtimestamp(now + remaining, ZoneInfo(timezone)).isoformat()
        if remaining is not None
        else None,
        "updated_time": datetime.fromtimestamp(now, ZoneInfo(timezone)).isoformat(),
        "epochs_per_second": rate,
        "provisional": session_completed < 2,
    }


class Tracker:
    def __init__(self, uri: str, directory: Path, config: dict[str, Any], run_id: str | None):
        self.client: Any = None
        self.run_id = run_id
        if not uri:
            return
        try:
            from mlflow import MlflowClient
        except ImportError as error:
            raise RuntimeError("Tracking requested: install the tracking extra") from error
        self.client = MlflowClient(tracking_uri=uri)
        experiment = self.client.get_experiment_by_name("GameFormer")
        experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment(
                "GameFormer",
                artifact_location=(directory.parent / "mlartifacts").resolve().as_uri(),
            )
        )
        if run_id:
            self.client.get_run(run_id)
            self.client.update_run(run_id, status="RUNNING")
        else:
            run = self.client.create_run(
                experiment_id,
                tags={
                    "mlflow.runName": config["recipe"],
                    "data_kind": "synthetic_diagnostic" if config["synthetic"] else "real_subset",
                    "initialization": "random; no pretrained weights",
                },
            )
            self.run_id = run.info.run_id
            self.client.log_param(
                self.run_id, "resolved_config", json.dumps(config, sort_keys=True)
            )
        # MLflow parameters are immutable. Runtime loader/device settings may change
        # at resume, so preserve the initial parameter and expose the current one.
        self.client.set_tag(
            self.run_id, "current_resolved_config", json.dumps(config, sort_keys=True)
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.client:
            for name, value in metrics.items():
                self.client.log_metric(self.run_id, name, float(value), step=step)

    def progress(self, snapshot: dict[str, Any], step: int) -> None:
        if self.client:
            summary = (
                f"Progress: {snapshot['percent']:.1f}%\n\n"
                f"Active elapsed: {snapshot['active_seconds']:.2f}s; wall elapsed: {snapshot['wall_seconds']:.2f}s\n\n"
                f"Remaining: {snapshot['remaining_seconds']}s\n\n"
                f"Expected finish: {snapshot['finish_time']}\n\n"
                f"Updated: {snapshot['updated_time']}; provisional: {snapshot['provisional']}\n\n"
                "Measured session rate includes observed validation/checkpoint overhead; resumed downtime is excluded."
            )
            self.client.set_tag(self.run_id, "mlflow.note.content", summary)
            self.log(
                {
                    f"progress/{key}": float(snapshot[key])
                    for key in ("percent", "active_seconds", "wall_seconds", "session_seconds")
                },
                step,
            )
            if snapshot["remaining_seconds"] is not None:
                self.log({"progress/remaining_seconds": snapshot["remaining_seconds"]}, step)

    def artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.client:
            self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    def finish(self, status: str = "FINISHED") -> None:
        if self.client:
            self.client.set_terminated(self.run_id, status=status)
