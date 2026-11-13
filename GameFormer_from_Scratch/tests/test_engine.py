"""Resume equivalence, measured progress and genuine local tracking side effects."""

from dataclasses import replace

import pytest
import torch

from gameformer.config import recipe
from gameformer.engine import load_checkpoint, train
from gameformer.tracking import progress_snapshot


def test_eta_uses_remaining_work_and_new_session_rate():
    snapshot = progress_snapshot(
        completed=6,
        total=10,
        session_completed=2,
        session_seconds=20.0,
        prior_active=40.0,
        original_started=900.0,
        now=1000.0,
        timezone="Asia/Tokyo",
    )
    assert snapshot["remaining_seconds"] == 40
    assert snapshot["active_seconds"] == 60
    assert snapshot["wall_seconds"] == 100
    assert snapshot["percent"] == 60
    assert snapshot["finish_time"] == "1970-01-01T09:17:20+09:00"
    cold = progress_snapshot(6, 10, 0, 0.0, 40.0, 900.0, 1000.0)
    assert cold["remaining_seconds"] is None and cold["provisional"]


def test_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    config = replace(
        recipe("smoke"), synthetic=True, train_size=5, validation_size=2, accumulation=2
    )
    full = train(config, tmp_path / "full")
    first = train(config, tmp_path / "resume", stop_after_epoch=1)
    assert first["completed_epochs"] == 1
    resumed = train(config, tmp_path / "resume", resume=tmp_path / "resume/latest.pt")
    assert resumed["global_step"] == full["global_step"] == 4
    a, b = (
        load_checkpoint(tmp_path / "full/latest.pt"),
        load_checkpoint(tmp_path / "resume/latest.pt"),
    )
    assert a["scheduler"] == b["scheduler"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="configuration"):
        train(
            replace(config, learning_rate=0.2),
            tmp_path / "resume",
            resume=tmp_path / "resume/latest.pt",
        )


def test_mlflow_writes_metrics_and_progress_summary(tmp_path):
    from mlflow import MlflowClient

    uri = "sqlite:///" + str(tmp_path / "tracking.db")
    config = replace(
        recipe("smoke"), synthetic=True, epochs=1, train_size=2, validation_size=2, tracking_uri=uri
    )
    result = train(config, tmp_path / "run")
    run = MlflowClient(tracking_uri=uri).get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert "diagnostic/validation_ADE_m" in run.data.metrics
    assert "Expected finish:" in run.data.tags["mlflow.note.content"]
    assert run.data.metrics["progress/percent"] == 100
    assert (tmp_path / "run/epoch-001.png").is_file()
