"""Resume equivalence, measured progress and genuine local tracking side effects."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from gameformer.config import recipe
from gameformer.data import synthetic_sample
from gameformer.engine import build, load_checkpoint, make_loaders, move, train, update
from gameformer.losses import objective
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


@pytest.mark.parametrize("workers", [0, 1])
def test_epoch_boundary_resume_matches_uninterrupted(tmp_path, workers):
    config = replace(
        recipe("smoke"),
        synthetic=True,
        train_size=5,
        validation_size=2,
        accumulation=2,
        workers=workers,
        persistent_workers=bool(workers),
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


def test_tracking_resume_records_permitted_runtime_changes(tmp_path):
    from mlflow import MlflowClient

    uri = "sqlite:///" + str(tmp_path / "tracking.db")
    config = replace(recipe("smoke"), synthetic=True, train_size=2, tracking_uri=uri)
    first = train(config, tmp_path / "run", stop_after_epoch=1)
    initial_config = (tmp_path / "run/config.json").read_bytes()
    initial_environment = (tmp_path / "run/environment.json").read_bytes()
    resumed = train(
        replace(config, prefetch=3), tmp_path / "run", resume=tmp_path / "run/latest.pt"
    )
    assert resumed["run_id"] == first["run_id"]
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(resumed["run_id"])
    assert run.info.status == "FINISHED"
    # The original immutable parameter and the current runtime provenance coexist.
    assert json.loads(run.data.params["resolved_config"])["prefetch"] == 2
    assert json.loads(run.data.tags["current_resolved_config"])["prefetch"] == 3
    assert (tmp_path / "run/config.json").read_bytes() == initial_config
    assert (tmp_path / "run/environment.json").read_bytes() == initial_environment
    snapshots = list((tmp_path / "run/sessions").glob("*/config.json"))
    assert sorted(json.loads(path.read_text())["prefetch"] for path in snapshots) == [2, 3]
    sessions = client.list_artifacts(resumed["run_id"], "sessions")
    assert len(sessions) == 2
    for session in sessions:
        assert {
            item.path.rsplit("/", 1)[-1]
            for item in client.list_artifacts(resumed["run_id"], session.path)
        } == {"config.json", "environment.json"}


def test_accumulation_matches_concatenation_for_uniform_validity():
    config = replace(recipe("smoke"), synthetic=True, train_size=3, microbatch=2)
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    group = list(loader)  # A partial microbatch contributes one of the three samples.
    _, accumulated, optimizer_a, _, scaler_a = build(config)
    _, concatenated, optimizer_b, _, scaler_b = build(config)
    whole = {key: torch.cat([b[key] for b in group]) for key in group[0]}
    values_a = update(accumulated, optimizer_a, scaler_a, group, config)
    values_b = update(concatenated, optimizer_b, scaler_b, [whole], config)
    assert values_a["loss"] == pytest.approx(values_b["loss"], rel=1e-5)
    for a, b in zip(accumulated.parameters(), concatenated.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


def test_tiny_batch_learns_with_actual_joint_objective():
    config = replace(recipe("smoke"), synthetic=True, train_size=2)
    loader, _ = make_loaders(config, torch.Generator().manual_seed(config.seed))
    group = [next(iter(loader))]
    _, model, optimizer, _, scaler = build(config)
    inputs = move(group[0], config.device)
    model.eval()
    with torch.no_grad():
        before = float(objective(model(inputs), inputs, "joint")["loss"])
    for _ in range(40):
        update(model, optimizer, scaler, group, config)
    model.eval()
    with torch.no_grad():
        after = float(objective(model(inputs), inputs, "joint")["loss"])
    assert after < before * 0.25, (before, after)


@pytest.mark.parametrize("changed_split", ["train", "validation"])
def test_resume_rejects_changed_cache_manifest_before_overwriting_provenance(
    tmp_path, changed_split
):
    config = recipe("smoke")
    for split in ("train", "validation"):
        directory = tmp_path / split
        directory.mkdir()
        sample = synthetic_sample(config.model, 0)
        path = directory / "scene.npz"
        np.savez(path, **{key: value.numpy() for key, value in sample.items()})
        manifest = {
            "schema_version": 1,
            "split": split,
            "records": [
                {
                    "file": "scene.npz",
                    "source_id": "fixture",
                    "scenario_id": split,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    config = replace(
        config, train_data=str(tmp_path / "train"), validation_data=str(tmp_path / "validation")
    )
    output = tmp_path / "run"
    train(config, output, stop_after_epoch=1)
    original_environment = (output / "environment.json").read_bytes()
    manifest_path = tmp_path / changed_split / "manifest.json"
    changed = json.loads(manifest_path.read_text())
    changed["records"][0]["scenario_id"] = "another-scenario"
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="dataset|manifest"):
        train(config, output, resume=output / "latest.pt")
    assert (output / "environment.json").read_bytes() == original_environment
