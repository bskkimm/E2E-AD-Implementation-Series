import hashlib
import json

import numpy as np
import pytest
import torch

from gameformer.config import ModelConfig
from gameformer.data import SceneDataset, assert_disjoint, synthetic_sample
from gameformer.evaluation import decode, diagnostic_totals, official_metric_inputs


def test_cache_checksums_and_split_leakage(tmp_path):
    config = ModelConfig.smoke()
    sample = synthetic_sample(config, 0)
    path = tmp_path / "scene.npz"
    np.savez(path, **{k: v.numpy() for k, v in sample.items()})
    record = {
        "file": "scene.npz",
        "source_id": "log-a",
        "scenario_id": "scenario-a",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "split": "train", "records": [record]})
    )
    data = SceneDataset(tmp_path, config, "train")
    torch.testing.assert_close(data[0]["future"], sample["future"])
    with pytest.raises(ValueError, match="same source scenario"):
        assert_disjoint(data, data)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        data[0]


def test_decoding_never_uses_ground_truth_and_sdk_axes_match():
    config = ModelConfig.smoke()
    batch = {k: v[None] for k, v in synthetic_sample(config, 0).items()}
    predictions = torch.zeros(1, 1, 2, 6, 80, 4)
    scores = torch.zeros(1, 1, 2, 6)
    scores[..., 3] = 10
    predictions[:, :, :, 3, :, 0] = 9
    output = {"trajectories": predictions, "scores": scores}
    decoded, _ = decode(output)
    assert (decoded[..., 0] == 9).all()
    sdk = official_metric_inputs(output, batch)
    assert sdk["prediction_trajectory"].shape == (1, 1, 6, 2, 16, 2)
    assert sdk["ground_truth_trajectory"].shape == (1, 2, 91, 7)
    assert sdk["prediction_ground_truth_indices"].tolist() == [[[0, 1]]]
    batch["future_valid"].zero_()
    totals = diagnostic_totals(output, batch)
    assert totals["valid_steps"] == totals["valid_agents"] == 0


def test_planning_selects_a_mode_for_each_agent():
    trajectories = torch.zeros(1, 1, 2, 2, 5, 4)
    trajectories[..., 0, :, 0] = 1
    trajectories[..., 1, :, 0] = 2
    scores = torch.tensor([[[[4.0, 0.0], [0.0, 8.0]]]])
    selected, probabilities = decode({"trajectories": trajectories, "scores": scores}, "planning")
    assert selected[0, :, 0, 0].tolist() == [1.0, 2.0]
    assert probabilities.shape == (1, 2, 2)
