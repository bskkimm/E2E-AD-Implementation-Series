"""Analytical fixtures: tests fail on changes to units, masks or joint selection."""

import torch

from gameformer.config import ModelConfig
from gameformer.data import synthetic_sample
from gameformer.geometry import local_to_world, world_to_local
from gameformer.losses import interaction_potential, objective
from gameformer.model import GameFormer, segment_points


def batch(config, size=2):
    samples = [synthetic_sample(config, i, 7) for i in range(size)]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def test_geometry_retains_valid_origin_and_rotates_velocity():
    xy = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    origin = torch.tensor([0.0, 1.0])
    angle = torch.tensor(torch.pi / 2)
    local = world_to_local(xy, origin, angle)
    torch.testing.assert_close(local, torch.tensor([[0.0, 0.0], [0.0, -1.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(local_to_world(local, origin, angle), xy)


def test_partial_and_empty_pooling_dont_admit_padding():
    points = torch.tensor([[[[1.0, 2.0], [99.0, 99.0], [5.0, 6.0], [7.0, 8.0]]]])
    valid = torch.tensor([[[True, False, False, False]]])
    features, mask = segment_points(points, valid, 2, partial=True)
    torch.testing.assert_close(features, torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
    assert mask.tolist() == [[False, True]]
    _, strict_mask = segment_points(points, valid, 2, partial=False)
    assert strict_mask.all()


def test_joint_winner_and_gaussian_nll_are_analytical():
    target = torch.zeros(1, 2, 80, 2)
    # mode zero has agent distances 0 and 10; mode one has distances 2 and 2.
    pred = torch.zeros(1, 1, 2, 2, 80, 4)
    pred[0, 0, 1, 0, :, 0] = 10
    pred[0, 0, :, 1, :, 0] = 2
    scores = torch.zeros(1, 1, 2, 2)
    values = objective(
        {"trajectories": pred, "scores": scores},
        {"future": target, "future_valid": torch.ones(1, 2, 80, dtype=torch.bool)},
        "joint",
    )
    # Per valid agent: NLL=2, temporal mean + 3 horizons gives 8.
    torch.testing.assert_close(values["regression"], torch.tensor(8.0))
    torch.testing.assert_close(values["classification"], 2 * torch.log(torch.tensor(2.0)))


def test_missing_targets_have_zero_finite_gradients():
    pred = torch.randn(2, 1, 2, 6, 80, 4, requires_grad=True)
    scores = torch.randn(2, 1, 2, 6, requires_grad=True)
    values = objective(
        {"trajectories": pred, "scores": scores},
        {
            "future": torch.full((1, 2, 80, 2), float("nan")),
            "future_valid": torch.zeros(1, 2, 80, dtype=torch.bool),
        },
        "joint",
    )
    assert values["loss"].item() == 0
    values["loss"].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.count_nonzero() == 0
    assert torch.isfinite(scores.grad).all() and scores.grad.count_nonzero() == 0


def test_cross_level_potential_ignores_invalid_neighbors():
    current = torch.zeros(1, 2, 1, 1, 2, requires_grad=True)
    previous = torch.zeros_like(current)
    previous[:, 1, :, :, 0] = 1
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    torch.testing.assert_close(interaction_potential(current, previous, valid), torch.tensor(0.5))
    valid[:, 1] = False
    assert interaction_potential(current, previous, valid).item() == 0


def test_model_has_every_level_and_finite_backward_with_empty_map():
    torch.manual_seed(3)
    config = ModelConfig.smoke()
    inputs = batch(config)
    inputs["lane_valid"].zero_()
    inputs["crosswalk_valid"].zero_()
    model = GameFormer(config)
    output = model(inputs)
    assert output["trajectories"].shape == (2, 2, 2, 6, 80, 4)
    assert output["scores"].shape == (2, 2, 2, 6)
    objective(output, inputs, "joint")["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.future_encoder.mlp[0].weight.grad.abs().sum() > 0


def test_padding_cannot_change_valid_predictions():
    torch.manual_seed(4)
    config = ModelConfig.smoke()
    inputs = batch(config, 1)
    inputs["actor_valid"][:, -1] = False
    inputs["lane_valid"][:, :, -1] = False
    model = GameFormer(config).eval()
    with torch.no_grad():
        before = model(inputs)["trajectories"]
        inputs["neighbors_state"][:, -1] = 999
        inputs["map_lanes"][:, :, -1] = 999
        after = model(inputs)["trajectories"]
    torch.testing.assert_close(before, after)
