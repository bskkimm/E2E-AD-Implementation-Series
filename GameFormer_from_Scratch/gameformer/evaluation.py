"""Inference-selected diagnostics and an explicit optional official SDK boundary."""

import torch
from torch import Tensor


def decode(outputs: dict[str, Tensor], task: str = "joint") -> tuple[Tensor, Tensor]:
    trajectories, scores = outputs["trajectories"][-1], outputs["scores"][-1]
    b, p, _, t, _ = trajectories.shape
    if task == "joint":
        probabilities = scores.sum(1).softmax(-1)
        mode = probabilities.argmax(-1)[:, None].expand(-1, p)
    elif task == "planning":
        probabilities = scores.softmax(-1)
        mode = probabilities.argmax(-1)
    else:
        raise ValueError("task must be joint or planning")
    selected = trajectories.gather(2, mode[:, :, None, None, None].expand(b, p, 1, t, 4)).squeeze(2)
    return selected[..., :2], probabilities


def diagnostic_totals(
    outputs: dict[str, Tensor], batch: dict[str, Tensor], task: str = "joint"
) -> dict[str, float]:
    prediction, _ = decode(outputs, task)
    valid = batch["future_valid"]
    target = torch.where(valid[..., None], batch["future"], 0)
    distance = (prediction.float() - target).norm(dim=-1)
    steps = torch.arange(valid.shape[-1], device=valid.device)
    last = torch.where(valid, steps, -1).amax(-1)
    present = last >= 0
    final = distance.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
    return {
        "distance_sum": float(torch.where(valid, distance, 0).sum()),
        "valid_steps": float(valid.sum()),
        "final_distance_sum": float(torch.where(present, final, 0).sum()),
        "valid_agents": float(present.sum()),
    }


def official_metric_inputs(
    outputs: dict[str, Tensor], batch: dict[str, Tensor]
) -> dict[str, Tensor]:
    """WOMD joint layout: B,1,M,2,16,2; actual labeled state/validity required."""
    future = batch["future_state"]
    if future.shape[1:3] != (2, 80):
        raise ValueError("Official joint metrics require two agents and 80 future steps")
    actors = torch.cat((batch["ego_state"][:, None], batch["neighbors_state"]), 1)[:, :2]
    all_state = torch.cat((actors[..., :5], future), 2)
    sizes = actors[:, :, -1, 5:7][:, :, None].expand(-1, -1, all_state.shape[2], -1)
    gt = torch.cat((all_state[..., :2], sizes, all_state[..., 2:]), -1)
    probabilities = outputs["scores"][-1].sum(1).softmax(-1)
    b = future.shape[0]
    return {
        "prediction_trajectory": outputs["trajectories"][-1, ..., 4::5, :2].permute(0, 2, 1, 3, 4)[
            :, None
        ],
        "prediction_score": probabilities[:, None],
        "ground_truth_trajectory": gt,
        "ground_truth_is_valid": torch.cat(
            (batch["actor_valid"][:, :2], batch["future_valid"]), -1
        ),
        "object_type": actors[:, :, -1, 8].long(),
        "prediction_ground_truth_indices": torch.arange(2, device=future.device)[None, None].expand(
            b, 1, -1
        ),
        "prediction_ground_truth_indices_mask": batch["future_valid"].any(-1)[:, None],
    }


def official_metrics(inputs: dict[str, Tensor]) -> dict[str, float]:
    try:
        import tensorflow as tf
        from google.protobuf import text_format
        from waymo_open_dataset.metrics.ops import py_metrics_ops
        from waymo_open_dataset.metrics.python import config_util_py
        from waymo_open_dataset.protos import motion_metrics_pb2
    except ImportError as error:
        raise RuntimeError(
            "Official evaluation requires the compatible Waymo/TensorFlow SDK; see docs/data.md"
        ) from error
    config = motion_metrics_pb2.MotionMetricsConfig()
    text_format.Parse(
        """track_steps_per_second:10 prediction_steps_per_second:2
    track_history_samples:10 track_future_samples:80 speed_lower_bound:1.4
    speed_upper_bound:11.0 speed_scale_lower:0.5 speed_scale_upper:1.0
    step_configurations {measurement_step:5 lateral_miss_threshold:1.0 longitudinal_miss_threshold:2.0}
    step_configurations {measurement_step:9 lateral_miss_threshold:1.8 longitudinal_miss_threshold:3.6}
    step_configurations {measurement_step:15 lateral_miss_threshold:3.0 longitudinal_miss_threshold:6.0}
    max_predictions:6""",
        config,
    )
    values = py_metrics_ops.motion_metrics(
        config=config.SerializeToString(),
        **{
            name: tf.convert_to_tensor(value.detach().cpu().numpy())
            for name, value in inputs.items()
        },
    )
    breakdowns = config_util_py.get_breakdown_names_from_motion_config(config)
    return {
        f"official/{metric}/{name}": float(values[i][j].numpy())
        for i, metric in enumerate(("minADE", "minFDE", "miss_rate", "overlap_rate", "mAP"))
        for j, name in enumerate(breakdowns)
    }
