"""Optional Waymo Scenario reader; practical vector-map preprocessing is explicit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import shapes, validate_sample
from .geometry import world_to_local, wrap_angle


def _points(polyline):
    xy = np.array([(p.x, p.y) for p in polyline], dtype=np.float32)
    if len(xy) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    dxy = np.concatenate((np.diff(xy, axis=0), np.diff(xy, axis=0)[-1:]))
    return np.column_stack((xy, np.arctan2(dxy[:, 1], dxy[:, 0])))


def _resample(points, count):
    return points[np.linspace(0, len(points) - 1, count).round().astype(int)]


def _state(state, kind):
    return [
        state.center_x,
        state.center_y,
        state.heading,
        state.velocity_x,
        state.velocity_y,
        state.length,
        state.width,
        state.height,
        kind if kind in (1, 2, 3) else 0,
    ]


def scenario_sample(scene, pair: tuple[int, ...], c: ModelConfig):
    """Convert labeled world states without zero-coordinate absence heuristics.

    This is the documented practical map recipe, not exact released cache parity.
    Nearest lane seeds, their forward paths and nearby polygon boundaries are
    resampled; optional SDK objects are only accessed through their fields.
    """
    current = scene.current_time_index
    if current < c.history - 1 or len(pair) != c.predicted:
        raise ValueError("Scenario does not match history/predicted-agent configuration")
    sample = {
        name: torch.zeros(shape, dtype=torch.bool if name.endswith("valid") else torch.float32)
        for name, shape in shapes(c).items()
    }
    center_state = scene.tracks[pair[0]].states[current]
    center = torch.tensor([center_state.center_x, center_state.center_y])
    angle = torch.tensor(center_state.heading)
    centers = [
        np.array(
            [scene.tracks[i].states[current].center_x, scene.tracks[i].states[current].center_y]
        )
        for i in pair
    ]
    backgrounds = []
    for i, track in enumerate(scene.tracks):
        if i in pair or not track.states[current].valid:
            continue
        p = np.array([track.states[current].center_x, track.states[current].center_y])
        backgrounds.append((min(np.linalg.norm(p - origin) for origin in centers), i))
    ids = list(pair) + [i for _, i in sorted(backgrounds)[: c.actors - len(pair)]]
    actors = torch.zeros(c.actors, c.history, 9)
    for a, i in enumerate(ids):
        track = scene.tracks[i]
        states = track.states[current - c.history + 1 : current + 1]
        for t, state in enumerate(states):
            if state.valid:
                actors[a, t] = torch.tensor(_state(state, track.object_type))
                sample["actor_valid"][a, t] = True
        if a < c.predicted:
            for t, state in enumerate(track.states[current + 1 : current + 1 + c.future]):
                if state.valid:
                    sample["future_state"][a, t] = torch.tensor(
                        _state(state, track.object_type)[:5]
                    )
                    sample["future_valid"][a, t] = True
    for states, valid in (
        (actors, sample["actor_valid"]),
        (sample["future_state"], sample["future_valid"]),
    ):
        states[..., :2] = world_to_local(states[..., :2], center, angle)
        states[..., 2] = wrap_angle(states[..., 2] - angle)
        states[..., 3:5] = world_to_local(states[..., 3:5], torch.zeros_like(center), angle)
        states.masked_fill_(~valid[..., None], 0)
    sample["ego_state"], sample["neighbors_state"] = actors[0], actors[1:]
    sample["future"] = sample["future_state"][..., :2].clone()
    lanes, roads, crosswalks, stops = {}, {}, [], set()
    for feature in scene.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind == "lane":
            points = _points(feature.lane.polyline)
            if len(points):
                lanes[feature.id] = (feature.lane, points)
        elif kind in ("road_line", "road_edge"):
            roads[feature.id] = _points(getattr(feature, kind).polyline)
        elif kind == "crosswalk":
            polygon = list(feature.crosswalk.polygon)
            if polygon:
                points = _points([*polygon, polygon[0]])
                if len(points):
                    crosswalks.append(points)
        elif kind == "stop_sign":
            stops.update(feature.stop_sign.lane)
    signals = {}
    if current < len(scene.dynamic_map_states):
        signals = {s.lane: s for s in scene.dynamic_map_states[current].lane_states}
    for agent, track_id in enumerate(pair):
        origin = centers[agent]
        ranked = sorted(
            lanes,
            key=lambda key: float(np.linalg.norm(lanes[key][1][:, :2] - origin, axis=-1).min()),
        )
        paths = []
        for seed in ranked:
            lane, points = lanes[seed]
            start = int(np.linalg.norm(points[:, :2] - origin, axis=-1).argmin())
            queue = [([seed], start)]
            while queue and len(paths) < c.lanes:
                path, first = queue.pop(0)
                length = sum(len(lanes[key][1]) for key in path) - first
                exits = [
                    key for key in lanes[path[-1]][0].exit_lanes if key in lanes and key not in path
                ]
                if length >= 300 or not exits:
                    signature = tuple(path)
                    if signature not in [tuple(p[0]) for p in paths]:
                        paths.append((path, first))
                else:
                    queue.extend(([*path, key], first) for key in exits[: c.lanes])
                if len(queue) > 100:
                    queue = queue[:100]
            if len(paths) >= c.lanes:
                break
        for slot, (path, start) in enumerate(paths[: c.lanes]):
            vectors = []
            for k, key in enumerate(path):
                lane, points = lanes[key]
                if k == 0:
                    points = points[start:]
                data = np.zeros((len(points), 17), dtype=np.float32)
                data[:, :3] = points
                data[:, 9], data[:, 10] = lane.speed_limit_mph / 2.237, lane.type
                data[:, 15], data[:, 16] = lane.interpolating, key in stops
                if key in signals:
                    data[:, 13] = signals[key].state
                    stop = signals[key].stop_point
                    data[:, 14] = np.linalg.norm(points[:, :2] - [stop.x, stop.y], axis=-1) < 3
                for boundaries, offset, column in (
                    (lane.left_boundaries, 3, 11),
                    (lane.right_boundaries, 6, 12),
                ):
                    for boundary in boundaries:
                        road = roads.get(boundary.boundary_feature_id)
                        if road is None or not len(road):
                            continue
                        nearest = np.linalg.norm(
                            points[:, None, :2] - road[None, :, :2], axis=-1
                        ).argmin(-1)
                        data[:, offset : offset + 3] = road[nearest]
                        data[:, column] = min(boundary.boundary_type, 10)
                vectors.append(data)
            if vectors:
                sample["map_lanes"][agent, slot] = torch.from_numpy(
                    _resample(np.concatenate(vectors), c.lane_points)
                )
                sample["lane_valid"][agent, slot] = True
        ranked_cw = sorted(
            crosswalks,
            key=lambda points: float(np.linalg.norm(points[:, :2] - origin, axis=-1).min()),
        )
        for slot, points in enumerate(ranked_cw[: c.crosswalks]):
            if np.linalg.norm(points[:, :2] - origin, axis=-1).min() < 50:
                sample["map_crosswalks"][agent, slot] = torch.from_numpy(
                    _resample(points, c.crosswalk_points)
                )
                sample["crosswalk_valid"][agent, slot] = True
    for key, offsets, mask_key in (
        ("map_lanes", (0, 3, 6), "lane_valid"),
        ("map_crosswalks", (0,), "crosswalk_valid"),
    ):
        for offset in offsets:
            sample[key][..., offset : offset + 2] = world_to_local(
                sample[key][..., offset : offset + 2], center, angle
            )
            sample[key][..., offset + 2] = wrap_angle(sample[key][..., offset + 2] - angle)
        sample[key].masked_fill_(~sample[mask_key][..., None], 0)
    validate_sample(sample, c)
    return sample


def prepare_waymo(
    source: str,
    output: str,
    config: ModelConfig,
    split: str,
    max_scenarios: int = 100,
    max_records: int = 200,
):
    if config.predicted != 2 or config.future != 80:
        raise ValueError("Raw Scenario preparation currently supports the joint 80-step task")
    if max_scenarios < 1 or max_records < 1:
        raise ValueError("Preparation limits must be positive")
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as error:
        raise RuntimeError(
            "Raw preparation requires compatible Waymo/TensorFlow SDK; see docs/data.md"
        ) from error
    files = sorted(Path(source).glob("*tfrecord*"))
    if not files:
        raise FileNotFoundError("No Waymo TFRecord shards at the requested source")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists() or list(directory.glob("*.npz")):
        raise ValueError("Preparation output must be empty")
    records, seen = [], 0
    for shard in files:
        for raw in tf.data.TFRecordDataset(str(shard)):
            scene = scenario_pb2.Scenario()
            scene.ParseFromString(raw.numpy())
            seen += 1
            targets = [
                item.track_index
                for item in scene.tracks_to_predict
                if scene.tracks[item.track_index].states[scene.current_time_index].valid
            ]
            interest = [i for i in targets if scene.tracks[i].id in scene.objects_of_interest]
            if split == "validation":
                pairs = [tuple(interest)] if len(interest) == 2 else []
            else:
                pairs = []
                for i in targets:
                    others = [j for j in targets if j != i]
                    others.sort(
                        key=lambda j: (
                            (
                                scene.tracks[j].states[scene.current_time_index].center_x
                                - scene.tracks[i].states[scene.current_time_index].center_x
                            )
                            ** 2
                            + (
                                scene.tracks[j].states[scene.current_time_index].center_y
                                - scene.tracks[i].states[scene.current_time_index].center_y
                            )
                            ** 2
                        )
                    )
                    pairs.extend((i, j) for j in others[:2])
            for pair in pairs:
                sample = scenario_sample(scene, pair, config)
                name = f"scene-{len(records):08d}.npz"
                path = directory / name
                np.savez_compressed(path, **{key: value.numpy() for key, value in sample.items()})
                records.append(
                    {
                        "file": name,
                        "source_id": "womd-v1.1",
                        "scenario_id": scene.scenario_id,
                        "target_track_ids": [scene.tracks[i].id for i in pair],
                        "shard": shard.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                if len(records) >= max_records:
                    break
            if seen >= max_scenarios or len(records) >= max_records:
                break
        if seen >= max_scenarios or len(records) >= max_records:
            break
    if not records:
        raise ValueError(
            "No labeled pairs found; verify interactive validation shards and SDK version"
        )
    manifest = {
        "schema_version": 1,
        "split": split,
        "preprocessing": "practical_vector_map_v1",
        "records": records,
        "scenarios_seen": seen,
        "limit_scenarios": max_scenarios,
        "limit_records": max_records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
