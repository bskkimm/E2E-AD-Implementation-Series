"""Opt-in current-source compatibility test; requires the exact pinned clone."""

import argparse
import ast
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from gameformer.config import ModelConfig
from gameformer.data import synthetic_sample
from gameformer.losses import objective
from gameformer.model import GameFormer

REVISION = "fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128"


def upstream_name(name):
    for local, reference in (
        ("ego_encoder.", "encoder.ego_encoder."),
        ("agent_encoder.", "encoder.agent_encoder."),
        ("scene_encoder.", "encoder.fusion_encoder."),
        ("mode_embedding.", "decoder.initial_stage.multi_modal_query_embedding."),
        ("agent_embedding.", "decoder.initial_stage.agent_query_embedding."),
        ("initial_cross.", "decoder.initial_stage.query_encoder."),
        ("initial_head.", "decoder.initial_stage.predictor."),
        ("future_encoder.", "decoder.interaction_stage.0.future_encoder."),
        ("crosswalk_encoder.pointnet.", "encoder.crosswalk_encoder.point_net."),
    ):
        if name.startswith(local):
            name = reference + name[len(local) :]
            break
    if name.startswith("lane_encoder."):
        name = "encoder." + name
        for i, target in enumerate(("self_line", "left_line", "right_line")):
            name = name.replace(f".lines.{i}.", f".{target}.")
        for i, target in enumerate(
            (
                "self_type",
                "left_type",
                "right_type",
                "traffic_light_type",
                "interpolating",
                "stop_sign",
            )
        ):
            name = name.replace(f".attributes.{i}.", f".{target}.")
        name = name.replace(".speed.", ".speed_limit.").replace(".position.", ".position_encode.")
    if name.startswith("interactions."):
        _, index, part, suffix = name.split(".", 3)
        component = {
            "interaction": "interaction_encoder",
            "cross": "query_encoder",
            "head": "decoder",
        }[part]
        name = f"decoder.interaction_stage.{index}.{component}.{suffix}"
    if ".attention." in name:
        kind = "self_attention" if ".interaction_encoder." in name else "cross_attention"
        name = name.replace(".attention.", f".{kind}.")
    if name.startswith("encoder.fusion_encoder."):
        return name
    return name.replace(".norm1.", ".norm_1.").replace(".norm2.", ".norm_2.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REVISION:
        raise ValueError(f"Expected official source at {REVISION}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    ).splitlines()
    if any(
        not (line.startswith("?? ") and "/__pycache__/" in line and line.endswith(".pyc"))
        for line in status
    ):
        raise ValueError("Official reference clone must be clean")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    reference_class = importlib.import_module("model.GameFormer").GameFormer
    torch.set_num_threads(1)
    torch.manual_seed(101)
    config = replace(ModelConfig(), encoder_layers=1, levels=1, actors=4, modes=2)
    reference = reference_class(
        modalities=2, neighbors_to_predict=1, future_len=80, encoder_layers=1, decoder_levels=1
    ).eval()
    local = GameFormer(config).eval()
    mapping = {name: upstream_name(name) for name in local.state_dict()}
    reference_state = reference.state_dict()
    local.load_state_dict(
        {
            name: reference_state[target].reshape(value.shape)
            for name, value in local.state_dict().items()
            for target in [mapping[name]]
        },
        strict=True,
    )
    assert sum(p.numel() for p in local.parameters() if p.requires_grad) == sum(
        p.numel() for p in reference.parameters() if p.requires_grad
    )
    sample = synthetic_sample(config, 0)
    batch = {key: value[None] for key, value in sample.items()}
    # Deliberate domain: all-valid, nonzero map coordinates, so the old and new
    # absence heuristics coincide. Explicit mask edge cases are tested separately.
    batch["map_lanes"][..., 0] += 5.1
    official = reference(batch)
    translated = {
        "trajectories": torch.stack([official[f"level_{k}_interactions"] for k in range(2)]),
        "scores": torch.stack([official[f"level_{k}_scores"] for k in range(2)]),
    }
    actual = local(batch)
    errors = {}
    for name in actual:
        torch.testing.assert_close(actual[name], translated[name], atol=3e-5, rtol=3e-5)
        errors[name] = float((actual[name] - translated[name]).abs().max().detach())
    # Extract only the three reviewed local PyTorch loss functions, avoiding the
    # file's unrelated TensorFlow/Waymo import-time dependencies.
    tree = ast.parse((source / "utils/inter_pred_utils.py").read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("imitation_loss", "gmm_loss", "level_k_loss")
    ]
    namespace = {"torch": torch, "F": F}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "pinned_upstream_losses", "exec"),
        namespace,
    )
    expected_loss = namespace["level_k_loss"](
        official, batch["future_state"][:, 0], batch["future_state"][:, 1], 1
    )[0]
    actual_loss = objective(actual, batch, "joint")["loss"]
    torch.testing.assert_close(actual_loss, expected_loss, atol=3e-5, rtol=3e-5)
    expected_loss.backward()
    actual_loss.backward()
    reference_params = dict(reference.named_parameters())
    gradient_error = 0.0
    for name, parameter in local.named_parameters():
        other = reference_params[mapping[name]]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=2e-4, msg=name)
            gradient_error = max(gradient_error, float((parameter.grad - other.grad).abs().max()))
    report = {
        "official_revision": head,
        "torch": str(torch.__version__),
        "domain": "fully valid synthetic fixture; D256 E1 K1 M2; eval mode",
        "mapped_tensors": len(mapping),
        "max_output_absolute_error": errors,
        "max_gradient_absolute_error": gradient_error,
        "local_loss": float(actual_loss.detach()),
        "upstream_loss": float(expected_loss.detach()),
        "claim": "operator/parameter/loss/gradient compatibility on this fixture; not trained-checkpoint or dataset parity",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
