"""All-level joint imitation and released planning interaction objectives."""

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, value, 0).sum() / valid.sum().clamp_min(1)


def interaction_potential(current: Tensor, previous: Tensor, valid: Tensor) -> Tensor:
    """B,P,M,T,2. Ego->previous neighbors; neighbors->previous other neighbors."""
    b, n, m, t, _ = current.shape
    if n < 2:
        return current.sum() * 0
    prev = previous[:, 1:].permute(0, 3, 1, 2, 4).reshape(b * t, (n - 1) * m, 2)
    others_valid = (
        valid[:, 1:].permute(0, 2, 1)[..., None].expand(-1, -1, -1, m).reshape(b * t, (n - 1) * m)
    )
    ego = current[:, 0].permute(0, 2, 1, 3).reshape(b * t, m, 2)
    distance = torch.cdist(ego, prev).masked_fill(~others_valid[:, None], 1000).amin(-1)
    ego_valid = valid[:, 0].reshape(b * t, 1)
    potential = torch.where((distance < 3) & ego_valid, 1 / (distance + 1), 0)
    ego_loss = potential.reshape(b, t, m).sum((1, 2)).mean()
    neighbors = current[:, 1:].permute(0, 3, 1, 2, 4).reshape(b * t, (n - 1) * m, 2)
    distance = torch.cdist(neighbors, prev)
    same_agent = (
        torch.eye(n - 1, dtype=torch.bool, device=current.device)
        .repeat_interleave(m, 0)
        .repeat_interleave(m, 1)
    )
    pair_mask = (~others_valid[:, None]) | same_agent[None]
    distance = distance.masked_fill(pair_mask, 1000).amin(-1)
    potential = torch.where((distance < 3) & others_valid, 1 / (distance + 1), 0)
    # Match source's time sum and mean across current neighbor-mode entries.
    denominator = valid[:, 1:].any(-1).sum(-1).clamp_min(1) * m
    neighbor_loss = (potential.reshape(b, t, -1).sum((1, 2)) / denominator).mean()
    return ego_loss + neighbor_loss


def objective(outputs: dict[str, Tensor], batch: dict[str, Tensor], task: str) -> dict[str, Tensor]:
    if task not in ("joint", "planning"):
        raise ValueError("task must be joint or planning")
    valid = batch["future_valid"]
    target = torch.where(valid[..., None], batch["future"].float(), 0)
    stages = outputs["trajectories"].float()
    logits = outputs["scores"].float()
    valid_agents = valid.any(-1)
    valid_scenes = valid_agents.any(-1)
    zero = stages.sum() * 0 + logits.sum() * 0
    regression, classification, interaction = zero, zero, zero
    for level, raw in enumerate(stages):
        clean = torch.where(valid[:, :, None, :, None], raw, 0)
        distance = (clean[..., :2] - target[:, :, None]).norm(dim=-1)
        if task == "joint":
            if target.shape[1:3] != (2, 80):
                raise ValueError("Joint objective requires two agents and 80 future samples")
            temporal = (distance * valid[:, :, None]).sum(-1) / valid.sum(-1)[:, :, None].clamp_min(
                1
            )
            horizons = torch.tensor([29, 49, 79], device=target.device)
            temporal = temporal + (distance[..., horizons] * valid[:, :, None, horizons]).sum(-1)
        else:
            sampled = valid[..., 4::5]
            temporal = (distance[..., 4::5] * sampled[:, :, None]).sum(-1) / sampled.sum(-1)[
                :, :, None
            ].clamp_min(1)
        joint_distance = (temporal * valid_agents[..., None]).sum(1) / valid_agents.sum(1)[
            :, None
        ].clamp_min(1)
        mode = joint_distance.argmin(-1)
        b, p, _, t, _ = raw.shape
        chosen = clean.gather(2, mode[:, None, None, None, None].expand(b, p, 1, t, 4)).squeeze(2)
        dx = target - chosen[..., :2]
        lo, hi = (0, 5) if task == "joint" else (-2, 2)
        log_std = chosen[..., 2:].clamp(lo, hi)
        nll = log_std.sum(-1) + 0.5 * (dx * torch.exp(-log_std)).square().sum(-1)
        nll = torch.where(valid, nll, 0)
        if task == "joint":
            per_agent = nll.sum(-1) / valid.sum(-1).clamp_min(1) + nll[..., horizons].sum(-1)
            regression = regression + masked_mean(per_agent, valid_agents)
            scene_logits = (logits[level] * valid_agents[..., None]).sum(1)
            ce = F.cross_entropy(scene_logits, mode, label_smoothing=0.2, reduction="none")
            classification = classification + 2 * masked_mean(ce, valid_scenes)
        else:
            regression = regression + masked_mean(nll, valid) + masked_mean(nll[:, 0], valid[:, 0])
            ce = F.cross_entropy(
                logits[level].transpose(1, 2),
                mode[:, None].expand(-1, p),
                label_smoothing=0.2,
                reduction="none",
            )
            classification = classification + masked_mean(ce, valid_agents)
            if level:
                interaction = interaction + 0.1 * interaction_potential(
                    clean[..., :2],
                    torch.where(valid[:, :, None, :, None], stages[level - 1, ..., :2], 0),
                    valid,
                )
    return {
        "loss": regression + classification + interaction,
        "regression": regression,
        "classification": classification,
        "interaction": interaction,
    }
