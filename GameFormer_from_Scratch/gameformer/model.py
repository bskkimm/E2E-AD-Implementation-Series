"""Local GameFormer operators. Axes and release-specific behavior are explicit."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


def segment_points(
    features: Tensor, valid: Tensor, stride: int, partial: bool = False
) -> tuple[Tensor, Tensor]:
    """[B,L,Q,D] -> [B,L*Q/stride,D], True padding mask; never pool padding."""
    b, lanes, points, width = features.shape
    grouped = features.reshape(b, lanes, points // stride, stride, width)
    flags = valid.reshape(b, lanes, points // stride, stride)
    pooled = grouped.masked_fill(~flags[..., None], -torch.inf).amax(-2)
    any_valid = flags.any(-1)
    pooled = torch.where(any_valid[..., None], pooled, 0)
    usable = any_valid if partial else flags.all(-1)
    return pooled.reshape(b, -1, width), ~usable.reshape(b, -1)


class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, width: int, length: int, dropout: float):
        super().__init__()
        positions = torch.arange(length)[:, None]
        divisor = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(length, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * divisor), torch.cos(positions * divisor)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe)


class AgentEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.motion = nn.LSTM(8, width, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, width, padding_idx=0)

    def forward(self, states: Tensor) -> Tensor:
        encoded, _ = self.motion(states[..., :8])
        return encoded[:, -1] + self.type_emb(states[:, -1, 8].long())


class LaneEncoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.lines = nn.ModuleList([nn.Linear(3, 128) for _ in range(3)])
        self.speed = nn.Linear(1, 64)
        self.attributes = nn.ModuleList(
            [
                nn.Embedding(n, 64, padding_idx=0 if i < 4 else None)
                for i, n in enumerate((4, 11, 11, 9, 2, 2))
            ]
        )
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, c.width))
        self.position = PositionalEncoding(c.width, c.lane_points, c.dropout)
        self.columns = (
            (10, 11, 12, 13, 14, 15) if c.legacy_lane_columns else (10, 11, 12, 13, 15, 16)
        )

    def forward(self, lanes: Tensor) -> Tensor:
        numeric = [layer(lanes[..., i * 3 : i * 3 + 3]) for i, layer in enumerate(self.lines)]
        semantic = sum(
            layer(lanes[..., col].long())
            for col, layer in zip(self.columns, self.attributes, strict=True)
        )
        return self.position(
            self.pointnet(torch.cat([*numeric, self.speed(lanes[..., 9:10]), semantic], -1))
        )


class CrosswalkEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.pointnet = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, width)
        )

    def forward(self, points: Tensor) -> Tensor:
        return self.pointnet(points)


class AttentionBlock(nn.Module):
    """Post-norm attention. Cross-attention has no query residual in the release."""

    def __init__(self, c: ModelConfig, self_attention: bool = False):
        super().__init__()
        self.attention = nn.MultiheadAttention(c.width, c.heads, c.dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.ffn = nn.Sequential(
            nn.Linear(c.width, c.width * 4),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.width * 4, c.width),
            nn.Dropout(c.dropout),
        )
        self.self_attention = self_attention

    def forward(self, query: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        attended = self.attention(
            query, context, context, key_padding_mask=mask, need_weights=False
        )[0]
        attended = self.norm1(attended + query if self.self_attention else attended)
        return self.norm2(attended + self.ffn(attended))


class FutureEncoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, c.width))
        self.type_emb = nn.Embedding(4, c.width, padding_idx=0)
        self.legacy_heading = c.legacy_heading

    def forward(self, trajectories: Tensor, current: Tensor) -> Tensor:
        # B,P,M,T,2; append the observed current xy before finite differences.
        start = current[:, :, None, None, :2].expand(-1, -1, trajectories.shape[2], -1, -1)
        delta = torch.diff(torch.cat((start, trajectories), -2), dim=-2)
        dx = delta[..., 0].clamp_min(1e-3) if self.legacy_heading else delta[..., 0]
        angle = torch.atan2(delta[..., 1], dx)[..., None]
        size = current[:, :, None, None, 5:8].expand(*trajectories.shape[:-1], 3)
        states = torch.cat((trajectories, angle, delta / 0.1, size), -1)
        features = self.mlp(states.detach()).amax(-2)
        return features + self.type_emb(current[:, :, None, 8].long())


class GaussianHead(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.future = c.future
        self.gaussian = nn.Sequential(
            nn.Linear(c.width, c.width * 2),
            nn.ELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.width * 2, c.future * 4),
        )
        self.score = nn.Sequential(
            nn.Linear(c.width, 64), nn.ELU(), nn.Dropout(c.dropout), nn.Linear(64, 1)
        )

    def forward(self, content: Tensor, current: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.gaussian(content).reshape(*content.shape[:-1], self.future, 4)
        result = torch.cat((raw[..., :2] + current[:, :, None, None, :2], raw[..., 2:]), -1)
        return result, self.score(content).squeeze(-1)


class InteractionDecoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.interaction = AttentionBlock(c, self_attention=True)
        self.cross = AttentionBlock(c)
        self.head = GaussianHead(c)


class GameFormer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = c = config
        self.ego_encoder, self.agent_encoder = AgentEncoder(c.width), AgentEncoder(c.width)
        self.lane_encoder, self.crosswalk_encoder = LaneEncoder(c), CrosswalkEncoder(c.width)
        layer = nn.TransformerEncoderLayer(
            c.width, c.heads, c.width * 4, c.dropout, activation=F.gelu, batch_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(
            layer, c.encoder_layers, enable_nested_tensor=False
        )
        self.mode_embedding = nn.Embedding(c.modes, c.width)
        self.agent_embedding = nn.Embedding(c.predicted, c.width)
        self.initial_cross, self.initial_head = AttentionBlock(c), GaussianHead(c)
        self.future_encoder = FutureEncoder(c)
        self.interactions = nn.ModuleList([InteractionDecoder(c) for _ in range(c.levels)])

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        c = self.config
        actors = torch.cat((batch["ego_state"][:, None], batch["neighbors_state"]), 1)
        actors = torch.where(batch["actor_valid"][..., None], actors, 0)
        b = actors.shape[0]
        actor_mask = ~batch["actor_valid"][..., -1]
        if actor_mask[:, 0].any():
            raise ValueError("Focal actor must be present")
        encoded_ego = self.ego_encoder(actors[:, 0])[:, None]
        if c.actors > 1:
            others = self.agent_encoder(actors[:, 1:].reshape(-1, c.history, 9)).reshape(
                b, c.actors - 1, c.width
            )
            encoded_actors = torch.cat((encoded_ego, others), 1)
        else:
            encoded_actors = encoded_ego
        lanes = torch.where(batch["lane_valid"][..., None], batch["map_lanes"], 0)
        crosswalks = torch.where(batch["crosswalk_valid"][..., None], batch["map_crosswalks"], 0)
        lane_features = self.lane_encoder(lanes).reshape(
            b * c.predicted, c.lanes, c.lane_points, c.width
        )
        cw_features = self.crosswalk_encoder(crosswalks).reshape(
            b * c.predicted, c.crosswalks, c.crosswalk_points, c.width
        )
        lane_tokens, lane_mask = segment_points(
            lane_features,
            batch["lane_valid"].reshape(b * c.predicted, c.lanes, c.lane_points),
            c.lane_stride,
            c.partial_segments,
        )
        cw_tokens, cw_mask = segment_points(
            cw_features,
            batch["crosswalk_valid"].reshape(b * c.predicted, c.crosswalks, c.crosswalk_points),
            c.crosswalk_stride,
            c.partial_segments,
        )
        scene = torch.cat(
            (
                encoded_actors[:, None]
                .expand(-1, c.predicted, -1, -1)
                .reshape(b * c.predicted, c.actors, c.width),
                lane_tokens,
                cw_tokens,
            ),
            1,
        )
        mask = torch.cat(
            (
                actor_mask[:, None].expand(-1, c.predicted, -1).reshape(b * c.predicted, c.actors),
                lane_mask,
                cw_mask,
            ),
            1,
        )
        scene = self.scene_encoder(scene, src_key_padding_mask=mask)
        scene = scene.reshape(b, c.predicted, -1, c.width)
        mask = mask.reshape(b, c.predicted, -1)
        focal_history = scene[
            :,
            torch.arange(c.predicted, device=actors.device),
            torch.arange(c.predicted, device=actors.device),
        ]
        query = (
            focal_history[:, :, None]
            + self.mode_embedding.weight[None, None]
            + self.agent_embedding.weight[None, :, None]
        )
        content = self.initial_cross(
            query.reshape(b * c.predicted, c.modes, c.width),
            scene.reshape(b * c.predicted, -1, c.width),
            mask.reshape(b * c.predicted, -1),
        ).reshape(b, c.predicted, c.modes, c.width)
        current = actors[:, : c.predicted, -1]
        trajectories, scores = self.initial_head(content, current)
        all_trajectories, all_scores = [trajectories], [scores]
        for module in self.interactions:
            block = cast(InteractionDecoder, module)
            multi = self.future_encoder(trajectories[..., :2], current)
            weighted = multi * scores.softmax(-1)[..., None]
            futures = weighted.sum(2) if c.normalized_future_pool else weighted.mean(2)
            interaction = block.interaction(futures, futures, actor_mask[:, : c.predicted])
            context = torch.cat((interaction[:, None].expand(-1, c.predicted, -1, -1), scene), 2)
            future_mask = actor_mask[:, None, : c.predicted].expand(-1, c.predicted, -1).clone()
            future_mask |= torch.eye(c.predicted, device=actors.device, dtype=torch.bool)[None]
            combined_mask = torch.cat((future_mask, mask), 2)
            content = block.cross(
                (content + multi).reshape(b * c.predicted, c.modes, c.width),
                context.reshape(b * c.predicted, -1, c.width),
                combined_mask.reshape(b * c.predicted, -1),
            ).reshape(b, c.predicted, c.modes, c.width)
            trajectories, scores = block.head(content, current)
            all_trajectories.append(trajectories)
            all_scores.append(scores)
        return {"trajectories": torch.stack(all_trajectories), "scores": torch.stack(all_scores)}
