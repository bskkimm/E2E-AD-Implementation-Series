"""Meters, radians, column-vector rotation; local +x follows the focal heading."""

import torch
from torch import Tensor


def world_to_local(xy: Tensor, origin: Tensor, heading: Tensor) -> Tensor:
    while origin.ndim < xy.ndim:
        origin = origin.unsqueeze(-2)
    while heading.ndim < xy.ndim - 1:
        heading = heading.unsqueeze(-1)
    delta = xy - origin
    c, s = heading.cos(), heading.sin()
    return torch.stack(
        (c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1
    )


def local_to_world(xy: Tensor, origin: Tensor, heading: Tensor) -> Tensor:
    while origin.ndim < xy.ndim:
        origin = origin.unsqueeze(-2)
    return world_to_local(xy, torch.zeros_like(origin), -heading) + origin


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(angle.sin(), angle.cos())
