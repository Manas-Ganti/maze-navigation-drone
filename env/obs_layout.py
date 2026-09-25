"""Observation layout -- the single definition shared by the env and the policy. PURE.

Flat vector (rsl_rl needs (N, D)):

    [ image  C*H*W   (only if observation.camera_enabled) ]  RGB in [0, 1], channel-first
    [ lin_vel_b      3 ]   body-frame linear velocity * lin_vel_scale
    [ ang_vel_b      3 ]   body-frame angular velocity * ang_vel_scale
    [ rot6d          6 ]   first two columns of the body->world rotation
    [ goal_b         3 ]   body-frame vector to the goal * goal_vec_scale
    [ last_action    4 ]   previous policy output in [-1, 1]

No map, no route, no waypoints: the only global information is the goal
vector the spec allows (CLAUDE.md section 3). Note that a goal vector plus a
fixed maze is enough to memorise a route without looking -- the
``observation.zero_image`` ablation measures how much the camera actually
contributes, and the README must not claim "vision-based" skill without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

PROPRIO_PARTS: Tuple[Tuple[str, int], ...] = (
    ("lin_vel_b", 3),
    ("ang_vel_b", 3),
    ("rot6d", 6),
    ("goal_b", 3),
    ("last_action", 4),
)
NUM_ACTIONS = 4


@dataclass(frozen=True)
class ObsLayout:
    image_shape: Optional[Tuple[int, int, int]]   # (C, H, W) or None
    proprio_dim: int

    @property
    def image_dim(self) -> int:
        if self.image_shape is None:
            return 0
        c, h, w = self.image_shape
        return c * h * w

    @property
    def total_dim(self) -> int:
        return self.image_dim + self.proprio_dim

    def slices(self) -> Dict[str, slice]:
        out = {}
        i = 0
        if self.image_shape is not None:
            out["image"] = slice(0, self.image_dim)
            i = self.image_dim
        for name, n in PROPRIO_PARTS:
            out[name] = slice(i, i + n)
            i += n
        return out


def build_layout(data: Mapping[str, Any]) -> ObsLayout:
    cam = data["camera"]
    shape = (3, int(cam["height"]), int(cam["width"])) if data["observation"]["camera_enabled"] else None
    return ObsLayout(shape, sum(n for _, n in PROPRIO_PARTS))
