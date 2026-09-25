"""Scripted waypoint pilot: follows an authored route by emitting the SAME
velocity commands the policy emits. Used for validation check 6.5 (scripted
flythrough, laptop model and Isaac Sim) and as the "oracle route" baseline.

Pure pursuit on a polyline: project the drone onto its current segment,
put a carrot ``lookahead_m`` further along the path, fly toward it at the
route's design speed (slowing inside ``brake_radius_m`` of the goal), and yaw
to face the direction of travel so the camera looks where the drone goes.

PURE torch, batched: every env follows the same polyline with its own progress.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from drone.math3d import wrap_pi, world_to_heading


class WaypointPilot:
    def __init__(
        self,
        waypoints: Sequence[Tuple[float, float, float]],
        n_envs: int,
        speed: float,
        lookahead_m: float = 0.8,
        brake_radius_m: float = 2.0,
        yaw_gain: float = 2.5,
        max_yaw_rate: float = 1.5,
        device: str | torch.device = "cpu",
    ):
        self.wp = torch.tensor(waypoints, dtype=torch.float32, device=device)   # (K, 3)
        seg = self.wp[1:] - self.wp[:-1]
        self.seg_len = torch.linalg.norm(seg, dim=-1)
        self.seg_dir = seg / self.seg_len[:, None].clamp(min=1e-9)
        self.cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(self.seg_len, 0)])
        self.total = float(self.cum[-1])
        self.k = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.speed = speed
        self.lookahead = lookahead_m
        self.brake = brake_radius_m
        self.yaw_gain = yaw_gain
        self.max_yaw_rate = max_yaw_rate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.k.zero_()
        else:
            self.k[env_ids] = 0

    def _point_at(self, s: torch.Tensor) -> torch.Tensor:
        """Arc-length s (N,) -> point on the polyline (N, 3)."""
        s = s.clamp(0.0, self.total)
        k = torch.searchsorted(self.cum, s, right=True).sub(1).clamp(0, len(self.seg_len) - 1)
        return self.wp[k] + self.seg_dir[k] * (s - self.cum[k])[:, None]

    def act(self, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """pos (N, 3) maze-local, yaw (N,) -> command (N, 4) [vx, vy, vz, yaw_rate] in the heading frame."""
        last = len(self.seg_len) - 1
        for _ in range(2):   # allow skipping at most two short segments per step
            a = self.wp[self.k]
            t = ((pos - a) * self.seg_dir[self.k]).sum(-1)
            advance = (t >= self.seg_len[self.k]) & (self.k < last)
            self.k = self.k + advance.long()
        a = self.wp[self.k]
        t = ((pos - a) * self.seg_dir[self.k]).sum(-1).clamp(min=0.0)
        s_here = self.cum[self.k] + torch.minimum(t, self.seg_len[self.k])
        carrot = self._point_at(s_here + self.lookahead)

        to_carrot = carrot - pos
        to_goal = torch.linalg.norm(self.wp[-1] - pos, dim=-1)
        speed = torch.clamp(self.speed * to_goal / self.brake, max=self.speed)
        v_w = to_carrot / torch.linalg.norm(to_carrot, dim=-1, keepdim=True).clamp(min=1e-6) * speed[:, None]

        heading = torch.atan2(v_w[:, 1], v_w[:, 0])
        horiz = torch.linalg.norm(v_w[:, :2], dim=-1)
        yaw_rate = torch.where(
            horiz > 0.2,
            (self.yaw_gain * wrap_pi(heading - yaw)).clamp(-self.max_yaw_rate, self.max_yaw_rate),
            torch.zeros_like(yaw),
        )
        v_h = world_to_heading(yaw, v_w)
        return torch.cat([v_h, yaw_rate[:, None]], dim=-1)
