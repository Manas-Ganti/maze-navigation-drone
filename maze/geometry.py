"""Analytic maze geometry: exact signed distance from points to the maze.

PURE torch (CPU or GPU). This is the collision GROUND TRUTH used by the env's
termination, the voxel grid, the route-clearance checks and the pre-GPU
flythrough -- one implementation everywhere, so "the validator passed" and
"the env terminated" are statements about the same geometry.

Why analytic instead of PhysX contacts (playbook rules 1, 4 and 5.5):
  * exact against the config (no stale extents, no convex-hull inflation);
  * checked at every physics sub-step, so a fast drone cannot tunnel between
    two policy steps without being caught;
  * no per-prim contact-filter expressions (the "expected 64, found 3072"
    trap) and no ground-support false positives.
PhysX contacts are still read in the smoke test as a CROSS-CHECK: the two must
agree on the forced-crash probe, or one of them is lying.

Signed distance convention: positive = free space, negative = inside geometry.
``clearance(p)`` is the distance from p to the nearest surface, including the
bounds (floor, analytic ceiling, perimeter faces).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from maze.spec import MazeSpec


class MazeGeometry:
    """Batched SDF over one maze's colliders, on a given device."""

    def __init__(self, spec: MazeSpec, device: str | torch.device = "cpu"):
        self.spec = spec
        self.device = torch.device(device)
        f32 = dict(dtype=torch.float32, device=self.device)

        boxes = [p for p in spec.colliders() if p.type == "box"]
        cyls = [p for p in spec.colliders() if p.type == "cylinder"]
        self.box_ids = [p.id for p in boxes]
        self.cyl_ids = [p.id for p in cyls]

        self.box_c = torch.tensor([p.pos for p in boxes], **f32).reshape(-1, 3)
        self.box_h = torch.tensor([p.half_extents for p in boxes], **f32).reshape(-1, 3)
        yaw = torch.tensor([math.radians(p.yaw_deg) for p in boxes], **f32)
        self.box_cos, self.box_sin = torch.cos(yaw), torch.sin(yaw)

        self.cyl_c = torch.tensor([p.pos for p in cyls], **f32).reshape(-1, 3)
        self.cyl_r = torch.tensor([p.radius for p in cyls], **f32)
        self.cyl_hh = torch.tensor([p.height / 2.0 for p in cyls], **f32)

        self.bounds = torch.tensor(spec.bounds, **f32)

    # ------------------------------------------------------------------
    def box_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N, B) signed distance to each yawed box."""
        if self.box_c.shape[0] == 0:
            return points.new_full((points.shape[0], 0), float("inf"))
        d = points[:, None, :] - self.box_c[None, :, :]
        # rotate the offset into the box frame (by -yaw)
        x = self.box_cos * d[..., 0] + self.box_sin * d[..., 1]
        y = -self.box_sin * d[..., 0] + self.box_cos * d[..., 1]
        q = torch.stack([x.abs(), y.abs(), d[..., 2].abs()], dim=-1) - self.box_h[None]
        outside = torch.linalg.norm(q.clamp(min=0.0), dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0.0)
        return outside + inside

    def cylinder_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N, C) signed distance to each vertical cylinder."""
        if self.cyl_c.shape[0] == 0:
            return points.new_full((points.shape[0], 0), float("inf"))
        d = points[:, None, :] - self.cyl_c[None, :, :]
        radial = torch.linalg.norm(d[..., :2], dim=-1) - self.cyl_r[None]
        axial = d[..., 2].abs() - self.cyl_hh[None]
        q = torch.stack([radial, axial], dim=-1)
        outside = torch.linalg.norm(q.clamp(min=0.0), dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0.0)
        return outside + inside

    def bounds_clearance(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N,) distance to the nearest face of the flyable box
        (negative outside it). Floor z=0, analytic ceiling z=Z, perimeter faces."""
        lo = points
        hi = self.bounds[None] - points
        return torch.minimum(lo.min(dim=-1).values, hi.min(dim=-1).values)

    def obstacle_clearance(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N,) distance to the nearest interior primitive."""
        parts = [self.box_sdf(points), self.cylinder_sdf(points)]
        allp = torch.cat(parts, dim=-1)
        if allp.shape[-1] == 0:
            return points.new_full((points.shape[0],), float("inf"))
        return allp.min(dim=-1).values

    def clearance(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N,) signed distance to the nearest surface of any kind."""
        points = points.to(self.device, torch.float32)
        return torch.minimum(self.obstacle_clearance(points), self.bounds_clearance(points))

    def nearest_primitive(self, point: torch.Tensor) -> str:
        """Id of the closest primitive to ONE point (diagnostics only)."""
        p = point.reshape(1, 3).to(self.device, torch.float32)
        cands = []
        if self.box_ids:
            b = self.box_sdf(p)[0]
            cands.append((float(b.min()), self.box_ids[int(b.argmin())]))
        if self.cyl_ids:
            c = self.cylinder_sdf(p)[0]
            cands.append((float(c.min()), self.cyl_ids[int(c.argmin())]))
        cands.append((float(self.bounds_clearance(p)[0]), "bounds"))
        return min(cands)[1]

    # ------------------------------------------------------------------
    def collides(self, points: torch.Tensor, radius: float) -> torch.Tensor:
        """(N, 3) -> (N,) bool: a sphere of ``radius`` at each point touches geometry."""
        return self.clearance(points) < radius

    def segment_min_clearance(
        self, a: torch.Tensor, b: torch.Tensor, step_m: float = 0.02, max_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Minimum clearance along straight segments a[i] -> b[i] (both (N, 3)).

        Sampled at <= ``step_m`` spacing. The SDF is 1-Lipschitz, so between
        two samples the true minimum can be at most ``step_m / 2`` lower than
        the sampled one -- callers should budget that into their margins.
        """
        a = a.to(self.device, torch.float32).reshape(-1, 3)
        b = b.to(self.device, torch.float32).reshape(-1, 3)
        length = torch.linalg.norm(b - a, dim=-1).max().item()
        n = max(2, int(math.ceil(length / step_m)) + 1)
        if max_samples is not None:
            n = min(n, max_samples)
        t = torch.linspace(0.0, 1.0, n, device=self.device)
        pts = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]   # (N, n, 3)
        clr = self.clearance(pts.reshape(-1, 3)).reshape(a.shape[0], n)
        return clr.min(dim=-1).values
