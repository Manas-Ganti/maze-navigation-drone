"""Voxel free-space grid, reachability, and the geodesic distance field.

PURE torch. Used for:
  * validation check 2 (reachability) -- a pathfinding check against the
    config, independent of the hand-authored waypoints;
  * the "unintended shortcut" check -- if the geodesic shortest path is much
    shorter than the authored fast route, the maze has a leak the designer did
    not intend, and the route-choice story is about the wrong routes;
  * the shaping potential Phi(s) = -geodesic_distance_to_goal(s).

Why geodesic, not Euclidean, distance for shaping: in a maze the straight line
to the goal often points into a wall or a dead end. Euclidean shaping then
rewards flying INTO that dead end (the sparse-exploration / local-optimum
failure in CLAUDE.md section 8). Geodesic distance is still a state potential,
so the Ng et al. (1999) invariance argument still holds -- it changes how fast
the optimum is found, not what it is. ``reward.potential: euclidean`` stays
available as an ablation.

Free space = voxel centres whose clearance >= the drone's collision radius,
i.e. where the drone's CENTRE may be. Distances use 26-connectivity with
exact step lengths (1, sqrt2, sqrt3) x voxel size, so the field
overestimates the true geodesic by at most ~8% (26-neighbour metric error)
and never cuts through a wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from maze.geometry import MazeGeometry

_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


@dataclass
class VoxelGrid:
    """A dense grid over the maze bounds. Index (i, j, k) has centre
    ``(i + 0.5, j + 0.5, k + 0.5) * res``."""

    res: float
    shape: Tuple[int, int, int]
    free: torch.Tensor          # (nx, ny, nz) bool
    clearance: torch.Tensor     # (nx, ny, nz) float, SDF at voxel centres

    def index_of(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) maze-local points -> (N, 3) long indices, clamped into the grid."""
        idx = torch.floor(points / self.res).long()
        hi = torch.tensor(self.shape, device=points.device) - 1
        return torch.minimum(idx.clamp(min=0), hi)

    def centers(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        dev = self.free.device
        axes = [(torch.arange(n, device=dev, dtype=torch.float32) + 0.5) * self.res for n in (nx, ny, nz)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1)


def build_grid(geom: MazeGeometry, res: float, robot_radius: float, chunk: int = 200_000) -> VoxelGrid:
    """Voxelise the maze bounds at ``res`` and mark voxels the drone centre may occupy."""
    X, Y, Z = geom.spec.bounds
    shape = (int(math.ceil(X / res)), int(math.ceil(Y / res)), int(math.ceil(Z / res)))
    tmp = VoxelGrid(res, shape, torch.zeros(shape, dtype=torch.bool, device=geom.device),
                    torch.zeros(shape, device=geom.device))
    pts = tmp.centers().reshape(-1, 3)
    clr = torch.cat([geom.clearance(pts[i : i + chunk]) for i in range(0, pts.shape[0], chunk)])
    clr = clr.reshape(shape)
    return VoxelGrid(res, shape, clr >= robot_radius, clr)


def _relax(dist: torch.Tensor, update_mask: torch.Tensor, source_mask: torch.Tensor,
           res: float, max_iters: int) -> Tuple[torch.Tensor, int]:
    """Bellman-Ford sweeps: dist[v] = min(dist[v], dist[u] + |u - v|) for
    neighbours u in ``source_mask``, applied only where ``update_mask``."""
    inf = float("inf")
    steps = [(o, res * math.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)) for o in _OFFSETS]
    nx, ny, nz = dist.shape
    for it in range(max_iters):
        src = torch.where(source_mask, dist, torch.full_like(dist, inf))
        # pad once; each neighbour offset is then a view: best[v] <- src[v - offset] + |offset|
        p = F.pad(src[None, None], (1, 1, 1, 1, 1, 1), mode="constant", value=inf)[0, 0]
        best = dist.clone()
        for (dx, dy, dz), w in steps:
            best = torch.minimum(best, p[1 - dx : 1 - dx + nx, 1 - dy : 1 - dy + ny, 1 - dz : 1 - dz + nz] + w)
        new = torch.where(update_mask, best, dist)
        if torch.equal(new, dist):
            return new, it
        dist = new
    return dist, max_iters


def geodesic_field(grid: VoxelGrid, goal: Tuple[float, float, float], goal_radius: float,
                   max_iters: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Geodesic distance (m) from every voxel to the goal, through free space.

    Phase 1: exact-ish shortest paths through FREE voxels (seeded with the
    Euclidean distance inside the goal sphere).
    Phase 2: fill non-free / unreachable voxels from their neighbours WITHOUT
    letting them feed back into free space. The drone's centre can sit in a
    "non-free" voxel while still being collision-free (voxel centre vs true
    position), so every voxel needs a finite, locally-consistent value.
    """
    dev = grid.free.device
    centers = grid.centers()
    g = torch.tensor(goal, device=dev, dtype=torch.float32)
    d_goal = torch.linalg.norm(centers - g, dim=-1)
    inf = float("inf")
    seed = grid.free & (d_goal <= max(goal_radius, grid.res))
    if not bool(seed.any()):
        raise ValueError("goal region contains no free voxel -- goal is inside geometry or grid too coarse")
    dist = torch.where(seed, d_goal, torch.full_like(d_goal, inf))
    dist, _ = _relax(dist, grid.free, grid.free, grid.res, max_iters)
    connected = torch.isfinite(dist)
    # phase 2: fill the rest from anything already finite
    dist, _ = _relax(dist, ~connected, torch.ones_like(grid.free), grid.res, max_iters)
    return dist, connected


def lookup(field: torch.Tensor, grid: VoxelGrid, points: torch.Tensor) -> torch.Tensor:
    """Nearest-voxel lookup of a field at maze-local points (N, 3) -> (N,)."""
    idx = grid.index_of(points)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]]


def reachable(grid: VoxelGrid, connected: torch.Tensor, point: Tuple[float, float, float]) -> bool:
    """True iff ``point``'s voxel is free AND connected to the goal through free space."""
    p = torch.tensor([point], device=grid.free.device, dtype=torch.float32)
    i = grid.index_of(p)[0]
    return bool(connected[i[0], i[1], i[2]])
