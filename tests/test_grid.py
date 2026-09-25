"""Voxel reachability + geodesic field."""

import torch

from maze.geometry import MazeGeometry
from maze.grid import build_grid, geodesic_field, lookup, reachable
from tests.test_geometry import tiny


def test_open_room_geodesic_close_to_euclidean():
    spec = tiny([])
    grid = build_grid(MazeGeometry(spec), 0.25, 0.15)
    field, conn = geodesic_field(grid, spec.goal_pos, spec.goal_radius_m)
    d = float(lookup(field, grid, torch.tensor([spec.start_pos]))[0])
    euclid = ((8 ** 2 + 8 ** 2) ** 0.5)
    assert euclid - 0.5 <= d <= euclid * 1.09   # 26-neighbour metric error <= ~8%
    assert reachable(grid, conn, spec.start_pos)


def test_wall_forces_detour_and_is_never_cut():
    # wall across x = 5 except a gap at the top (y 8.5-10)
    spec = tiny([{"id": "w", "type": "box", "pos": [5, 4.25, 2.5], "size": [0.3, 8.5, 5]}])
    grid = build_grid(MazeGeometry(spec), 0.2, 0.15)
    field, conn = geodesic_field(grid, spec.goal_pos, spec.goal_radius_m)
    s = torch.tensor([[2.0, 2.0, 1.0]])
    d = float(lookup(field, grid, s)[0])
    straight = float(torch.linalg.norm(s[0] - torch.tensor(spec.goal_pos)))
    via_gap = 7.8 + 4.0   # (2,2) -> gap near (5, 9.2) -> goal (9, 9)
    assert d > straight + 1.5 and abs(d - via_gap) < 0.1 * via_gap


def test_sealed_start_is_unreachable_even_though_field_is_filled():
    spec = tiny([{"id": "w", "type": "box", "pos": [5, 5, 2.5], "size": [0.3, 10, 5]}])
    grid = build_grid(MazeGeometry(spec), 0.25, 0.15)
    field, conn = geodesic_field(grid, spec.goal_pos, spec.goal_radius_m)
    assert not reachable(grid, conn, spec.start_pos)
    assert torch.isfinite(lookup(field, grid, torch.tensor([spec.start_pos]))).all()   # phase-2 fill
