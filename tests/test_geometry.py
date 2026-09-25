"""Analytic SDF: exact distances on hand-checkable cases, yaw, cylinders, bounds."""

import math

import torch

from maze.geometry import MazeGeometry
from maze.spec import maze_from_dict


def tiny(prims):
    return maze_from_dict({
        "name": "t", "bounds": [10, 10, 5],
        "start": {"pos": [1, 1, 1]}, "goal": {"pos": [9, 9, 1], "radius": 0.5},
        "primitives": prims,
        "routes": {
            "fast": {"design_speed_mps": 1, "waypoints": [[1, 1, 1], [9, 9, 1]],
                     "signature": [{"pos": [5, 5, 1], "size": [1, 1, 1]}]},
            "slow": {"design_speed_mps": 1, "waypoints": [[1, 1, 1], [9, 1, 1], [9, 9, 1]],
                     "signature": [{"pos": [9, 1, 1], "size": [1, 1, 1]}]},
        },
    })


def test_box_distance_outside_inside_and_corner():
    g = MazeGeometry(tiny([{"id": "b", "type": "box", "pos": [5, 5, 2.5], "size": [2, 2, 5]}]))
    pts = torch.tensor([[7.0, 5.0, 2.5], [5.0, 5.0, 2.5], [7.0, 7.0, 2.5]])
    d = g.box_sdf(pts)[:, 0]
    assert torch.allclose(d, torch.tensor([1.0, -1.0, math.sqrt(2)]), atol=1e-5)


def test_box_yaw_rotates_the_footprint():
    g = MazeGeometry(tiny([{"id": "b", "type": "box", "pos": [5, 5, 2.5], "size": [4, 0.2, 5], "yaw_deg": 90}]))
    # rotated 90 deg: long axis along y, so (5, 6.5) is inside and (6.5, 5) is 1.4 m out
    d = g.box_sdf(torch.tensor([[5.0, 6.5, 1.0], [6.5, 5.0, 1.0]]))[:, 0]
    assert d[0] < 0 and abs(d[1] - 1.4) < 1e-5


def test_cylinder_radial_and_axial():
    g = MazeGeometry(tiny([{"id": "c", "type": "cylinder", "pos": [5, 5, 2], "radius": 0.5, "height": 1}]))
    d = g.cylinder_sdf(torch.tensor([[6.5, 5.0, 2.0], [5.0, 5.0, 3.0], [5.0, 5.0, 2.0]]))[:, 0]
    assert torch.allclose(d, torch.tensor([1.0, 0.5, -0.5]), atol=1e-5)


def test_bounds_are_walls_floor_and_ceiling():
    g = MazeGeometry(tiny([]))
    c = g.clearance(torch.tensor([[5.0, 5.0, 0.1], [5.0, 5.0, 4.8], [0.2, 5.0, 2.0], [5.0, 11.0, 2.0]]))
    assert torch.allclose(c, torch.tensor([0.1, 0.2, 0.2, -1.0]), atol=1e-5)


def test_decor_is_not_a_collider():
    g = MazeGeometry(tiny([{"id": "d", "type": "box", "pos": [5, 5, 0.01], "size": [3, 3, 0.02],
                            "tag": "decor", "collision": False}]))
    assert g.box_c.shape[0] == 0


def test_segment_check_catches_a_thin_wall_between_samples():
    g = MazeGeometry(tiny([{"id": "w", "type": "box", "pos": [5, 5, 2.5], "size": [0.2, 10, 5]}]))
    a, b = torch.tensor([[2.0, 5.0, 1.0]]), torch.tensor([[8.0, 5.0, 1.0]])
    # both endpoints are clear (1 m: the floor), the segment passes straight through the wall
    assert g.clearance(torch.cat([a, b])).min() > 0.9
    assert g.segment_min_clearance(a, b)[0] < 0
