"""Route classification, signature tracking and the scripted pilot."""

import torch

from drone.pilot import WaypointPilot
from maze.routes import ROUTE_BOTH, ROUTE_FAST, ROUTE_NONE, ROUTE_SLOW, RouteTracker, classify, densify
from maze.spec import load_maze


def test_classify_table():
    assert classify(True, False) == ROUTE_FAST
    assert classify(False, True) == ROUTE_SLOW
    assert classify(True, True) == ROUTE_BOTH
    assert classify(False, False) == ROUTE_NONE


def test_tracker_needs_every_signature_region():
    spec = load_maze("configs/mazes/maze_b_overpass.yaml")   # fast route has TWO slots
    tr = RouteTracker(spec, 2)
    slot1 = torch.tensor([2.5, 7.0, 2.9])
    slot2 = torch.tensor([2.5, 13.0, 2.9])
    tr.update(torch.stack([slot1, slot1]))
    assert tr.taken()[:, 0].tolist() == [False, False]
    tr.update(torch.stack([slot2, torch.tensor([1.0, 1.0, 1.0])]))
    assert tr.taken()[:, 0].tolist() == [True, False]
    tr.reset(torch.tensor([0]))
    assert not tr.taken()[0, 0]


def test_pilot_heads_along_the_path():
    p = WaypointPilot([[0, 0, 1], [10, 0, 1], [10, 10, 1]], 2, speed=2.0)
    cmd = p.act(torch.tensor([[1.0, 0.0, 1.0], [10.0, 5.0, 1.0]]), torch.tensor([0.0, 0.0]))
    assert cmd[0, 0] > 1.9 and abs(cmd[0, 1]) < 0.1          # first leg: straight ahead
    assert cmd[1, 1] > 1.5 and cmd[1, 3] > 0                  # on the second leg: move +y (left) and yaw left


def test_densify_spacing():
    pts = densify([[0, 0, 0], [1, 0, 0], [1, 2, 0]], 0.1)
    step = torch.linalg.norm(pts[1:] - pts[:-1], dim=-1)
    assert step.max() <= 0.1 + 1e-6 and torch.allclose(pts[-1], torch.tensor([1.0, 2.0, 0.0]))
