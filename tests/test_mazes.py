"""Every shipped maze passes the pure s6 checklist -- the same gate as tools/check_mazes.py."""

from pathlib import Path

import pytest

from maze.routes import break_even
from maze.spec import load_maze
from maze.validate import validate_maze
from training.config import load_config, policy_dt, shaping_gamma

MAZES = sorted(Path("configs/mazes").glob("*.yaml"))


def test_there_are_two_or_three_mazes():
    assert 2 <= len(MAZES) <= 3   # CLAUDE.md s5.1


@pytest.mark.parametrize("path", MAZES, ids=lambda p: p.stem)
def test_maze_passes_checklist(path):
    cfg = load_config("configs/train.yaml")
    report = validate_maze(load_maze(path), cfg)
    assert report.passed, report.text()


@pytest.mark.parametrize("path", MAZES, ids=lambda p: p.stem)
def test_break_even_is_informative(path):
    """The collision penalty must make the trade-off real: the fast route is worth it only
    below some crash probability that is neither ~0 (never risk it) nor ~1 (risk is free)."""
    cfg = load_config("configs/train.yaml")
    spec = load_maze(path)
    rw = cfg["reward"]
    be = break_even(spec.route("fast"), spec.route("slow"), rw["step_penalty"], rw["goal_bonus"],
                    rw["collision_penalty"], shaping_gamma(cfg), policy_dt(cfg))
    assert 0.05 < be.p_crash_break_even < 0.6, be
