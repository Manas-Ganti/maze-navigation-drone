"""Reward (CLAUDE.md section 4). PURE torch -- the env only gathers state and calls this.

    r = w * (gamma_s * Phi(s') - Phi(s))     potential-based shaping, Phi = -distance-to-goal
      - step_penalty                          every step: the speed incentive
      - collision_penalty * collided          terminal
      + goal_bonus * reached                  terminal

Shaping is EXACTLY potential-based (Ng, Harada & Russell 1999): with
``gamma_s`` equal to PPO's gamma and ``Phi(terminal) = 0`` the discounted
shaping return from any start state is the constant ``-w * Phi(s0)`` for
EVERY trajectory, so it cannot change which route is optimal -- only how fast
it is found. ``tests/test_reward.py`` checks that telescoping numerically.

Two consequences worth knowing when reading the logged components:
  * per step, shaping includes ``w * (1 - gamma_s) * d(s')`` -- a small
    positive term that grows with distance. It is part of the invariance, not a
    bug; it is cancelled in return by the terminal step.
  * on a crash, the terminal step's shaping is ``+w * d(s)`` (Phi(terminal) = 0).
    The magnitude that decides behaviour is therefore goal_bonus vs
    collision_penalty vs accumulated time, which is what
    ``maze.routes.break_even`` computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

COMPONENTS = ("progress", "time", "collision", "goal")


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float
    shaping_gamma: float
    terminal_potential_zero: bool
    step_penalty: float
    collision_penalty: float
    goal_bonus: float

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "RewardConfig":
        from training.config import shaping_gamma  # noqa: PLC0415

        rw = data["reward"]
        return cls(
            progress_weight=float(rw["progress_weight"]),
            shaping_gamma=shaping_gamma(data),
            terminal_potential_zero=bool(rw["terminal_potential_zero"]),
            step_penalty=float(rw["step_penalty"]),
            collision_penalty=float(rw["collision_penalty"]),
            goal_bonus=float(rw["goal_bonus"]),
        )


def compute_reward(
    dist_prev: torch.Tensor,
    dist_next: torch.Tensor,
    reached: torch.Tensor,
    collided: torch.Tensor,
    cfg: RewardConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """All inputs (N,). ``dist_*`` are potential distances (geodesic or Euclidean, metres).

    ``reached`` and ``collided`` must be mutually exclusive (the env gives
    collision precedence); both are terminal.
    """
    terminal = reached | collided
    phi_prev = -dist_prev
    phi_next = -dist_next
    if cfg.terminal_potential_zero:
        phi_next = torch.where(terminal, torch.zeros_like(phi_next), phi_next)
    progress = cfg.progress_weight * (cfg.shaping_gamma * phi_next - phi_prev)
    comps = {
        "progress": progress,
        "time": torch.full_like(progress, -cfg.step_penalty),
        "collision": -cfg.collision_penalty * collided.float(),
        "goal": cfg.goal_bonus * reached.float(),
    }
    total = comps["progress"] + comps["time"] + comps["collision"] + comps["goal"]
    return total, comps
