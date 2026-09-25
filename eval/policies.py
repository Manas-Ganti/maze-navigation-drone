"""Policies the eval harness can run through the SAME env loop: trained checkpoint,
random, and the scripted route-follower (the sanity triad, playbook s7)."""

from __future__ import annotations

from typing import Any, Dict

import torch

from drone.math3d import quat_to_rotmat, yaw_of
from drone.pilot import WaypointPilot


class CheckpointPolicy:
    """rsl_rl checkpoint -> deterministic (mean) or stochastic actions."""

    def __init__(self, wrapped_env: Any, data: Dict[str, Any], deterministic: bool = True):
        from training.ppo_config import build_runner  # noqa: PLC0415

        self.runner = build_runner(wrapped_env, data, log_dir=None, device=str(wrapped_env.unwrapped.device))
        self.deterministic = deterministic

    def load(self, path: str) -> None:
        self.runner.load(path, load_optimizer=False)
        self.policy = self.runner.alg.policy
        self.policy.eval()

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.act_inference(obs) if self.deterministic else self.policy.act(obs)

    def reset(self, env_ids=None) -> None:
        pass


class RandomPolicy:
    def __init__(self, env: Any):
        self.env = env

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.rand(self.env.num_envs, 4, device=self.env.device) * 2 - 1

    def reset(self, env_ids=None) -> None:
        pass


class ScriptedRoutePolicy:
    """Flies one authored route with the waypoint pilot (an oracle that knows the route)."""

    def __init__(self, env: Any, kind: str):
        r = env.maze.route(kind)
        self.env = env
        self.pilot = WaypointPilot(r.waypoints, env.num_envs, r.design_speed_mps, device=env.device)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        pos = self.env.pos_local()
        yaw = yaw_of(quat_to_rotmat(self.env.robot.data.root_quat_w))
        return self.env.unscale_commands(self.pilot.act(pos, yaw))

    def reset(self, env_ids=None) -> None:
        self.pilot.reset(env_ids)
