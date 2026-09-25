"""Laptop-side rigid-body quadrotor model (torch). NOT the simulator.

Exists so the controller, the scripted pilot and the maze layouts can be
exercised together BEFORE any GPU time (CLAUDE.md section 6.5 asks for a
scripted flythrough in the sim; this is the cheap rehearsal that catches
controller sign errors, corner overshoot into walls, and gaps that are only
"visually plausible"). The real check still runs in Isaac Sim
(``eval/validate_sim.py``) because PhysX, not this model, is the ground truth.

Model: a single rigid body, body-frame thrust along +z and body torques,
gravity, no drag, semi-implicit Euler. Default mass/inertia are the Crazyflie
2.x nominal values Isaac Lab's cf2x asset is built around; the sim reads the
real ones from PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from drone.controller import GRAVITY
from drone.math3d import quat_mul, quat_to_rotmat

CF_MASS_KG = 0.028
CF_INERTIA = (1.4e-5, 1.4e-5, 2.2e-5)


@dataclass
class RigidBodyState:
    pos: torch.Tensor       # (N, 3) world
    quat: torch.Tensor      # (N, 4) wxyz body->world
    vel: torch.Tensor       # (N, 3) world
    ang_vel_b: torch.Tensor  # (N, 3) body


class QuadrotorModel:
    def __init__(self, n: int, mass: float = CF_MASS_KG, inertia=CF_INERTIA, device: str = "cpu"):
        self.n = n
        self.mass = torch.full((n,), mass, device=device)
        self.J = torch.tensor(inertia, device=device).expand(n, 3).clone()
        self.device = device

    def step(self, s: RigidBodyState, force_b: torch.Tensor, torque_b: torch.Tensor, dt: float) -> RigidBodyState:
        R = quat_to_rotmat(s.quat)
        acc = torch.einsum("nij,nj->ni", R, force_b) / self.mass[:, None]
        acc[:, 2] -= GRAVITY
        vel = s.vel + acc * dt
        pos = s.pos + vel * dt
        Jw = self.J * s.ang_vel_b
        w_dot = (torque_b - torch.linalg.cross(s.ang_vel_b, Jw)) / self.J
        w = s.ang_vel_b + w_dot * dt
        dq = torch.cat([torch.zeros(self.n, 1, device=self.device), w], dim=-1)
        quat = s.quat + 0.5 * quat_mul(s.quat, dq) * dt
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
        return RigidBodyState(pos, quat, vel, w)
