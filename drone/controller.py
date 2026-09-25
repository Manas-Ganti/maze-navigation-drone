"""Low-level velocity controller: (vx, vy, vz, yaw_rate) -> body thrust + torque.

The policy is NOT trained to fly; it commands velocities (CLAUDE.md section 3)
and this fixed controller turns them into the wrench Isaac Lab's quadcopter
demo applies to the Crazyflie body (``set_external_force_and_torque``, body
frame). It is a cascaded geometric controller (Lee, Leok & McClamroch 2010):

  velocity loop:  a_cmd = Kv (v_cmd - v)          (tilt- and thrust-limited)
                  f_des = m (a_cmd + g e3)
  thrust:         T = f_des . b3                  (clamped to [0, T/W * m g])
  attitude:       R_des from (f_des direction, integrated yaw setpoint)
                  e_R = 1/2 vee(R_des^T R - R^T R_des)
                  tau = J (-kR e_R - kW e_W) + W x J W

Commands are in the HEADING frame (gravity-aligned, rotated by the current
yaw): vx forward, vy left, vz up. That matches what the forward camera sees.

PURE torch -- the same code runs in the Isaac env (GPU) and in the
laptop-side rigid-body model (``drone/dynamics.py``) used by the tests and
the pre-GPU flythrough, so a sign error surfaces on the laptop, not after a
queue wait. Mass and inertia are inputs: in the sim they are READ BACK FROM
PHYSX (playbook rule 2), never assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch

from drone.math3d import heading_to_world, quat_to_rotmat, wrap_pi, yaw_of

GRAVITY = 9.81


@dataclass(frozen=True)
class ControllerGains:
    vel_gain_xy: float = 3.0          # 1/s
    vel_gain_z: float = 4.0           # 1/s
    max_tilt_deg: float = 35.0
    max_acc_z: float = 6.0            # m/s^2, up or down
    att_natural_freq: float = 25.0    # rad/s (roll/pitch)
    att_damping: float = 0.9
    yaw_natural_freq: float = 8.0     # rad/s
    yaw_damping: float = 1.0
    max_yaw_error_rad: float = 0.6    # yaw setpoint is kept within this of the actual yaw
    thrust_to_weight: float = 1.9     # Isaac Lab quadcopter demo value
    max_torque_nm: float = 0.01       # Isaac Lab demo moment_scale

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "ControllerGains":
        fields = set(cls.__dataclass_fields__)
        unknown = set(raw) - fields
        if unknown:
            raise KeyError(f"unknown controller keys {sorted(unknown)}; valid: {sorted(fields)}")
        return cls(**{k: float(v) for k, v in raw.items()})


class VelocityController:
    """Batched controller with a per-env yaw setpoint state."""

    def __init__(self, gains: ControllerGains, mass: torch.Tensor, inertia_diag: torch.Tensor, dt: float):
        """
        mass:         (N,) kg -- total articulation mass (body + rotors), from PhysX
        inertia_diag: (N, 3) kg m^2 -- body principal inertia, from PhysX
        dt:           physics step (s): the controller runs every physics step
        """
        self.g = gains
        self.mass = mass
        self.J = inertia_diag
        self.dt = dt
        n = mass.shape[0]
        self.yaw_sp = torch.zeros(n, device=mass.device)
        self.max_thrust = gains.thrust_to_weight * mass * GRAVITY
        w_rp, z_rp = gains.att_natural_freq, gains.att_damping
        w_y, z_y = gains.yaw_natural_freq, gains.yaw_damping
        dev = mass.device
        self.kR = torch.tensor([w_rp ** 2, w_rp ** 2, w_y ** 2], device=dev)
        self.kW = torch.tensor([2 * z_rp * w_rp, 2 * z_rp * w_rp, 2 * z_y * w_y], device=dev)
        self.max_tilt = math.radians(gains.max_tilt_deg)

    def reset(self, env_ids: torch.Tensor, yaw: torch.Tensor) -> None:
        self.yaw_sp[env_ids] = yaw

    def compute(
        self, quat_w: torch.Tensor, lin_vel_w: torch.Tensor, ang_vel_b: torch.Tensor, cmd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cmd (N, 4) = [vx, vy, vz, yaw_rate] in the heading frame (m/s, rad/s).

        Returns (force_b, torque_b), each (N, 3), in the BODY frame.
        """
        g = self.g
        R = quat_to_rotmat(quat_w)
        yaw = yaw_of(R)

        # ---- velocity loop -------------------------------------------------
        v_cmd_w = heading_to_world(yaw, cmd[:, :3])
        err = v_cmd_w - lin_vel_w
        # feed-forward: a heading-frame command rotates with the yaw rate, so its
        # world-frame derivative is w_z x v_cmd. Without it a forward+turn command
        # lags sideways by v * w / Kv (1 m/s at 3 m/s and 1 rad/s).
        a_ff = cmd[:, 3:4] * torch.stack([-v_cmd_w[:, 1], v_cmd_w[:, 0]], dim=-1)
        a_xy = g.vel_gain_xy * err[:, :2] + a_ff
        a_z = (g.vel_gain_z * err[:, 2]).clamp(-g.max_acc_z, g.max_acc_z)
        # tilt limit: horizontal accel <= (g + a_z) tan(max_tilt)
        a_xy_max = (GRAVITY + a_z).clamp(min=0.5) * math.tan(self.max_tilt)
        n_xy = torch.linalg.norm(a_xy, dim=-1).clamp(min=1e-9)
        a_xy = a_xy * torch.clamp(a_xy_max / n_xy, max=1.0)[:, None]
        f_des = self.mass[:, None] * torch.cat([a_xy, (a_z + GRAVITY)[:, None]], dim=-1)

        b3 = R[:, :, 2]
        thrust = (f_des * b3).sum(-1).clamp(min=0.0)
        thrust = torch.minimum(thrust, self.max_thrust)

        # ---- attitude setpoint --------------------------------------------
        self.yaw_sp = yaw + wrap_pi(self.yaw_sp + cmd[:, 3] * self.dt - yaw).clamp(
            -g.max_yaw_error_rad, g.max_yaw_error_rad
        )
        b3_des = f_des / torch.linalg.norm(f_des, dim=-1, keepdim=True).clamp(min=1e-9)
        b1c = torch.stack([torch.cos(self.yaw_sp), torch.sin(self.yaw_sp), torch.zeros_like(yaw)], dim=-1)
        b2_des = torch.linalg.cross(b3_des, b1c)
        b2_des = b2_des / torch.linalg.norm(b2_des, dim=-1, keepdim=True).clamp(min=1e-9)
        b1_des = torch.linalg.cross(b2_des, b3_des)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)   # columns

        # ---- attitude loop --------------------------------------------------
        E = torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des)
        e_R = 0.5 * torch.stack([E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]], dim=-1)
        w_des_des = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), cmd[:, 3]], dim=-1)
        w_des_b = torch.einsum("nji,njk,nk->ni", R, R_des, w_des_des)   # R^T R_des w_des
        e_W = ang_vel_b - w_des_b
        Jw = self.J * ang_vel_b
        torque = self.J * (-self.kR * e_R - self.kW * e_W) + torch.linalg.cross(ang_vel_b, Jw)
        torque = torque.clamp(-g.max_torque_nm, g.max_torque_nm)

        force = torch.zeros_like(torque)
        force[:, 2] = thrust
        return force, torque
