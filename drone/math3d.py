"""Small batched rotation helpers (torch). Quaternions are (w, x, y, z), as in Isaac Lab."""

from __future__ import annotations

import math

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz -> (N, 3, 3) body-to-world rotation."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw (rad) -> (N, 4) wxyz."""
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def yaw_of(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> (N,) heading of the body x-axis projected on the ground plane."""
    return torch.atan2(R[:, 1, 0], R[:, 0, 0])


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def rot6d(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> (N, 6): first two COLUMNS (continuous rotation rep, Zhou et al. 2019)."""
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)


def world_to_body(R: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors (N, 3) into the body frame."""
    return torch.einsum("nji,nj->ni", R, v_w)


def heading_to_world(yaw: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
    """Heading (gravity-aligned, yawed) frame -> world. v_h: (N, 3)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v_h[:, 0] - s * v_h[:, 1], s * v_h[:, 0] + c * v_h[:, 1], v_h[:, 2]], dim=-1)


def world_to_heading(yaw: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1], v_w[:, 2]], dim=-1)
