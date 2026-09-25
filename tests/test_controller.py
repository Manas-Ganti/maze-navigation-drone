"""Controller + rigid-body model: signs, tracking, yaw, and thrust saturation."""

import math

import torch

from drone.controller import ControllerGains, VelocityController
from drone.dynamics import QuadrotorModel, RigidBodyState
from drone.math3d import quat_from_yaw, quat_to_rotmat, world_to_heading, yaw_of

DT = 0.005


def fly(cmd, yaw0=0.0, seconds=3.0, gains=ControllerGains()):
    n = cmd.shape[0]
    m = QuadrotorModel(n)
    c = VelocityController(gains, m.mass, m.J, DT)
    y = torch.full((n,), yaw0)
    s = RigidBodyState(torch.tensor([[0.0, 0.0, 2.0]]).repeat(n, 1), quat_from_yaw(y), torch.zeros(n, 3),
                       torch.zeros(n, 3))
    c.reset(torch.arange(n), y)
    for _ in range(int(seconds / DT)):
        f, t = c.compute(s.quat, s.vel, s.ang_vel_b, cmd)
        s = m.step(s, f, t, DT)
    return s


def test_hover_holds_altitude():
    s = fly(torch.zeros(1, 4))
    assert abs(s.pos[0, 2].item() - 2.0) < 0.02 and s.vel.norm() < 0.02


def test_heading_frame_velocity_tracking_any_yaw():
    for yaw0 in (0.0, 1.2, -2.5):
        cmd = torch.tensor([[2.0, -1.0, 0.5, 0.0]])
        s = fly(cmd, yaw0)
        v_h = world_to_heading(yaw_of(quat_to_rotmat(s.quat)), s.vel)
        assert torch.allclose(v_h, cmd[:, :3], atol=0.05), (yaw0, v_h)


def test_forward_means_camera_forward():
    s = fly(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), yaw0=math.pi / 2)
    assert s.vel[0, 1] > 0.9 and abs(s.vel[0, 0]) < 0.05   # yaw 90 deg: body +x is world +y


def test_yaw_rate_and_turning_flight():
    s = fly(torch.tensor([[3.0, 0.0, 0.0, 1.0]]))
    v_h = world_to_heading(yaw_of(quat_to_rotmat(s.quat)), s.vel)
    assert abs(s.ang_vel_b[0, 2].item() - 1.0) < 0.1
    assert abs(v_h[0, 0].item() - 3.0) < 0.2 and abs(v_h[0, 1].item()) < 0.2   # feed-forward removes the lag


def test_tilt_limit_respected():
    g = ControllerGains()
    n, m = 1, QuadrotorModel(1)
    c = VelocityController(g, m.mass, m.J, DT)
    s = RigidBodyState(torch.tensor([[0.0, 0.0, 2.0]]), quat_from_yaw(torch.zeros(1)), torch.zeros(1, 3), torch.zeros(1, 3))
    c.reset(torch.arange(n), torch.zeros(1))
    worst = 0.0
    for _ in range(int(2.0 / DT)):
        f, t = c.compute(s.quat, s.vel, s.ang_vel_b, torch.tensor([[4.0, 0.0, 0.0, 0.0]]))
        s = m.step(s, f, t, DT)
        worst = max(worst, math.degrees(math.acos(quat_to_rotmat(s.quat)[0, 2, 2].clamp(-1, 1).item())))
    assert worst < g.max_tilt_deg + 5.0


def test_thrust_never_exceeds_thrust_to_weight():
    m = QuadrotorModel(1)
    c = VelocityController(ControllerGains(), m.mass, m.J, DT)
    f, _ = c.compute(quat_from_yaw(torch.zeros(1)), torch.tensor([[0.0, 0.0, -5.0]]), torch.zeros(1, 3),
                     torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    assert f[0, 2] <= ControllerGains().thrust_to_weight * m.mass[0] * 9.81 + 1e-9
