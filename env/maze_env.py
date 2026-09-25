"""Isaac Lab environment: a Crazyflie flies a config-built maze on velocity commands.

THIS IS THE ONLY MODULE (with eval/validate_sim.py and the record script)
THAT IMPORTS ISAAC SIM. Everything testable lives in pure modules and is
covered by tests/: geometry + collision (maze/geometry.py), potential
(maze/grid.py), reward (env/reward.py), controller (drone/controller.py),
observation layout (env/obs_layout.py), route tracking (maze/routes.py).
This file only moves state between PhysX and those functions.

Uncertain Isaac Lab 2.1 calls are marked ``# VERIFY ON ARC:`` and each one has
a probe in ``eval/validate_sim.py`` that measures the physical response
(playbook rule 1: no error does not mean it works).

Step order (Isaac Lab DirectRLEnv.step -- playbook rule 6):
    _pre_physics_step -> [ _apply_action -> sim.step -> scene.update ] x decimation
    -> _get_dones -> _get_rewards -> _reset_idx(done envs) -> _get_observations
Per-step state (collision, goal, potential) is computed in ``_get_dones``;
collision is also checked at EVERY physics sub-step inside ``_apply_action``,
so a fast drone cannot pass through a thin wall between policy steps unseen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from drone.controller import ControllerGains, VelocityController
from drone.math3d import quat_from_yaw, quat_to_rotmat, rot6d, world_to_body
from env.obs_layout import NUM_ACTIONS, build_layout
from env.reward import COMPONENTS, RewardConfig, compute_reward
from maze.geometry import MazeGeometry
from maze.grid import build_grid, geodesic_field, lookup
from maze.routes import RouteTracker, classify
from maze.spec import MazeSpec, Primitive, load_maze

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets import CRAZYFLIE_CFG  # noqa: E402

OUTCOME_RUNNING, OUTCOME_SUCCESS, OUTCOME_COLLISION, OUTCOME_TIMEOUT = 0, 1, 2, 3
OUTCOME_NAMES = {0: "running", 1: "success", 2: "collision", 3: "timeout"}


@dataclass
class EpisodeRecord:
    """One finished episode (eval/record only). All fields measured, none scored by hand."""

    env_id: int
    outcome: str
    time_s: float
    route: str                 # fast | slow | both | none   (maze/routes.py)
    path_length_m: float
    min_clearance_m: float
    mean_speed_mps: float
    reward: Dict[str, float]   # per-component episode sums


@configclass
class MazeDroneEnvCfg(DirectRLEnvCfg):
    decimation: int = 10
    episode_length_s: float = 40.0
    action_space: int = NUM_ACTIONS
    observation_space: int = 1          # overwritten from the ObsLayout
    state_space: int = 0
    rerender_on_reset: bool = False

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=24.0, replicate_physics=True)

    robot: ArticulationCfg = None       # type: ignore[assignment]
    camera: TiledCameraCfg = None       # type: ignore[assignment]
    contact: ContactSensorCfg = None    # type: ignore[assignment]

    raw: Dict[str, Any] = None          # the full resolved YAML (study settings live here)


def build_env_cfg(data: Dict[str, Any]) -> MazeDroneEnvCfg:
    """The single YAML -> Isaac Lab cfg translation point."""
    sim_y, env_y, drone_y, cam_y = data["sim"], data["env"], data["drone"], data["camera"]
    layout = build_layout(data)
    out = MazeDroneEnvCfg()
    out.decimation = int(sim_y["decimation"])
    out.episode_length_s = float(env_y["episode_length_s"])
    out.observation_space = layout.total_dim
    out.rerender_on_reset = bool(env_y["rerender_on_reset"])
    out.seed = int(data["seed"])   # playbook 5.13: unset => "Seed not set" and nondeterminism
    out.raw = data

    # PhysxCfg: only real fields (playbook 5.1); an unknown key fails here with the valid names.
    physx_y = dict(sim_y["physx"])
    valid = set(getattr(sim_utils.PhysxCfg, "__dataclass_fields__", {}))
    unknown = sorted(set(physx_y) - valid)
    if unknown:
        raise ValueError(f"sim.physx keys not in PhysxCfg: {unknown}. Valid: {sorted(valid)}")
    out.sim = SimulationCfg(
        dt=float(sim_y["dt"]),
        render_interval=int(sim_y["decimation"]),
        device=str(sim_y["device"]),
        use_fabric=bool(sim_y["use_fabric"]),
        physx=sim_utils.PhysxCfg(**physx_y),
    )
    out.scene = InteractiveSceneCfg(
        num_envs=int(env_y["num_envs"]), env_spacing=float(env_y["env_spacing"]), replicate_physics=True
    )

    # Robot: Isaac Lab's own Crazyflie cfg (the quadcopter demo's dynamics base).
    spawn = CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True)
    if drone_y.get("usd_path"):
        spawn = spawn.replace(usd_path=str(drone_y["usd_path"]))
    out.robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn)

    # PhysX contact CROSS-CHECK on every drone body (termination itself is analytic).
    # Unfiltered net force is fine here: a flying drone touches nothing, so ANY
    # contact is a crash (unlike a wheeled robot resting on the ground, playbook 5.5).
    out.contact = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=0,
                                   track_air_time=False)

    if data["observation"]["camera_enabled"]:
        # VERIFY ON ARC: the camera prim must sit under the body LINK so it moves
        # with the drone (validate_sim frames from different poses must differ).
        out.camera = TiledCameraCfg(
            prim_path=f"/World/envs/env_.*/Robot/{drone_y['body_name']}/front_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(float(v) for v in cam_y["offset_pos"]),
                                            rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(cam_y["focal_length"]),
                horizontal_aperture=float(cam_y["horizontal_aperture"]),
                clipping_range=tuple(float(v) for v in cam_y["clipping_range"]),
            ),
            width=int(cam_y["width"]),
            height=int(cam_y["height"]),
        )
    else:
        out.camera = None
    return out


def _yaw_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


class MazeDroneEnv(DirectRLEnv):
    cfg: MazeDroneEnvCfg

    def __init__(self, cfg: MazeDroneEnvCfg, render_mode: Optional[str] = None, **kwargs: Any):
        self._raw = cfg.raw
        self.maze: MazeSpec = load_maze(self._raw["maze"]["config"])
        self.layout = build_layout(self._raw)
        self.reward_cfg = RewardConfig.from_config(self._raw)
        self.policy_dt = float(self._raw["sim"]["dt"]) * int(self._raw["sim"]["decimation"])
        self.radius = float(self._raw["drone"]["collision_radius_m"])
        # Probe/eval switches (never set during training):
        self.terminate_on_collision = True
        self.record_episodes = False
        self.forced_start: Optional[Tuple[torch.Tensor, torch.Tensor]] = None   # (pos_local (N,3), yaw (N,))
        super().__init__(cfg, render_mode, **kwargs)
        self._init_buffers()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact = ContactSensor(self.cfg.contact)
        self.scene.sensors["contact"] = self.contact
        self.camera = None
        if self.cfg.camera is not None:
            self.camera = TiledCamera(self.cfg.camera)
            self.scene.sensors["camera"] = self.camera

        # Maze geometry is authored ONCE into env_0 and cloned. Flat prim names:
        # spawners create only the leaf prim (playbook 5.7).
        # VERIFY ON ARC: static colliders under env_0 replicate to every env
        # (validate_sim: forced-crash probe must see PhysX contacts in env N-1,
        # and camera frames from env N-1 must show walls).
        off = self.maze.center_offset
        for prim in list(self.maze.primitives) + self.maze.perimeter_walls() + self._pads_and_floor():
            self._spawn_primitive(prim, off)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        dome = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        dome.func("/World/DomeLight", dome)
        # a directional light gives wall faces different shading -> corners readable at 64 px
        sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=0.5, color=(1.0, 0.97, 0.9))
        q = torch.tensor([0.87, 0.2, 0.3, 0.33])
        sun.func("/World/SunLight", sun, orientation=tuple((q / q.norm()).tolist()))

    def _pads_and_floor(self) -> List[Primitive]:
        X, Y, _ = self.maze.bounds
        t = self.maze.perimeter_thickness_m
        sx, sy = self.maze.start_pos[:2]
        gx, gy = self.maze.goal_pos[:2]
        return [
            # the floor is a real collider (a crash into the ground is a crash)
            Primitive("floor", "box", (X / 2, Y / 2, -0.05), (X + 2 * t, Y + 2 * t, 0.1),
                      color=self.maze.floor_color, tag="floor"),
            Primitive("start_pad", "box", (sx, sy, 0.012), (1.2, 1.2, 0.01), color=(0.1, 0.7, 0.2),
                      tag="decor", collision=False),
            Primitive("goal_pad", "box", (gx, gy, 0.012), (1.4, 1.4, 0.01), color=(0.95, 0.85, 0.1),
                      tag="decor", collision=False),
        ]

    def _spawn_primitive(self, p: Primitive, off: Tuple[float, float, float]) -> None:
        common = dict(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(p.color), roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True) if p.collision else None,
        )
        if p.type == "box":
            cfg = sim_utils.CuboidCfg(size=tuple(p.size), **common)
            orient = _yaw_quat(p.yaw_deg)
        else:
            cfg = sim_utils.CylinderCfg(radius=float(p.radius), height=float(p.height), axis="Z", **common)
            orient = (1.0, 0.0, 0.0, 0.0)
        pos = (p.pos[0] + off[0], p.pos[1] + off[1], p.pos[2] + off[2])
        cfg.func(f"/World/envs/env_0/Maze_{p.id}", cfg, translation=pos, orientation=orient)

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    def _init_buffers(self) -> None:
        n, dev = self.num_envs, self.device
        self.geom = MazeGeometry(self.maze, dev)
        self._offset = torch.tensor(self.maze.center_offset, device=dev)
        self.goal_local = torch.tensor(self.maze.goal_pos, device=dev)

        # Shaping potential field (geodesic) -- built on the CPU once, looked up on GPU.
        rw = self._raw["reward"]
        self.potential_kind = str(rw["potential"])
        if self.potential_kind == "geodesic":
            grid = build_grid(MazeGeometry(self.maze, "cpu"), float(rw["geodesic_grid_res_m"]), self.radius)
            field_, _ = geodesic_field(grid, self.maze.goal_pos, self.maze.goal_radius_m)
            grid.free, grid.clearance = grid.free.to(dev), grid.clearance.to(dev)
            self._grid, self._field = grid, field_.to(dev)

        self._body_id = self.robot.find_bodies(self._raw["drone"]["body_name"])[0]
        # Mass / inertia READ BACK FROM PHYSX (playbook rule 2), never assumed.
        # VERIFY ON ARC: get_masses() (N, B) and get_inertias() (N, B, 9) are CPU tensors.
        masses = self.robot.root_physx_view.get_masses().to(dev)
        inertias = self.robot.root_physx_view.get_inertias().to(dev)
        self.mass = masses.sum(dim=1)
        b = self._body_id[0]
        self.inertia_diag = inertias[:, b, [0, 4, 8]]
        self.controller = VelocityController(
            ControllerGains.from_config(self._raw["controller"]), self.mass, self.inertia_diag, self.physics_dt
        )

        a = self._raw["action"]
        ranges = torch.tensor([a["vx_range"], a["vy_range"], a["vz_range"], a["yaw_rate_range"]], device=dev)
        self._act_mid = ranges.mean(dim=1)
        self._act_half = (ranges[:, 1] - ranges[:, 0]) / 2.0

        self.actions = torch.zeros(n, NUM_ACTIONS, device=dev)
        self.last_actions = torch.zeros(n, NUM_ACTIONS, device=dev)
        self.commands = torch.zeros(n, NUM_ACTIONS, device=dev)
        self._force = torch.zeros(n, 1, 3, device=dev)
        self._torque = torch.zeros(n, 1, 3, device=dev)

        self.collided_substep = torch.zeros(n, dtype=torch.bool, device=dev)
        self.collided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.physx_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dist_prev = torch.zeros(n, device=dev)
        self.dist_next = torch.zeros(n, device=dev)
        self.min_clearance = torch.full((n,), float("inf"), device=dev)
        self.path_length = torch.zeros(n, device=dev)
        self.prev_pos = torch.zeros(n, 3, device=dev)
        self.outcome = torch.zeros(n, dtype=torch.long, device=dev)
        self.ep_sums = {k: torch.zeros(n, device=dev) for k in COMPONENTS}
        self.routes = RouteTracker(self.maze, n, dev)
        self._records: List[EpisodeRecord] = []
        # diagnostics (kept on GPU; .item() only when read)
        self.diag = {k: torch.zeros((), device=dev, dtype=torch.long)
                     for k in ("analytic_hits", "physx_hits", "physx_without_analytic", "nan_states")}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def pos_local(self) -> torch.Tensor:
        """Drone position in the MAZE-local frame (N, 3)."""
        return self.robot.data.root_pos_w - self.scene.env_origins - self._offset

    def potential_distance(self, pos_local: torch.Tensor) -> torch.Tensor:
        if self.potential_kind == "geodesic":
            return lookup(self._field, self._grid, pos_local)
        return torch.linalg.norm(pos_local - self.goal_local, dim=-1)

    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self._act_mid + actions.clamp(-1.0, 1.0) * self._act_half

    def unscale_commands(self, cmd: torch.Tensor) -> torch.Tensor:
        """Physical command -> policy action space (used by the scripted pilot)."""
        return ((cmd - self._act_mid) / self._act_half).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # rsl_rl appends extras["log"] on EVERY step it is present; clear it so
        # a step without resets does not re-log the previous step's episodes.
        self.extras.pop("log", None)
        self.last_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.commands = self.scale_actions(self.actions)

    def _apply_action(self) -> None:
        # sub-step collision check on the state PhysX just produced
        clr = self.geom.clearance(self.pos_local())
        self.collided_substep |= clr < self.radius
        self.min_clearance = torch.minimum(self.min_clearance, clr)

        d = self.robot.data
        force, torque = self.controller.compute(d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_b, self.commands)
        self._force[:, 0], self._torque[:, 0] = force, torque
        self.robot.set_external_force_and_torque(self._force, self._torque, body_ids=self._body_id)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self.pos_local()
        bad = ~torch.isfinite(pos).all(dim=-1)
        pos = torch.where(bad[:, None], self.prev_pos, pos)
        clr = self.geom.clearance(pos)
        self.min_clearance = torch.minimum(self.min_clearance, clr)
        analytic = self.collided_substep | (clr < self.radius) | bad

        forces = self.contact.data.net_forces_w   # (N, B, 3)
        self.physx_contact = (torch.linalg.norm(forces, dim=-1) >
                              float(self._raw["drone"]["contact_force_threshold_n"])).any(dim=-1)
        self.diag["analytic_hits"] += analytic.sum()
        self.diag["physx_hits"] += self.physx_contact.sum()
        self.diag["physx_without_analytic"] += (self.physx_contact & ~analytic).sum()
        self.diag["nan_states"] += bad.sum()

        self.collided = analytic if self.terminate_on_collision else torch.zeros_like(analytic)
        self.reached = (torch.linalg.norm(pos - self.goal_local, dim=-1) < self.maze.goal_radius_m) & ~analytic
        self.dist_next = self.potential_distance(pos)
        self.path_length += torch.linalg.norm(pos - self.prev_pos, dim=-1)
        self.prev_pos = pos
        self.routes.update(pos)
        self.collided_substep.zero_()

        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        code = lambda c: torch.full_like(self.outcome, c)   # noqa: E731
        self.outcome = torch.where(
            self.reached, code(OUTCOME_SUCCESS),
            torch.where(self.collided, code(OUTCOME_COLLISION),
                        torch.where(timed_out, code(OUTCOME_TIMEOUT), code(OUTCOME_RUNNING))),
        )
        return self.reached | self.collided, timed_out

    def _get_rewards(self) -> torch.Tensor:
        total, comps = compute_reward(self.dist_prev, self.dist_next, self.reached, self.collided, self.reward_cfg)
        for k, v in comps.items():
            self.ep_sums[k] += v
        self.dist_prev = self.dist_next.clone()
        return total

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        d = self.robot.data
        R = quat_to_rotmat(d.root_quat_w)
        o = self._raw["observation"]
        goal_b = world_to_body(R, self.goal_local - self.pos_local())
        parts = []
        if self.layout.image_shape is not None:
            if o["zero_image"]:
                parts.append(torch.zeros(self.num_envs, self.layout.image_dim, device=self.device))
            else:
                rgb = self.camera.data.output["rgb"][..., :3]            # (N, H, W, 3) uint8
                parts.append((rgb.float() / 255.0).permute(0, 3, 1, 2).reshape(self.num_envs, -1))
        parts += [
            d.root_lin_vel_b * float(o["lin_vel_scale"]),
            d.root_ang_vel_b * float(o["ang_vel_scale"]),
            rot6d(R),
            goal_b * float(o["goal_vec_scale"]),
            self.actions,
        ]
        obs = torch.cat(parts, dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Optional[Sequence[int]]) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES   # type: ignore[attr-defined]
        if hasattr(self, "ep_sums"):   # the first reset runs inside super().__init__
            self._log_finished(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, "ep_sums"):
            return
        n = len(env_ids)
        dev = self.device

        if self.forced_start is not None:
            pos_l, yaw = self.forced_start[0][env_ids], self.forced_start[1][env_ids]
        else:
            noise = self._raw["env"]["start_noise"]
            pos_l = torch.tensor(self.maze.start_pos, device=dev).repeat(n, 1)
            pos_l += (torch.rand(n, 3, device=dev) * 2 - 1) * float(noise["pos_m"])
            yaw = torch.full((n,), math.radians(self.maze.start_yaw_deg), device=dev)
            yaw += (torch.rand(n, device=dev) * 2 - 1) * math.radians(float(noise["yaw_deg"]))

        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] = pos_l + self._offset + self.scene.env_origins[env_ids]
        root[:, 3:7] = quat_from_yaw(yaw)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jp = self.robot.data.default_joint_pos[env_ids]
        jv = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)
        self.robot.reset(env_ids)
        self.controller.reset(env_ids, yaw)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self.collided_substep[env_ids] = False
        self.min_clearance[env_ids] = float("inf")
        self.path_length[env_ids] = 0.0
        self.prev_pos[env_ids] = pos_l
        self.dist_prev[env_ids] = self.potential_distance(pos_l)
        for v in self.ep_sums.values():
            v[env_ids] = 0.0
        self.routes.reset(env_ids)

    def _log_finished(self, env_ids: torch.Tensor) -> None:
        """Episode statistics for rsl_rl / W&B (extras['log']) and, in eval, full records."""
        done = self.outcome[env_ids] != OUTCOME_RUNNING
        ids = env_ids[done]
        if len(ids) == 0:
            return
        out = self.outcome[ids]
        # DirectRLEnv increments episode_length_buf BEFORE _get_dones: it already counts this step
        steps = self.episode_length_buf[ids].float()
        taken = self.routes.taken()[ids]
        log: Dict[str, Any] = {}
        for k, v in self.ep_sums.items():
            log[f"Episode_Reward/{k}"] = v[ids].mean()
            log[f"Episode_RewardAbs/{k}"] = v[ids].abs().mean()
        for code, name in ((OUTCOME_SUCCESS, "success"), (OUTCOME_COLLISION, "collision"), (OUTCOME_TIMEOUT, "timeout")):
            log[f"Episode_Termination/{name}"] = (out == code).float().mean()
        succ = out == OUTCOME_SUCCESS
        log["Metrics/route_fast"] = (taken[:, 0] & ~taken[:, 1]).float().mean()
        log["Metrics/route_slow"] = (taken[:, 1] & ~taken[:, 0]).float().mean()
        if bool(succ.any()):
            log["Metrics/time_to_goal_s"] = (steps[succ] * self.policy_dt).mean()
            log["Metrics/fast_given_success"] = (taken[succ, 0] & ~taken[succ, 1]).float().mean()
        log["Metrics/min_clearance_m"] = self.min_clearance[ids].clamp(max=5.0).mean()
        self.extras["log"] = log

        if self.record_episodes:
            t = (steps * self.policy_dt).tolist()
            pl = self.path_length[ids].tolist()
            for j, i in enumerate(ids.tolist()):
                f, s = taken[j].tolist()
                self._records.append(EpisodeRecord(
                    env_id=i,
                    outcome=OUTCOME_NAMES[int(out[j])],
                    time_s=t[j],
                    route=classify(f, s),
                    path_length_m=pl[j],
                    min_clearance_m=float(self.min_clearance[i]),
                    mean_speed_mps=pl[j] / max(t[j], 1e-6),
                    reward={k: float(v[i]) for k, v in self.ep_sums.items()},
                ))
        self.outcome[ids] = OUTCOME_RUNNING

    def drain_records(self) -> List[EpisodeRecord]:
        out, self._records = self._records, []
        return out

    def diagnostics(self) -> Dict[str, int]:
        return {k: int(v.item()) for k, v in self.diag.items()}
