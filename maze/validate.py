"""The pre-training environment validation checklist (CLAUDE.md section 6), pure half.

Runs on a laptop or the ARC login node in seconds -- no Isaac, no GPU, no
queue (playbook rule 9). Covers:

  6.1 spawn validity       start (and goal) clear of every collider by a margin
  6.2 reachability         voxel pathfinding start -> goal, independent of the
                           authored waypoints
  6.3 route length gap     slow / fast >= min ratio, from WAYPOINT path length;
                           plus an "unintended shortcut" check: the geodesic
                           shortest path must not beat the authored fast route
  6.4 scale / units        every route passable at the drone's physical radius
                           (+margin); walls thick enough; slow route really
                           safe (clearance, altitude) and fast route really
                           risky (narrow gap / vertical shortcut / weave)
  6.5 flythrough (model)   the scripted pilot + real controller on a rigid-body
                           model flies each route at design speed, checked for
                           collision at EVERY physics step

6.5 is repeated in Isaac Sim, and 6.6 (visual distinguishability) can only be
done there -- see ``eval/validate_sim.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import torch

from drone.controller import ControllerGains, VelocityController
from drone.dynamics import QuadrotorModel, RigidBodyState
from drone.math3d import quat_from_yaw, quat_to_rotmat, yaw_of
from drone.pilot import WaypointPilot
from maze.geometry import MazeGeometry
from maze.grid import build_grid, geodesic_field, lookup, reachable
from maze.routes import classify_path, densify
from maze.spec import MazeSpec


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name:<34} {self.detail}"


@dataclass
class ValidationReport:
    maze: str
    checks: List[Check]
    metrics: Dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def text(self) -> str:
        head = f"== {self.maze}: {'ALL PASS' if self.passed else 'FAILED'} =="
        return "\n".join([head] + [c.line() for c in self.checks])


def _route_features(spec: MazeSpec, geom: MazeGeometry, kind: str, v: Mapping[str, Any]) -> Dict[str, Any]:
    route = spec.route(kind)
    pts = densify(route.waypoints, 0.02)
    clr = geom.clearance(pts)
    z = pts[:, 2]
    hanging = [p for p in spec.colliders() if p.tag == "hanging"]
    near = 0
    for p in hanging:
        c = torch.tensor(p.pos)
        d = torch.linalg.norm(pts - c, dim=-1).min().item() - max(p.half_extents)
        near += int(d < float(v["weave_distance_m"]))
    return {
        "length_m": route.length_m,
        "min_clearance_m": float(clr.min()),
        "z_min_m": float(z.min()),
        "z_max_m": float(z.max()),
        "hanging_near": near,
        "argmin_point": [round(float(x), 2) for x in pts[int(clr.argmin())]],
    }


def model_flythrough(spec: MazeSpec, geom: MazeGeometry, kind: str, radius: float,
                     gains: ControllerGains, physics_dt: float, timeout_s: float,
                     speed: float | None = None) -> Dict[str, Any]:
    """Fly one route with the scripted pilot on the rigid-body model. Collision
    is checked at every physics step, exactly as the env does."""
    route = spec.route(kind)
    v = speed or route.design_speed_mps
    model = QuadrotorModel(1)
    ctrl = VelocityController(gains, model.mass, model.J, physics_dt)
    pilot = WaypointPilot(route.waypoints, 1, speed=v)
    yaw0 = torch.tensor([math.radians(spec.start_yaw_deg)])
    s = RigidBodyState(torch.tensor([spec.start_pos]), quat_from_yaw(yaw0), torch.zeros(1, 3), torch.zeros(1, 3))
    ctrl.reset(torch.tensor([0]), yaw0)
    goal = torch.tensor(spec.goal_pos)
    decimation = max(1, int(round(0.05 / physics_dt)))   # pilot at ~20 Hz, like the policy
    traj = [s.pos[0].clone()]
    min_clr = float("inf")
    cmd = torch.zeros(1, 4)
    for i in range(int(timeout_s / physics_dt)):
        if i % decimation == 0:
            cmd = pilot.act(s.pos, yaw_of(quat_to_rotmat(s.quat)))
        f, t = ctrl.compute(s.quat, s.vel, s.ang_vel_b, cmd)
        s = model.step(s, f, t, physics_dt)
        clr = float(geom.clearance(s.pos)[0])
        min_clr = min(min_clr, clr)
        traj.append(s.pos[0].clone())
        if clr < radius:
            return {"outcome": "collision", "t_s": i * physics_dt, "min_clearance_m": min_clr,
                    "where": [round(float(x), 2) for x in s.pos[0]],
                    "near": geom.nearest_primitive(s.pos[0]), "traj": torch.stack(traj)}
        if torch.linalg.norm(s.pos[0] - goal) < spec.goal_radius_m:
            return {"outcome": "success", "t_s": i * physics_dt, "min_clearance_m": min_clr,
                    "traj": torch.stack(traj)}
    return {"outcome": "timeout", "t_s": timeout_s, "min_clearance_m": min_clr, "traj": torch.stack(traj)}


def validate_maze(spec: MazeSpec, cfg: Mapping[str, Any]) -> ValidationReport:
    """``cfg`` is the full train config dict (drone, controller, validation, sim, env)."""
    v = cfg["validation"]
    r = float(cfg["drone"]["collision_radius_m"])
    geom = MazeGeometry(spec)
    checks: List[Check] = []
    metrics: Dict[str, Any] = {}

    # ---- 6.1 spawn validity ------------------------------------------------
    start_clr = float(geom.clearance(torch.tensor([spec.start_pos]))[0])
    goal_clr = float(geom.clearance(torch.tensor([spec.goal_pos]))[0])
    need = r + float(v["spawn_margin_m"]) + float(cfg["env"]["start_noise"]["pos_m"])
    checks.append(Check("6.1 spawn clear of colliders", start_clr >= need,
                        f"start clearance {start_clr:.2f} m (need >= {need:.2f}: radius + margin + spawn noise)"))
    checks.append(Check("6.1 goal in free space", goal_clr >= r,
                        f"goal clearance {goal_clr:.2f} m (need >= {r:.2f})"))
    metrics["start_clearance_m"] = start_clr

    # ---- 6.2 reachability (voxel pathfinding) ---------------------------------
    grid = build_grid(geom, float(v["grid_res_m"]), r)
    field_, connected = geodesic_field(grid, spec.goal_pos, spec.goal_radius_m)
    ok = reachable(grid, connected, spec.start_pos)
    geo = float(lookup(field_, grid, torch.tensor([spec.start_pos]))[0])
    checks.append(Check("6.2 goal reachable (voxel search)", ok,
                        f"geodesic start->goal {geo:.1f} m at {grid.res} m voxels" if ok
                        else "no free-space path at the drone radius"))
    metrics["geodesic_start_to_goal_m"] = geo
    metrics["free_voxel_fraction"] = float(grid.free.float().mean())

    # ---- 6.3 route length gap --------------------------------------------------
    fast, slow = spec.route("fast"), spec.route("slow")
    ratio = slow.length_m / fast.length_m
    min_ratio = float(v["min_route_ratio"])
    checks.append(Check("6.3 slow/fast path-length ratio", ratio >= min_ratio,
                        f"slow {slow.length_m:.1f} m / fast {fast.length_m:.1f} m = {ratio:.2f} (need >= {min_ratio})"))
    tol = float(v["shortcut_tolerance"])
    # geodesic overestimates by <= ~8% (26-neighbour metric), so a geodesic well
    # BELOW the fast route means a genuinely shorter, unintended path exists.
    checks.append(Check("6.3 no unintended shortcut", geo >= tol * fast.length_m,
                        f"geodesic {geo:.1f} m vs fast route {fast.length_m:.1f} m (need >= {tol:.2f}x)"))
    metrics.update(fast_length_m=fast.length_m, slow_length_m=slow.length_m, length_ratio=ratio)

    # The trade-off only exists if speed REQUIRES risk: the shortest path that
    # keeps the "safe" clearance everywhere must be about as long as the slow
    # route. If a wide, short path exists, the fast route is not risky -- it is
    # just a worse line -- and the route-choice metric measures nothing.
    safe_r = float(v["slow_min_clearance_m"])
    safe_grid = build_grid(geom, float(v["grid_res_m"]), safe_r)
    safe_field, safe_conn = geodesic_field(safe_grid, spec.goal_pos, spec.goal_radius_m)
    safe_len = float(lookup(safe_field, safe_grid, torch.tensor([spec.start_pos]))[0]) \
        if reachable(safe_grid, safe_conn, spec.start_pos) else float("inf")
    stol = float(v["safe_path_tolerance"])
    checks.append(Check("6.3 speed requires risk", safe_len >= stol * slow.length_m,
                        f"shortest path with >= {safe_r} m clearance {safe_len:.1f} m vs slow route "
                        f"{slow.length_m:.1f} m (need >= {stol:.2f}x)"))
    metrics["safe_geodesic_m"] = safe_len

    # ---- 6.4 scale / units / route character -------------------------------------
    margin = float(v["route_clearance_margin_m"])
    ff = _route_features(spec, geom, "fast", v)
    sf = _route_features(spec, geom, "slow", v)
    metrics["fast"], metrics["slow"] = ff, sf
    for kind, feat in (("fast", ff), ("slow", sf)):
        checks.append(Check(f"6.4 {kind} route passable", feat["min_clearance_m"] >= r + margin,
                            f"min clearance {feat['min_clearance_m']:.2f} m at {feat['argmin_point']} "
                            f"(need >= radius {r} + margin {margin})"))
    checks.append(Check("6.4 slow route is safe", sf["min_clearance_m"] >= float(v["slow_min_clearance_m"])
                        and sf["z_max_m"] <= float(v["slow_max_altitude_m"]),
                        f"clearance {sf['min_clearance_m']:.2f} m (>= {v['slow_min_clearance_m']}), "
                        f"max altitude {sf['z_max_m']:.2f} m (<= {v['slow_max_altitude_m']})"))
    features = []
    if ff["min_clearance_m"] <= float(v["narrow_gap_clearance_m"]):
        features.append("narrow-gap")
    if ff["z_max_m"] - min(ff["z_min_m"], spec.start_pos[2]) >= float(v["vertical_shortcut_dz_m"]):
        features.append("vertical-shortcut")
    if ff["hanging_near"] >= int(v["weave_obstacle_count"]):
        features.append(f"weave({ff['hanging_near']})")
    checks.append(Check("6.4 fast route is risky", bool(features) and ff["min_clearance_m"] < sf["min_clearance_m"],
                        f"features: {features or 'NONE'}; clearance {ff['min_clearance_m']:.2f} m "
                        f"vs slow {sf['min_clearance_m']:.2f} m"))
    metrics["fast_features"] = features
    thin = [p.id for p in spec.colliders() if p.min_thickness() < float(v["min_wall_thickness_m"])]
    checks.append(Check("6.4 collider thickness", not thin,
                        f"all >= {v['min_wall_thickness_m']} m" if not thin else f"too thin: {thin}"))
    X, Y, Z = spec.bounds
    outside = [
        p.id for p in spec.primitives
        if any(p.pos[i] < -1e-6 or p.pos[i] > (X, Y, Z)[i] + 1e-6 for i in range(3))
    ]
    checks.append(Check("6.4 primitives inside bounds", not outside,
                        "ok" if not outside else f"centre outside bounds: {outside}"))

    # signatures must classify each authored route as itself
    for kind in ("fast", "slow"):
        got = classify_path(spec, densify(spec.route(kind).waypoints, 0.05))
        checks.append(Check(f"6.4 {kind} signature self-check", got == kind,
                            f"authored {kind} route classified as '{got}'"))

    # ---- 6.5 flythrough on the rigid-body model ------------------------------
    gains = ControllerGains.from_config(cfg["controller"])
    dt = float(cfg["sim"]["dt"])
    fly: Dict[str, Any] = {}
    for kind in ("fast", "slow"):
        res = model_flythrough(spec, geom, kind, r, gains, dt, float(cfg["env"]["episode_length_s"]))
        fly[kind] = res
        got = classify_path(spec, res["traj"][::10])
        ok = res["outcome"] == "success" and got == kind
        extra = f" at {res.get('where')} near '{res.get('near')}'" if res["outcome"] == "collision" else ""
        checks.append(Check(f"6.5 model flythrough ({kind})", ok,
                            f"{res['outcome']} in {res['t_s']:.1f} s at {spec.route(kind).design_speed_mps} m/s, "
                            f"min clearance {res['min_clearance_m']:.2f} m, classified '{got}'{extra}"))
    metrics["flythrough"] = {k: {kk: vv for kk, vv in d.items() if kk != "traj"} for k, d in fly.items()}

    report = ValidationReport(spec.name, checks, metrics)
    report.trajectories = {k: d["traj"] for k, d in fly.items()}   # type: ignore[attr-defined]
    report.grid, report.field = grid, field_                     # type: ignore[attr-defined]
    return report
