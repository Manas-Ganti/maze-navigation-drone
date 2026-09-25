"""The in-sim half of the section 6 checklist + Phase 0 bring-up probes. Run BEFORE training.

    # on ARC (arc/validate.slurm runs both modes):
    python eval/validate_sim.py --headless --mode checks --set maze.config=configs/mazes/maze_a_window.yaml
    python eval/validate_sim.py --headless --mode throughput

Every probe FORCES a condition and measures the physical response, read back
from PhysX (playbook rules 1-2) -- "no error" is not evidence:

  identity      body/joint names, PhysX mass + inertia, measured collision footprint <= collision radius (6.4)
  hover         spawn at the start pose, zero command for 2 s: no contact, no drift, no tilt (6.1)
  controller    +vx / +vy / +vz / +yaw_rate steps from the most open point: sign, tracking, cross-talk
  crash         fly into geometry at 3 m/s with termination OFF: the analytic check fires in every env,
                PhysX registers contact in every env (incl. the LAST env: maze colliders replicated),
                and the drone never ends up inside a wall (no tunnelling) (6.5, s8)
  flythrough    scripted pilot flies BOTH authored routes at design speed in the real sim (6.5)
  reward audit  per-component episode sums from the flythrough + random flight; flags any term that
                dominates the others by > 10x (CLAUDE.md s4)
  frames        camera frames from points along each route -> a PNG grid for the human
                distinguishability check (6.6) + an automated "does the camera move" check
  throughput    env-steps/s at the training num_envs, and the wall-clock that implies (s7)

Output: results/validate_sim/<maze>/report.json, frames_*.png. Exit code 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.app import add_common_args, launch, load_run_config, make_env, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--mode", choices=["checks", "throughput"], default="checks")
    ap.add_argument("--probe-envs", type=int, default=32, help="num_envs for --mode checks")
    ap.add_argument("--throughput-steps", type=int, default=300)
    ap.add_argument("--out", default="results/validate_sim")
    return ap.parse_args()


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30} {detail}", flush=True)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def measure_footprint(prim_path: str) -> float | None:
    """Farthest collision-shape corner from the robot root (3D, metres), from the live stage.
    Extents from each shape's own attributes (ComputeExtentFromPlugins) -- authored
    extents can be stale (playbook 4.1)."""
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.XformCache()
        to_root = xf.GetLocalToWorldTransform(root).GetInverse()
        far = 0.0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            ext = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            if not ext:
                continue
            m = xf.GetLocalToWorldTransform(prim) * to_root
            for x in (ext[0][0], ext[1][0]):
                for y in (ext[0][1], ext[1][1]):
                    for z in (ext[0][2], ext[1][2]):
                        c = m.Transform(Gf.Vec3d(x, y, z))
                        far = max(far, math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2))
        return far * UsdGeom.GetStageMetersPerUnit(stage)
    except Exception as exc:   # diagnostic; reported as FAIL by the caller if None
        print(f"  (footprint measurement failed: {exc})")
        return None


def run_checks(args: argparse.Namespace, data: Dict[str, Any]) -> int:
    import torch  # noqa: PLC0415

    from drone.math3d import quat_to_rotmat, world_to_heading, yaw_of  # noqa: PLC0415
    from drone.pilot import WaypointPilot  # noqa: PLC0415
    from maze.grid import build_grid  # noqa: PLC0415
    from maze.routes import densify  # noqa: PLC0415

    data["env"]["num_envs"] = args.probe_envs
    data["env"]["rerender_on_reset"] = True
    env, _ = make_env(data)
    n, dev = env.num_envs, env.device
    maze = env.maze
    out = Path(args.out) / maze.name
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    r = env.radius
    zero_act = env.unscale_commands(torch.zeros(n, 4, device=dev))

    def step(actions, k=1):
        obs = None
        for _ in range(k):
            obs, _, _, _, _ = env.step(actions)
        return obs["policy"]

    def heading_vel():
        R = quat_to_rotmat(env.robot.data.root_quat_w)
        return world_to_heading(yaw_of(R), env.robot.data.root_lin_vel_w)

    print(f"\n=== validate_sim: {maze.name} ({n} envs) ===")
    # ---- identity ---------------------------------------------------------------
    print(f"  bodies: {env.robot.body_names}\n  joints: {env.robot.joint_names}")
    mass_g = env.mass[0].item() * 1000
    rep.check("PhysX mass sane", 15.0 < mass_g < 60.0, f"{mass_g:.1f} g (Crazyflie 2.x ~27-35 g)")
    spread = (env.mass.max() - env.mass.min()).item()
    rep.check("mass identical across envs", spread < 1e-9, f"spread {spread:.2e} kg")
    fp = measure_footprint("/World/envs/env_0/Robot")
    rep.check("6.4 footprint <= collision radius", fp is not None and fp <= r,
              f"measured {fp if fp is None else round(fp, 3)} m vs collision_radius_m {r}")
    rep.data["mass_g"], rep.data["footprint_m"] = mass_g, fp

    # ---- hover at the start pose (6.1) -----------------------------------------------
    env.reset()
    d0 = dict(env.diagnostics())
    p0 = env.pos_local().clone()
    step(zero_act, int(2.0 / env.policy_dt))
    drift = torch.linalg.norm(env.pos_local() - p0, dim=-1)
    R = quat_to_rotmat(env.robot.data.root_quat_w)
    tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
    d1 = env.diagnostics()
    hits = d1["analytic_hits"] - d0["analytic_hits"]
    phys = d1["physx_hits"] - d0["physx_hits"]
    rep.check("6.1 hover at spawn: no contact", hits == 0 and phys == 0, f"analytic {hits}, PhysX {phys}")
    rep.check("hover holds position", drift.max().item() < 0.15,
              f"max drift {drift.max().item():.3f} m over 2 s (zero command)")
    rep.check("hover level", tilt.max().item() < 5.0, f"max tilt {tilt.max().item():.2f} deg")

    # ---- controller probes from the most open point --------------------------------------
    grid = build_grid(env.geom.__class__(maze, "cpu"), 0.25, r)
    flat = grid.clearance.reshape(-1).clone()
    k = int(flat.argmax())
    open_pt = grid.centers().reshape(-1, 3)[k]
    open_pt[2] = 1.5
    rep.data["open_point"] = open_pt.tolist()
    env.forced_start = (open_pt.to(dev).repeat(n, 1), torch.zeros(n, device=dev))
    env.reset()
    step(zero_act, 10)
    probes = torch.zeros(n, 4, device=dev)
    group = torch.arange(n, device=dev) % 4
    targets = torch.tensor([1.5, 0.8, 0.5, 1.0], device=dev)
    probes[torch.arange(n, device=dev), group] = targets[group]
    yaw0 = yaw_of(quat_to_rotmat(env.robot.data.root_quat_w)).clone()
    step(env.unscale_commands(probes), int(1.0 / env.policy_dt))
    hv = heading_vel()
    wz = env.robot.data.root_ang_vel_b[:, 2]
    names = ["vx", "vy", "vz", "yaw_rate"]
    for gi, name in enumerate(names):
        m = group == gi
        meas = (wz[m] if gi == 3 else hv[m, gi]).mean().item()
        cross = 0.0 if gi == 3 else hv[m][:, [j for j in range(3) if j != gi]].abs().max().item()
        ok = meas / targets[gi].item() > 0.8 and (gi == 3 or cross < 0.3)
        rep.check(f"controller {name} step", ok,
                  f"cmd {targets[gi].item():.2f} -> {meas:.2f} after 1 s" + ("" if gi == 3 else f", cross-axis {cross:.2f}"))
    rep.data["yaw_change_deg"] = torch.rad2deg(yaw_of(quat_to_rotmat(env.robot.data.root_quat_w)) - yaw0)[group == 3].mean().item()

    # ---- forced crash + tunnelling (termination OFF) -------------------------------------------
    env.terminate_on_collision = False
    env.reset()
    step(zero_act, 5)
    first_hit = torch.full((n,), -1, device=dev)
    phys_seen = torch.zeros(n, dtype=torch.bool, device=dev)
    min_center_clr = torch.full((n,), float("inf"), device=dev)
    crash = env.unscale_commands(torch.tensor([[3.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1))
    for t in range(int(6.0 / env.policy_dt)):
        step(crash)
        clr = env.geom.clearance(env.pos_local())
        min_center_clr = torch.minimum(min_center_clr, clr)
        newly = (clr < r) & (first_hit < 0)
        first_hit[newly] = t
        phys_seen |= env.physx_contact
    rep.check("crash: analytic fires in every env", bool((first_hit >= 0).all()),
              f"{int((first_hit >= 0).sum())}/{n} envs")
    rep.check("crash: PhysX contact in every env", bool(phys_seen.all()),
              f"{int(phys_seen.sum())}/{n} envs (last env: {bool(phys_seen[-1])}) -- else maze colliders did not replicate")
    rep.check("crash: no tunnelling", min_center_clr.min().item() > -0.02,
              f"min centre clearance {min_center_clr.min().item():.3f} m (negative = inside a wall)")
    env.terminate_on_collision = True
    env.forced_start = None

    # ---- scripted flythrough of both routes (6.5) + reward audit ---------------------------------
    env.record_episodes = True
    env.reset()
    env.drain_records()
    half = n // 2
    groups = {"fast": torch.arange(0, half, device=dev), "slow": torch.arange(half, n, device=dev)}
    pilots = {kind: WaypointPilot(maze.route(kind).waypoints, len(ids), maze.route(kind).design_speed_mps,
                                  device=dev) for kind, ids in groups.items()}
    first: Dict[int, Any] = {}
    for _ in range(int(float(data["env"]["episode_length_s"]) / env.policy_dt) + 5):
        cmd = torch.zeros(n, 4, device=dev)
        pos, yaw = env.pos_local(), yaw_of(quat_to_rotmat(env.robot.data.root_quat_w))
        for kind, ids in groups.items():
            cmd[ids] = pilots[kind].act(pos[ids], yaw[ids])
        step(env.unscale_commands(cmd))
        for rec in env.drain_records():
            first.setdefault(rec.env_id, rec)
        if len(first) == n:
            break
    fly = {}
    for kind, ids in groups.items():
        recs = [first[i] for i in ids.tolist() if i in first]
        succ = [x for x in recs if x.outcome == "success"]
        right = [x for x in succ if x.route == kind]
        fly[kind] = {
            "episodes": len(recs), "success": len(succ), "classified_right": len(right),
            "mean_time_s": sum(x.time_s for x in succ) / max(1, len(succ)),
            "min_clearance_m": min((x.min_clearance_m for x in recs), default=float("nan")),
            "outcomes": [x.outcome for x in recs],
        }
        rep.check(f"6.5 sim flythrough ({kind})", len(recs) == len(ids) and len(right) == len(ids),
                  f"{len(succ)}/{len(ids)} success, {len(right)} classified '{kind}', "
                  f"{fly[kind]['mean_time_s']:.1f} s, min clearance {fly[kind]['min_clearance_m']:.2f} m")
    rep.data["flythrough"] = fly
    comps = list(next(iter(first.values())).reward) if first else []
    audit = {c: sum(abs(x.reward[c]) for x in first.values()) / max(1, len(first)) for c in comps}
    rep.data["reward_audit_scripted"] = audit
    nonzero = [v for v in audit.values() if v > 1e-9]
    ratio = max(nonzero) / min(nonzero) if nonzero else float("inf")
    rep.check("s4 reward balance (scripted)", ratio <= 10.0,
              "mean |episode sum| " + ", ".join(f"{c} {v:.2f}" for c, v in audit.items()) + f" -> max/min {ratio:.1f}x")

    # random flight: collisions dominate -- reported, not gated (only the success regime must be balanced)
    env.reset()
    env.drain_records()
    for _ in range(int(10.0 / env.policy_dt)):
        step(torch.rand(n, 4, device=dev) * 2 - 1)
    recs = env.drain_records()
    rep.data["reward_audit_random"] = {c: sum(abs(x.reward[c]) for x in recs) / max(1, len(recs)) for c in comps}
    rep.data["random_outcomes"] = {o: sum(x.outcome == o for x in recs) for o in ("success", "collision", "timeout")}
    print(f"  random policy: {rep.data['random_outcomes']}, audit {rep.data['reward_audit_random']}")
    env.record_episodes = False

    # ---- frames along each route (6.6) --------------------------------------------------------
    if env.layout.image_shape is not None:
        views = []
        for kind in ("fast", "slow"):
            pts = densify(maze.route(kind).waypoints, 0.05)
            L = len(pts)
            idx = torch.linspace(int(0.12 * L), int(0.88 * L), n // 2).long()
            for i in idx.tolist():
                d = pts[min(i + 5, L - 1)] - pts[max(i - 5, 0)]
                views.append((kind, pts[i], math.atan2(float(d[1]), float(d[0]))))
        views = views[:n]
        pos = torch.stack([v[1] for v in views] + [views[-1][1]] * (n - len(views))).to(dev)
        yaw = torch.tensor([v[2] for v in views] + [views[-1][2]] * (n - len(views)), device=dev)
        env.forced_start = (pos, yaw)
        env.reset()
        obs = step(zero_act, 2)
        env.forced_start = None
        C, H, W = env.layout.image_shape
        imgs = obs[:, : env.layout.image_dim].reshape(n, C, H, W).permute(0, 2, 3, 1).clamp(0, 1).cpu()
        per_env_std = imgs.reshape(n, -1).std(dim=1)
        between = torch.cdist(imgs.reshape(n, -1), imgs.reshape(n, -1)).mean().item()
        rep.check("camera renders (not blank)", bool((per_env_std > 0.02).all()),
                  f"min per-frame std {per_env_std.min().item():.3f} (last env {per_env_std[-1].item():.3f})")
        rep.check("camera moves with the drone", between > 1.0, f"mean pairwise frame distance {between:.2f}")
        save_frame_grid(imgs, [f"{v[0]} @({v[1][0]:.1f},{v[1][1]:.1f})" for v in views], out, H, W)
        print(f"  frames -> {out}/frames_grid.png  (HUMAN CHECK 6.6: can you tell fast from slow corridors?)")

    rep.data["diagnostics"] = env.diagnostics()
    write(out, rep, data, "checks")
    env.close()
    return 0 if rep.passed else 1


def save_frame_grid(imgs, labels, out: Path, H: int, W: int) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(labels)
    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    for i, ax in enumerate(axes.reshape(-1) if hasattr(axes, "reshape") else [axes]):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i].numpy(), interpolation="nearest")
            ax.set_title(labels[i], fontsize=6, color="tab:red" if labels[i].startswith("fast") else "tab:blue")
    fig.suptitle(f"policy camera frames at native {W}x{H} (red = fast route, blue = slow route)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "frames_grid.png", dpi=150)
    plt.close(fig)


def run_throughput(args: argparse.Namespace, data: Dict[str, Any]) -> int:
    import torch  # noqa: PLC0415

    env, _ = make_env(data)
    n = env.num_envs
    env.reset()
    act = lambda: torch.rand(n, 4, device=env.device) * 2 - 1   # noqa: E731
    for _ in range(20):   # warm-up (first renders build caches)
        env.step(act())
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.throughput_steps):
        env.step(act())
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = n * args.throughput_steps / dt
    ppo = data["algo"]["ppo"]
    steps_total = int(ppo["max_iterations"]) * int(ppo["num_steps_per_env"]) * n
    hours = steps_total / sps / 3600
    rep = Report()
    rep.data.update(num_envs=n, env_steps_per_s=sps, sim_seconds_per_wall_second=sps * env.policy_dt,
                    planned_env_steps=steps_total, est_collection_hours=hours)
    print(f"\n  throughput: {sps:,.0f} env-steps/s at {n} envs "
          f"(camera {env.layout.image_shape}); {ppo['max_iterations']} iters = {steps_total / 1e6:.1f}M steps "
          f"~ {hours:.1f} h of collection (+ PPO update time)")
    write(Path(args.out) / env.maze.name, rep, data, "throughput")
    env.close()
    return 0


def write(out: Path, rep: Report, data: Dict[str, Any], mode: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"report_{mode}.json"
    path.write_text(json.dumps({"passed": rep.passed, "checks": rep.checks, "data": rep.data,
                                "maze": data["maze"]["config"], "config": data}, indent=2, default=str))
    print(f"\n{'ALL PASS' if rep.passed else 'SOME CHECKS FAILED'} -> {path}")


def main() -> None:
    args = parse_args()
    data = load_run_config(args)
    data["logging"]["wandb"]["enabled"] = False
    app = launch(args, data, force_cameras=True)
    set_global_seed(int(data["seed"]))
    rc = run_checks(args, data) if args.mode == "checks" else run_throughput(args, data)
    app.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
