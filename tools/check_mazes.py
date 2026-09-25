"""Run the pure half of the s6 checklist on maze configs and draw top-down plots.

Laptop or ARC login node; seconds; no Isaac:

    python tools/check_mazes.py                         # all configs/mazes/*.yaml
    python tools/check_mazes.py configs/mazes/maze_a_window.yaml --no-plot

Writes results/maze_checks/<maze>.png (layout, routes, flown trajectories,
geodesic field) and results/maze_checks/summary.json. Exit code 1 if any
check fails -- do not submit training for a maze that fails here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maze.routes import break_even  # noqa: E402
from maze.spec import load_maze  # noqa: E402
from maze.validate import validate_maze  # noqa: E402
from training.config import load_config, policy_dt, shaping_gamma  # noqa: E402


def plot(spec, report, cfg, out: Path) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    X, Y, Z = spec.bounds
    for ax in axes:
        ax.set_xlim(-0.6, X + 0.6)
        ax.set_ylim(-0.6, Y + 0.6)
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), X, Y, fill=False, lw=2, ec="k"))

    ax = axes[0]
    # decor (floor strips) first and faint, so colliders are drawn on top
    for p in sorted(spec.primitives, key=lambda q: q.collision):
        if not p.collision:
            alpha, hatch, z = 0.25, None, 0
        else:
            alpha = 0.95 if p.pos[2] - p.half_extents[2] < 0.05 else 0.55   # floating things translucent
            hatch = "//" if p.pos[2] + p.half_extents[2] < Z - 0.05 else None
            z = 2
        if p.type == "box":
            hx, hy, _ = p.half_extents
            c, s = math.cos(math.radians(p.yaw_deg)), math.sin(math.radians(p.yaw_deg))
            corners = [(p.pos[0] + c * x - s * y, p.pos[1] + s * x + c * y)
                       for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            ax.add_patch(Polygon(corners, fc=p.color, ec="k", lw=0.4, alpha=alpha, hatch=hatch, zorder=z))
        else:
            ax.add_patch(Circle(p.pos[:2], p.radius, fc=p.color, ec="k", lw=0.4, alpha=alpha, hatch=hatch, zorder=z))
        if p.tag in ("partition", "hanging", "frame"):
            lo, hi = p.pos[2] - p.half_extents[2], p.pos[2] + p.half_extents[2]
            ax.annotate(f"z{lo:.1f}-{hi:.1f}", p.pos[:2], fontsize=5, ha="center", zorder=5)
    colors = {"fast": "tab:red", "slow": "tab:blue"}
    for kind in ("fast", "slow"):
        r = spec.route(kind)
        xs, ys = zip(*[(w[0], w[1]) for w in r.waypoints])
        ax.plot(xs, ys, "--", color=colors[kind], lw=1.5, label=f"{kind} (authored) {r.length_m:.1f} m")
        for reg in r.signature:
            ax.add_patch(Rectangle((reg.pos[0] - reg.size[0] / 2, reg.pos[1] - reg.size[1] / 2),
                                   reg.size[0], reg.size[1], fill=False, ec=colors[kind], lw=1, ls=":"))
        traj = report.trajectories[kind]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[kind], lw=0.8, alpha=0.7,
                label=f"{kind} flown (model) {report.metrics['flythrough'][kind]['outcome']}")
    ax.plot(*spec.start_pos[:2], "g^", ms=10, label="start")
    ax.add_patch(Circle(spec.goal_pos[:2], spec.goal_radius_m, fc="gold", ec="k", label="goal"))
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.set_title(f"{spec.name}: top view (hatched = below ceiling height, translucent = floating, "
                 "faint = floor strip)", fontsize=9)

    ax = axes[1]
    field = report.field
    k = min(int(round(spec.start_pos[2] / report.grid.res - 0.5)), field.shape[2] - 1)
    im = ax.imshow(field[:, :, k].T.clamp(max=60).numpy(), origin="lower", extent=(0, X, 0, Y), cmap="viridis")
    free = report.grid.free[:, :, k].T.numpy()
    ax.contour(free, levels=[0.5], colors="w", linewidths=0.5, extent=(0, X, 0, Y))
    fig.colorbar(im, ax=ax, fraction=0.046, label="geodesic distance to goal (m)")
    ax.set_title(f"geodesic potential at z = {(k + 0.5) * report.grid.res:.1f} m (white: free-space edge)", fontsize=9)

    txt = "\n".join(c.line() for c in report.checks)
    fig.text(0.01, 0.005, txt, fontsize=5.5, family="monospace", va="bottom")
    fig.subplots_adjust(bottom=0.26, top=0.95, left=0.03, right=0.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mazes", nargs="*", help="maze YAMLs (default: configs/mazes/*.yaml)")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default="results/maze_checks")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.mazes:
        paths = args.mazes
    elif any(o.startswith("maze.config=") for o in args.set):
        paths = [cfg["maze"]["config"]]   # the launcher passed the run's maze: check just that one
    else:
        paths = sorted(str(p) for p in Path("configs/mazes").glob("*.yaml"))
    rw = cfg["reward"]
    summary = {}
    ok_all = True
    for path in paths:
        spec = load_maze(path)
        report = validate_maze(spec, cfg)
        print(report.text())
        be = break_even(spec.route("fast"), spec.route("slow"), float(rw["step_penalty"]),
                        float(rw["goal_bonus"]), float(rw["collision_penalty"]), shaping_gamma(cfg),
                        policy_dt(cfg))
        print(f"  break-even: fast route preferred iff P(crash on fast) < {be.p_crash_break_even:.1%} "
              f"(fast {be.fast_time_s:.1f} s vs slow {be.slow_time_s:.1f} s at design speed)\n")
        summary[spec.name] = {"passed": report.passed, "metrics": report.metrics, "break_even": be.as_dict(),
                              "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                                         for c in report.checks]}
        ok_all &= report.passed
        if not args.no_plot:
            plot(spec, report, cfg, Path(args.out) / f"{spec.name}.png")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("ALL MAZES PASS" if ok_all else "SOME MAZES FAIL -- fix before training")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
