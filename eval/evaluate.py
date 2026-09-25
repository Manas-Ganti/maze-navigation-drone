"""Checkpoint sweep: the metrics table + route-choice-over-training (CLAUDE.md s9).

    python eval/evaluate.py --headless --run results/runs/a_s42
    python eval/evaluate.py --headless --run results/runs/a_s42 --checkpoints 0 500 2000 \\
        --baselines random scripted_fast scripted_slow

Uses the run's own ``resolved_config.json`` (same maze, reward, obs layout),
so a checkpoint is never evaluated in an env it was not trained in. ``--set``
overrides apply on top (e.g. ``eval.episodes_per_checkpoint=256``).

Unbiased sampling: every env is reset together and contributes exactly its
FIRST episode, round after round. Taking "the first N episodes to finish"
over-samples short episodes (crashes, fast runs) -- a classic eval bias.

Outputs (results/eval/<run>/): episodes.jsonl, checkpoints.csv, table.md, curves.png.
Every number in the README comes from these files (the PROOF RULE).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.app import launch, make_env, set_global_seed  # noqa: E402

FAST_COLOR, SLOW_COLOR, INK, MUTED = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/runs/<run> (holds model_<N>.pt + resolved_config.json)")
    ap.add_argument("--checkpoints", type=int, nargs="*", default=None,
                    help="iterations to evaluate (default: every eval.checkpoint_every, plus the last)")
    ap.add_argument("--baselines", nargs="*", default=[], choices=["random", "scripted_fast", "scripted_slow"])
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        AppLauncher.add_app_launcher_args(ap)
    except ImportError:
        ap.add_argument("--headless", action="store_true")
    return ap.parse_args()


def run_episodes(env: Any, policy: Any, n_episodes: int) -> List[Dict[str, Any]]:
    """Exactly one (the first) episode per env per round, all envs reset together."""
    out: List[Dict[str, Any]] = []
    max_steps = int(env.max_episode_length) + 2
    while len(out) < n_episodes:
        obs, _ = env.reset()
        policy.reset()
        env.drain_records()
        firsts: Dict[int, Any] = {}
        for _ in range(max_steps):
            obs, _, _, _, _ = env.step(policy.act(obs["policy"]))
            for rec in env.drain_records():
                firsts.setdefault(rec.env_id, rec)
            if len(firsts) == env.num_envs:
                break
        if len(firsts) < env.num_envs:
            raise RuntimeError(f"only {len(firsts)}/{env.num_envs} envs finished within the episode limit")
        out += [asdict(firsts[i]) for i in sorted(firsts)]
    return out[:n_episodes]


def plot_curves(rows: List[Dict[str, Any]], baselines: List[Dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    it = [r["iteration"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_xlabel("PPO iteration", color=MUTED)
        ax.tick_params(colors=MUTED)

    ax = axes[0]
    ax.fill_between(it, [r["success_ci_lo"] for r in rows], [r["success_ci_hi"] for r in rows],
                    color=INK, alpha=0.10, lw=0)
    ax.plot(it, [r["success_rate"] for r in rows], color=INK, lw=2, marker="o", ms=4)
    ax.set_ylim(0, 1.02)
    ax.set_title("Success rate (95% Wilson CI)", loc="left", fontsize=10)

    ax = axes[1]
    ax.plot(it, [r["time_to_goal_median_s"] for r in rows], color=INK, lw=2, marker="o", ms=4)
    for b in baselines:
        if b["label"].startswith("scripted") and b["time_to_goal_median_s"] == b["time_to_goal_median_s"]:
            c = FAST_COLOR if "fast" in b["label"] else SLOW_COLOR
            ax.axhline(b["time_to_goal_median_s"], color=c, lw=1, ls="--")
            ax.annotate(b["label"], (it[0], b["time_to_goal_median_s"]), color=MUTED, fontsize=7,
                        xytext=(2, 3), textcoords="offset points")
    ax.set_title("Median time-to-goal, successes (s)", loc="left", fontsize=10)

    ax = axes[2]
    for key, c, name in (("route_fast", FAST_COLOR, "fast route"), ("route_slow", SLOW_COLOR, "slow route")):
        ys = [r[key] for r in rows]
        ax.plot(it, ys, color=c, lw=2, marker="o", ms=4, label=name)
        ax.annotate(name, (it[-1], ys[-1]), color=MUTED, fontsize=8, xytext=(4, 0), textcoords="offset points",
                    va="center")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Route taken (share of all episodes)", loc="left", fontsize=10)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    from eval.metrics import markdown_table, summarize  # noqa: PLC0415
    from training.config import apply_overrides, validate_config  # noqa: PLC0415
    from training.ppo_config import list_checkpoints  # noqa: PLC0415

    run_dir = Path(args.run)
    data = json.loads((run_dir / "resolved_config.json").read_text())
    data["env"]["num_envs"] = int(data["eval"]["num_envs"])
    data["env"]["rerender_on_reset"] = True
    data["logging"]["wandb"]["enabled"] = False
    data["seed"] = int(data["seed"]) + 1000   # eval seeds differ from training seeds
    data = apply_overrides(data, args.set)
    validate_config(data)

    app = launch(args, data)
    set_global_seed(int(data["seed"]))
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

    from eval.policies import CheckpointPolicy, RandomPolicy, ScriptedRoutePolicy  # noqa: PLC0415

    env, _ = make_env(data)
    env.record_episodes = True
    n_eps = int(data["eval"]["episodes_per_checkpoint"])
    out = Path(args.out or Path("results/eval") / run_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no model_<N>.pt in {run_dir}")
    if args.checkpoints:
        missing = [i for i in args.checkpoints if i not in ckpts]
        if missing:
            raise SystemExit(f"checkpoints not found: {missing}; available: {list(ckpts)}")
        chosen = list(args.checkpoints)
    else:
        every = int(data["eval"]["checkpoint_every"])
        chosen = sorted({i for i in ckpts if i % every == 0} | {max(ckpts)})

    rows, base_rows = [], []
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for name in args.baselines:
            pol = RandomPolicy(env) if name == "random" else ScriptedRoutePolicy(env, name.split("_")[1])
            eps = run_episodes(env, pol, n_eps)
            base_rows.append({"label": name, "iteration": None, **summarize(eps)})
            for e in eps:
                fh.write(json.dumps({"policy": name, **e}) + "\n")
            print(f"[eval] {name}: success {base_rows[-1]['success_rate']:.2f}")
        policy = CheckpointPolicy(RslRlVecEnvWrapper(env), data, deterministic=not args.stochastic)
        for it in chosen:
            policy.load(str(ckpts[it]))
            eps = run_episodes(env, policy, n_eps)
            row = {"label": f"iter {it}", "iteration": it, **summarize(eps)}
            rows.append(row)
            for e in eps:
                fh.write(json.dumps({"policy": f"iter_{it}", **e}) + "\n")
            print(f"[eval] iter {it}: success {row['success_rate']:.2f}, fast {row['route_fast']:.2f}, "
                  f"slow {row['route_slow']:.2f}, median time {row['time_to_goal_median_s']:.1f} s", flush=True)

    all_rows = base_rows + rows
    with (out / "checkpoints.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    table = markdown_table(all_rows)
    header = (f"Maze `{env.maze.name}`, run `{run_dir.name}`, {n_eps} episodes per row "
              f"({'stochastic' if args.stochastic else 'deterministic'} policy, start noise on).\n\n")
    (out / "table.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    if len(rows) >= 2:
        plot_curves(rows, base_rows, out / "curves.png", f"{env.maze.name} -- {run_dir.name}")
    print(f"\nwrote {out}/ (episodes.jsonl, checkpoints.csv, table.md{', curves.png' if len(rows) >= 2 else ''})")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
