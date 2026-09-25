"""Train the maze policy: config -> Isaac env -> rsl_rl PPO -> checkpoints + W&B.

On ARC (see arc/train.slurm):

    python training/train.py --headless --run-name a_s42 --resume auto
    python training/train.py --headless --run-name b_s42 --resume auto \\
        --set maze.config=configs/mazes/maze_b_overpass.yaml

Run ``eval/validate_sim.py`` for the maze first -- training on a maze that
has not passed the section 6 checklist is how GPU-hours get wasted.

Checkpoints: results/runs/<run>/model_<iter>.pt every ``save_interval``; ALL
are kept (the improvement table and the early-vs-final videos need them).
``resolved_config.json`` is written next to them: a checkpoint without the
config that produced it (reward weights, maze, obs layout) is unusable later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.app import add_common_args, launch, load_run_config, make_env, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--run-name", default=None, help="run dir + W&B run id (fixed => resumable)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path | 'latest' (must exist) | 'auto' (resume if present, else fresh)")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--seed-from-array", action="store_true",
                    help="SLURM array: seed += SLURM_ARRAY_TASK_ID and '_s<seed>' is appended to --run-name")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = load_run_config(args)
    if args.max_iterations is not None:
        data["algo"]["ppo"]["max_iterations"] = args.max_iterations
    if args.run_name is not None:
        data["experiment"]["run_name"] = args.run_name
    if args.no_wandb:
        data["logging"]["wandb"]["enabled"] = False
    if args.seed_from_array:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            raise SystemExit("--seed-from-array but SLURM_ARRAY_TASK_ID is unset (not an array job)")
        data["seed"] = int(data["seed"]) + int(task)
        data["experiment"]["run_name"] = f"{data['experiment']['run_name'] or 'run'}_s{data['seed']}"
    if args.resume == "auto" and not data["experiment"]["run_name"]:
        raise SystemExit("--resume auto needs --run-name (a stable directory to resume from)")

    app = launch(args, data)
    set_global_seed(int(data["seed"]))

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

    from training.ppo_config import build_runner, resolve_checkpoint_path  # noqa: PLC0415

    env, _ = make_env(data)
    log_dir = setup_logging(data)
    runner = build_runner(RslRlVecEnvWrapper(env), data, log_dir=log_dir, device=str(env.device))

    ckpt = resolve_checkpoint_path(log_dir, args.resume)
    if ckpt:
        print(f"Resuming from {ckpt}")
        runner.load(ckpt)
    elif args.resume == "auto":
        print("No checkpoint in this run dir -- starting fresh.")

    # learn() runs N iterations MORE than the loaded one: ask only for what is left.
    total = int(data["algo"]["ppo"]["max_iterations"])
    done = int(getattr(runner, "current_learning_iteration", 0))
    remaining = max(0, total - done)
    print(f"\n=== Training {remaining} iterations ({done}/{total} done) on {env.maze.name} ===")
    if remaining > 0:
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"diagnostics: {env.diagnostics()}")
    env.close()
    app.close()


def setup_logging(data) -> str:
    from datetime import datetime  # noqa: PLC0415

    exp = data["experiment"]
    run = exp.get("run_name") or f"{datetime.now():%Y%m%d-%H%M%S}_seed{data['seed']}"
    exp["run_name"] = run
    log_dir = os.path.join(data["logging"]["log_dir"], run)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "resolved_config.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    w = data["logging"]["wandb"]
    if w["enabled"]:
        import wandb  # noqa: PLC0415

        # Stable id: a walltime-resumed job appends to the same W&B run. rsl_rl's
        # own wandb.init later in the process reuses this active run.
        wandb.init(project=w["project"], entity=w["entity"], name=run, id=run, resume="allow",
                   mode=w["mode"], config=data, notes=exp.get("notes", ""), tags=exp.get("tags", []),
                   dir=log_dir)
    print(f"Logging to {log_dir}")
    return log_dir


if __name__ == "__main__":
    main()
