"""Shared entry-point plumbing: argparse + AppLauncher + config + env construction.

Isaac Sim's ``AppLauncher`` MUST start before any ``isaaclab.*`` import, which
is why every Isaac import in the entry points sits inside a function called
after :func:`launch`. Moving one to module scope gives an "Isaac Sim not
initialized" crash.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an existing config key, e.g. --set maze.config=configs/mazes/maze_b_overpass.yaml")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        AppLauncher.add_app_launcher_args(ap)
    except ImportError:   # allows --help on a machine without Isaac
        ap.add_argument("--headless", action="store_true")


def load_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    from training.config import apply_overrides, load_yaml, validate_config  # noqa: PLC0415

    data = apply_overrides(load_yaml(args.config), args.set)
    if args.num_envs is not None:
        data["env"]["num_envs"] = args.num_envs
    if args.seed is not None:
        data["seed"] = args.seed
    validate_config(data)
    return data


def launch(args: argparse.Namespace, data: Dict[str, Any], force_cameras: bool = False) -> Any:
    """Start Isaac Sim. Cameras need the renderer: RT-core GPU (L40S), --enable_cameras."""
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    if data["observation"]["camera_enabled"] or force_cameras:
        args.enable_cameras = True
    return AppLauncher(args).app


def set_global_seed(seed: int) -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_env(data: Dict[str, Any], render_mode: str | None = None, viewer: Any = None) -> Tuple[Any, Any]:
    """Build the Isaac env (call only after :func:`launch`). Returns (env, env_cfg)."""
    from env.maze_env import MazeDroneEnv, build_env_cfg  # noqa: PLC0415

    env_cfg = build_env_cfg(data)
    if viewer is not None:
        env_cfg.viewer = viewer
    env = MazeDroneEnv(env_cfg, render_mode=render_mode)
    print(f"[env] maze={env.maze.name} num_envs={env.num_envs} obs_dim={env.layout.total_dim} "
          f"image={env.layout.image_shape} policy_dt={env.policy_dt:.3f}s device={env.device}")
    print(f"[env] drone mass (PhysX) {env.mass[0].item() * 1000:.1f} g, "
          f"body inertia {[round(x, 8) for x in env.inertia_diag[0].tolist()]} kg m^2")
    return env, env_cfg
