"""The seam between the YAML and rsl_rl 2.3.1. Nothing else imports rsl_rl.

Returns plain dicts (not rsl_rl dataclasses) so the shape can be tested on a
laptop and a renamed field fails in one obvious place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from env.obs_layout import build_layout


def build_runner_cfg(data: Mapping[str, Any]) -> Dict[str, Any]:
    ppo = data["algo"]["ppo"]
    pol = dict(ppo["policy"])
    layout = build_layout(data)
    policy = {
        "class_name": pol.pop("class_name"),
        "image_shape": list(layout.image_shape) if layout.image_shape else None,
        **pol,
    }
    return {
        "seed": int(data["seed"]),
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "empirical_normalization": bool(ppo["empirical_normalization"]),
        "experiment_name": str(data["experiment"]["name"]),
        "run_name": data["experiment"].get("run_name") or "",
        "logger": "wandb" if data["logging"]["wandb"]["enabled"] else "tensorboard",
        "wandb_project": data["logging"]["wandb"]["project"],
        "policy": policy,
        "algorithm": {"class_name": "PPO", **{k: v for k, v in ppo["algorithm"].items()}},
    }


def build_runner(env: Any, data: Mapping[str, Any], log_dir: Optional[str], device: str) -> Any:
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    from policy.cnn_actor_critic import register_with_rsl_rl  # noqa: PLC0415

    register_with_rsl_rl()
    return OnPolicyRunner(env, build_runner_cfg(data), log_dir=log_dir, device=device)


def checkpoint_iteration(path: str | Path) -> Optional[int]:
    stem = Path(path).stem
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if stem.startswith("model_") and tail.isdigit() else None


def list_checkpoints(run_dir: str | Path) -> Dict[int, Path]:
    """{iteration: path} for every model_<N>.pt in a run dir, numerically sorted.
    (Lexicographic order ranks model_900 above model_1000.)"""
    found = {checkpoint_iteration(p): p for p in Path(run_dir).glob("model_*.pt")}
    return dict(sorted((k, v) for k, v in found.items() if k is not None))


def resolve_checkpoint_path(log_dir: str, checkpoint: Optional[str]) -> Optional[str]:
    """'latest' -> newest model_<N>.pt (must exist); 'auto' -> same or None (fresh start)."""
    if checkpoint is None:
        return None
    if checkpoint not in ("latest", "auto"):
        return checkpoint
    ckpts = list_checkpoints(log_dir)
    if not ckpts:
        if checkpoint == "auto":
            return None
        raise FileNotFoundError(f"no model_<N>.pt under {log_dir}")
    return str(ckpts[max(ckpts)])
