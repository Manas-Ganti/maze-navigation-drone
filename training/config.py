"""Config loading, ``--set`` overrides and cross-field validation. PURE.

Validation runs AFTER overrides: an override is exactly how an invalid config
sneaks in, so the assertions must see the final values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """``--set a.b.c=value``; values are YAML-parsed (bools, numbers, lists).
    Only EXISTING keys may be overridden -- a typo cannot add a dead setting."""
    data = copy.deepcopy(data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects KEY=VALUE, got {override!r}")
        key, _, raw = override.partition("=")
        value = yaml.safe_load(raw)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"--set key '{key}': unknown path segment '{part}'")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"--set key '{key}' does not exist in the config")
        node[parts[-1]] = value
        print(f"[override] {key} = {value!r}")
    return data


def policy_dt(data: Dict[str, Any]) -> float:
    return float(data["sim"]["dt"]) * int(data["sim"]["decimation"])


def max_episode_steps(data: Dict[str, Any]) -> int:
    return int(round(float(data["env"]["episode_length_s"]) / policy_dt(data)))


def shaping_gamma(data: Dict[str, Any]) -> float:
    g = data["reward"]["shaping_gamma"]
    if g == "algo":
        return float(data["algo"]["ppo"]["algorithm"]["gamma"])
    return float(g)


def _conv_out(n: int, kernels: List[int], strides: List[int]) -> int:
    for k, s in zip(kernels, strides):
        n = (n - k) // s + 1
    return n


def validate_config(data: Dict[str, Any]) -> None:
    """Raise with a readable message on any inconsistent setting."""
    errors: List[str] = []
    rw, sim = data["reward"], data["sim"]
    ppo = data["algo"]["ppo"]

    if data["algo"]["name"] != "ppo" or data["algo"]["library"] != "rsl_rl":
        errors.append("only algo.name=ppo with algo.library=rsl_rl is wired")
    if int(sim["decimation"]) < 1 or float(sim["dt"]) <= 0:
        errors.append("sim.dt must be > 0 and sim.decimation >= 1")

    steps = max_episode_steps(data)
    worst_time = float(rw["step_penalty"]) * steps
    if float(rw["goal_bonus"]) <= worst_time:
        errors.append(
            f"reward.goal_bonus ({rw['goal_bonus']}) must exceed the largest possible accumulated time "
            f"penalty ({rw['step_penalty']} x {steps} steps = {worst_time:.2f}) so success always beats "
            "a faster failure (CLAUDE.md s4.4)"
        )
    for k in ("step_penalty", "collision_penalty", "goal_bonus", "progress_weight"):
        if float(rw[k]) < 0:
            errors.append(f"reward.{k} is a magnitude and must be >= 0 (signs are applied in env/reward.py)")
    if rw["potential"] not in ("geodesic", "euclidean"):
        errors.append("reward.potential must be 'geodesic' or 'euclidean'")
    g = shaping_gamma(data)
    if not 0.0 < g <= 1.0:
        errors.append(f"shaping gamma {g} must be in (0, 1]")

    for name in ("vx_range", "vy_range", "vz_range", "yaw_rate_range"):
        lo, hi = data["action"][name]
        if not lo < hi:
            errors.append(f"action.{name} must be [lo, hi] with lo < hi")

    pol = ppo["policy"]
    for side in ("width", "height"):
        n = _conv_out(int(data["camera"][side]), pol["conv_kernels"], pol["conv_strides"])
        if n < 1:
            errors.append(f"camera.{side}={data['camera'][side]} is too small for the conv stack")
    if not (len(pol["conv_channels"]) == len(pol["conv_kernels"]) == len(pol["conv_strides"])):
        errors.append("policy conv_channels / conv_kernels / conv_strides must have equal length")

    ev = data["eval"]
    if int(ev["checkpoint_every"]) % int(ppo["save_interval"]) != 0:
        errors.append("eval.checkpoint_every must be a multiple of algo.ppo.save_interval")

    if not Path(data["maze"]["config"]).exists():
        errors.append(f"maze.config not found: {data['maze']['config']} (paths are relative to the repo root)")

    from drone.controller import ControllerGains  # noqa: PLC0415
    try:
        ControllerGains.from_config(data["controller"])
    except KeyError as exc:
        errors.append(str(exc))

    if errors:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    data = apply_overrides(load_yaml(path), overrides or [])
    validate_config(data)
    return data
