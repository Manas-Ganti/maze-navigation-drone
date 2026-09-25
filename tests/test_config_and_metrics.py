"""Config validation, obs layout, metrics and the video take selection rule."""

import pytest

from env.obs_layout import build_layout
from eval.metrics import markdown_table, summarize, wilson
from eval.record_video import pick_representative
from training.config import apply_overrides, load_config, load_yaml, validate_config
from training.ppo_config import checkpoint_iteration, resolve_checkpoint_path


def test_default_config_is_valid():
    load_config("configs/train.yaml")


def test_goal_bonus_must_beat_worst_time_penalty():
    data = apply_overrides(load_yaml("configs/train.yaml"), ["reward.goal_bonus=5.0"])
    with pytest.raises(ValueError, match="goal_bonus"):
        validate_config(data)


def test_override_rejects_unknown_key():
    with pytest.raises(KeyError):
        apply_overrides(load_yaml("configs/train.yaml"), ["reward.colision_penalty=3"])


def test_obs_layout_dims():
    data = load_yaml("configs/train.yaml")
    lay = build_layout(data)
    assert lay.image_shape == (3, 64, 64) and lay.proprio_dim == 19
    assert lay.total_dim == 3 * 64 * 64 + 19
    data["observation"]["camera_enabled"] = False
    assert build_layout(data).total_dim == 19


def test_checkpoint_order_is_numeric(tmp_path):
    for i in (50, 900, 1000):
        (tmp_path / f"model_{i}.pt").write_bytes(b"")
    (tmp_path / "model_final.pt").write_bytes(b"")
    assert resolve_checkpoint_path(str(tmp_path), "latest").endswith("model_1000.pt")
    assert checkpoint_iteration("model_final.pt") is None
    assert resolve_checkpoint_path(str(tmp_path / "empty"), "auto") is None


def test_wilson_interval():
    lo, hi = wilson(64, 128)
    assert 0.41 < lo < 0.42 and 0.58 < hi < 0.59
    assert wilson(0, 10)[0] == 0.0


def _ep(outcome, t, route):
    return {"outcome": outcome, "time_s": t, "route": route, "path_length_m": 10.0,
            "min_clearance_m": 0.3, "mean_speed_mps": 10.0 / t}


def test_summarize_rates_and_routes():
    eps = [_ep("success", 8.0, "fast"), _ep("success", 14.0, "slow"), _ep("collision", 3.0, "fast"),
           _ep("timeout", 40.0, "none")]
    s = summarize(eps)
    assert s["success_rate"] == 0.5 and s["route_fast"] == 0.5
    assert s["success_rate_on_fast"] == 0.5 and s["time_on_slow_s"] == 14.0
    assert "iter 1" in markdown_table([{"label": "iter 1", **s}])


def test_representative_take_is_modal_and_median():
    takes = [{"outcome": "success", "time_s": t} for t in (9.0, 7.0, 12.0)] + [{"outcome": "collision", "time_s": 2.0}]
    assert pick_representative(takes) == 0          # median success time is 9.0
    takes = [{"outcome": "collision", "time_s": t} for t in (1.0, 3.0, 5.0)] + [{"outcome": "success", "time_s": 9.0}]
    assert pick_representative(takes) == 1          # the modal outcome is a crash: show a typical crash
