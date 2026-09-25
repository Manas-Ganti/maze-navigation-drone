"""The CNN policy inside the REAL rsl_rl 2.3.1 runner, against a fake vec-env.

This is the check that would otherwise cost a queue round: rsl_rl resolves
the policy class by ``eval`` in its own module and duck-types the interface,
so a missing method or a wrong ``image_shape`` only fails at runtime.
Skipped when rsl_rl is not installed (it is on ARC; locally: pip install rsl-rl-lib==2.3.1).
"""

from __future__ import annotations

import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")

from env.obs_layout import NUM_ACTIONS, build_layout  # noqa: E402
from policy.cnn_actor_critic import CNNActorCritic  # noqa: E402
from training.config import load_yaml  # noqa: E402
from training.ppo_config import build_runner, build_runner_cfg, list_checkpoints  # noqa: E402


class FakeVecEnv:
    """Minimal rsl_rl VecEnv: random images, episodes end at random."""

    def __init__(self, num_envs: int, obs_dim: int):
        self.num_envs = num_envs
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 50
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.obs_dim = obs_dim

    def get_observations(self):
        return torch.rand(self.num_envs, self.obs_dim), {"observations": {}}

    def reset(self):
        return self.get_observations()

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        obs = torch.rand(self.num_envs, self.obs_dim)
        rew = -actions.pow(2).sum(-1)
        dones = (torch.rand(self.num_envs) < 0.05).long()
        extras = {"observations": {}, "time_outs": torch.zeros(self.num_envs),
                  "log": {"Episode_Reward/progress": torch.tensor(0.1)}}
        return obs, rew, dones, extras


def small_cfg():
    data = load_yaml("configs/train.yaml")
    data["camera"]["width"] = data["camera"]["height"] = 40
    data["algo"]["ppo"]["num_steps_per_env"] = 8
    data["algo"]["ppo"]["save_interval"] = 1
    data["logging"]["wandb"]["enabled"] = False
    return data


def test_runner_learns_and_saves(tmp_path):
    data = small_cfg()
    layout = build_layout(data)
    env = FakeVecEnv(6, layout.total_dim)
    runner = build_runner(env, data, log_dir=str(tmp_path), device="cpu")
    assert isinstance(runner.alg.policy, CNNActorCritic)
    runner.learn(num_learning_iterations=2)
    ckpts = list_checkpoints(tmp_path)
    assert ckpts, "runner saved no checkpoint"
    # reload into a fresh runner and run inference
    runner2 = build_runner(env, data, log_dir=None, device="cpu")
    runner2.load(str(ckpts[max(ckpts)]))
    policy = runner2.get_inference_policy(device="cpu")
    out = policy(torch.rand(3, layout.total_dim))
    assert out.shape == (3, NUM_ACTIONS)


def test_runner_cfg_carries_image_shape():
    data = small_cfg()
    cfg = build_runner_cfg(data)
    assert cfg["policy"]["image_shape"] == [3, 40, 40]
    data["observation"]["camera_enabled"] = False
    assert build_runner_cfg(data)["policy"]["image_shape"] is None


def test_blind_policy_is_mlp():
    data = small_cfg()
    data["observation"]["camera_enabled"] = False
    layout = build_layout(data)
    pol = CNNActorCritic(layout.total_dim, layout.total_dim, NUM_ACTIONS, image_shape=None)
    assert pol.actor_encoder is None
    assert pol.act_inference(torch.rand(2, layout.total_dim)).shape == (2, NUM_ACTIONS)


def test_image_actually_reaches_the_policy():
    """Changing only the image must change the action (the encoder is wired, not bypassed)."""
    data = small_cfg()
    layout = build_layout(data)
    torch.manual_seed(0)
    pol = CNNActorCritic(layout.total_dim, layout.total_dim, NUM_ACTIONS, image_shape=layout.image_shape)
    obs = torch.rand(1, layout.total_dim)
    obs2 = obs.clone()
    obs2[:, : layout.image_dim] = torch.rand(1, layout.image_dim)
    assert not torch.allclose(pol.act_inference(obs), pol.act_inference(obs2))
