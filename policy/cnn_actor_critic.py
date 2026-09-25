"""CNN actor-critic for rsl_rl 2.3.1 (pure torch).

rsl_rl 2.3.1 ships MLP-only ``ActorCritic``. Its ``OnPolicyRunner`` builds the
policy with ``eval(policy_cfg["class_name"])`` inside
``rsl_rl.runners.on_policy_runner``, so a custom class only has to be visible
in that module's namespace -- :func:`register_with_rsl_rl` puts it there. The
class implements the exact duck-typed interface rsl_rl's PPO uses (act,
act_inference, evaluate, get_actions_log_prob, action_mean/std, entropy,
reset, is_recurrent), mirroring ``rsl_rl.modules.ActorCritic``.

Observation layout (flat, as rsl_rl requires): ``[image | proprio]`` where the
image is ``C*H*W`` channel-first floats in [0, 1] and proprio is the rest.
With ``image_shape=None`` (camera disabled) this degrades to an MLP.

Shared encoder by default: one conv trunk feeds both heads, trained by PPO's
combined loss. Separate trunks double the rendering-bound step's GPU cost for
little gain at this scale; ``shared_encoder: false`` is available.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "lrelu": nn.LeakyReLU}


def _act(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {name!r}")
    return _ACTIVATIONS[name]()


def _mlp(inp: int, hidden: Sequence[int], out: int, activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = inp
    for h in hidden:
        layers += [nn.Linear(last, h), _act(activation)]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """Nature-DQN-style conv stack -> ``feature_dim``."""

    def __init__(self, image_shape: Sequence[int], channels: Sequence[int], kernels: Sequence[int],
                 strides: Sequence[int], feature_dim: int, activation: str):
        super().__init__()
        c, h, w = image_shape
        layers: List[nn.Module] = []
        for ch, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv2d(c, ch, k, s), _act(activation)]
            c = ch
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, *image_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, feature_dim), _act(activation))
        self.image_shape = tuple(image_shape)

    def forward(self, flat_img: torch.Tensor) -> torch.Tensor:
        x = flat_img.reshape(-1, *self.image_shape) - 0.5   # [0,1] -> [-0.5, 0.5]
        return self.fc(self.conv(x))


class CNNActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: Optional[Sequence[int]] = None,
        conv_channels: Sequence[int] = (32, 64, 64),
        conv_kernels: Sequence[int] = (8, 4, 3),
        conv_strides: Sequence[int] = (4, 2, 1),
        feature_dim: int = 256,
        actor_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        shared_encoder: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(f"CNNActorCritic ignoring unexpected arguments: {sorted(kwargs)}")
        super().__init__()
        if num_actor_obs != num_critic_obs:
            raise ValueError("asymmetric actor/critic observations are not wired for the CNN policy")
        self.image_dim = 0 if image_shape is None else int(torch.tensor(image_shape).prod())
        self.proprio_dim = num_actor_obs - self.image_dim
        if self.proprio_dim < 0:
            raise ValueError(f"obs dim {num_actor_obs} smaller than image {tuple(image_shape or ())}")

        def encoder() -> Optional[ImageEncoder]:
            if image_shape is None:
                return None
            return ImageEncoder(image_shape, conv_channels, conv_kernels, conv_strides, feature_dim, activation)

        self.actor_encoder = encoder()
        self.critic_encoder = self.actor_encoder if shared_encoder else encoder()
        head_in = (feature_dim if image_shape is not None else 0) + self.proprio_dim
        self.actor = _mlp(head_in, actor_hidden_dims, num_actions, activation)
        self.critic = _mlp(head_in, critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)
        print(f"CNNActorCritic: image {tuple(image_shape) if image_shape else None} + proprio {self.proprio_dim} "
              f"-> {num_actions} actions; shared_encoder={shared_encoder}; "
              f"{sum(p.numel() for p in self.parameters()):,} params")

    # ---- features ------------------------------------------------------------
    def _features(self, obs: torch.Tensor, encoder: Optional[ImageEncoder]) -> torch.Tensor:
        proprio = obs[:, self.image_dim:]
        if encoder is None:
            return proprio
        return torch.cat([encoder(obs[:, : self.image_dim]), proprio], dim=-1)

    # ---- rsl_rl interface ---------------------------------------------------------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._features(observations, self.actor_encoder))
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(self._features(observations, self.actor_encoder))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(self._features(critic_observations, self.critic_encoder))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True   # rsl_rl: "this load resumes training"


def register_with_rsl_rl() -> None:
    """Make ``class_name: CNNActorCritic`` resolvable by rsl_rl's runner."""
    import rsl_rl.runners.on_policy_runner as opr  # noqa: PLC0415

    opr.CNNActorCritic = CNNActorCritic
