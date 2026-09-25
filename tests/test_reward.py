"""Reward: component signs, terminal handling, and EXACT potential-shaping invariance."""

import torch

from env.reward import RewardConfig, compute_reward

CFG = RewardConfig(progress_weight=0.5, shaping_gamma=0.997, terminal_potential_zero=True,
                   step_penalty=0.0125, collision_penalty=10.0, goal_bonus=20.0)


def discounted_shaping(dists, terminal_kind):
    """Discounted sum of the progress term along one trajectory of potential distances."""
    total, g = 0.0, 1.0
    for t in range(len(dists) - 1):
        last = t == len(dists) - 2
        reached = torch.tensor([last and terminal_kind == "goal"])
        collided = torch.tensor([last and terminal_kind == "crash"])
        _, c = compute_reward(torch.tensor([dists[t]]), torch.tensor([dists[t + 1]]), reached, collided, CFG)
        total += g * c["progress"].item()
        g *= CFG.shaping_gamma
    return total


def test_shaping_return_is_route_independent():
    """Ng et al.: with Phi(terminal)=0 every terminating trajectory from s0 earns -w*Phi(s0)."""
    d0 = 16.0
    short = torch.linspace(d0, 0.3, 160).tolist()                    # fast, straight
    detour = torch.cat([torch.linspace(d0, 19.0, 80), torch.linspace(19.0, 0.3, 400)]).tolist()   # goes AWAY first
    crash = torch.linspace(d0, 9.0, 50).tolist()
    expected = CFG.progress_weight * d0
    for traj, kind in ((short, "goal"), (detour, "goal"), (crash, "crash")):
        assert abs(discounted_shaping(traj, kind) - expected) < 1e-3


def test_terminal_components():
    total, c = compute_reward(torch.tensor([1.0, 5.0, 5.0]), torch.tensor([0.5, 5.0, 4.9]),
                              torch.tensor([True, False, False]), torch.tensor([False, True, False]), CFG)
    assert c["goal"].tolist() == [20.0, 0.0, 0.0]
    assert c["collision"].tolist() == [0.0, -10.0, 0.0]
    assert torch.allclose(c["time"], torch.full((3,), -0.0125))
    # non-terminal step: shaping = w * (gamma * -4.9 + 5.0)
    assert abs(c["progress"][2].item() - 0.5 * (5.0 - 0.997 * 4.9)) < 1e-6
    assert torch.allclose(total, sum(c.values()))


def test_moving_toward_goal_beats_hovering_per_step():
    _, a = compute_reward(torch.tensor([10.0]), torch.tensor([9.9]), torch.tensor([False]), torch.tensor([False]), CFG)
    _, b = compute_reward(torch.tensor([10.0]), torch.tensor([10.0]), torch.tensor([False]), torch.tensor([False]), CFG)
    assert a["progress"] > b["progress"]
