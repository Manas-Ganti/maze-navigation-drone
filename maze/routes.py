"""Route bookkeeping: signature regions, route-choice classification, and the
risk/speed break-even the reward is tuned around.

PURE (torch). The classifier is the measurement behind the deliverable "the
final policy takes the fast route more often than early checkpoints did (not
just faster execution of the same route)" -- CLAUDE.md section 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from maze.spec import MazeSpec, Route

ROUTE_FAST = "fast"
ROUTE_SLOW = "slow"
ROUTE_NONE = "none"      # touched no signature region (e.g. crashed near the start)
ROUTE_BOTH = "both"      # touched both (e.g. tried one branch, turned back)


class RouteTracker:
    """Per-env flags: has this episode's trajectory entered each route's signature regions?

    A route counts as taken when the drone has entered EVERY one of its
    signature regions (routes may list several, e.g. entry and exit of a gap).
    Updated every policy step on-device; cheap point-in-box tests.
    """

    def __init__(self, spec: MazeSpec, n_envs: int, device: str | torch.device = "cpu"):
        self.kinds: List[str] = [ROUTE_FAST, ROUTE_SLOW]
        regions = []
        owner = []
        for ri, kind in enumerate(self.kinds):
            for reg in spec.route(kind).signature:
                regions.append((reg.pos, reg.size))
                owner.append(ri)
        self.center = torch.tensor([r[0] for r in regions], dtype=torch.float32, device=device)
        self.half = torch.tensor([r[1] for r in regions], dtype=torch.float32, device=device) / 2.0
        self.owner = torch.tensor(owner, dtype=torch.long, device=device)
        self.hit = torch.zeros(n_envs, len(regions), dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.hit[env_ids] = False

    def update(self, pos_local: torch.Tensor) -> None:
        inside = ((pos_local[:, None, :] - self.center[None]).abs() <= self.half[None]).all(-1)
        self.hit |= inside

    def taken(self) -> torch.Tensor:
        """(N, 2) bool: [fast_taken, slow_taken]."""
        out = []
        for ri in range(len(self.kinds)):
            cols = self.hit[:, self.owner == ri]
            out.append(cols.all(-1))
        return torch.stack(out, dim=-1)


def classify(taken_fast: bool, taken_slow: bool) -> str:
    if taken_fast and taken_slow:
        return ROUTE_BOTH
    if taken_fast:
        return ROUTE_FAST
    if taken_slow:
        return ROUTE_SLOW
    return ROUTE_NONE


def classify_path(spec: MazeSpec, points: torch.Tensor) -> str:
    """Classify a single trajectory (T, 3) maze-local -- used on authored routes as a self-check."""
    tr = RouteTracker(spec, 1, points.device)
    for p in points:
        tr.update(p[None])
    f, s = tr.taken()[0].tolist()
    return classify(f, s)


def densify(waypoints: Sequence[Sequence[float]], step_m: float = 0.05) -> torch.Tensor:
    """Polyline -> (T, 3) points spaced <= step_m."""
    wp = torch.tensor(waypoints, dtype=torch.float32)
    pts = [wp[:1]]
    for a, b in zip(wp[:-1], wp[1:]):
        n = max(1, int(torch.ceil(torch.linalg.norm(b - a) / step_m)))
        t = torch.linspace(0, 1, n + 1)[1:, None]
        pts.append(a + (b - a) * t)
    return torch.cat(pts)


# ----------------------------------------------------------------------
# Break-even analysis: at what crash probability is the fast route worth it?
# ----------------------------------------------------------------------
@dataclass
class BreakEven:
    fast_time_s: float
    slow_time_s: float
    fast_return: float        # discounted return of a successful fast episode (shaping excluded)
    slow_return: float
    crash_return: float       # discounted return of a crash at the fast route's midpoint
    p_crash_break_even: float  # fast is preferred iff P(crash on fast) < this

    def as_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def break_even(fast: Route, slow: Route, step_penalty: float, goal_bonus: float,
               collision_penalty: float, gamma: float, policy_dt: float,
               fast_speed: float | None = None, slow_speed: float | None = None) -> BreakEven:
    """Expected-return comparison of the two routes under the configured reward.

    Shaping is excluded on purpose: with Phi(terminal) = 0 the discounted
    shaping return from the start state is the same constant for every
    trajectory (Ng et al. 1999), so it cannot favour either route. What
    decides the route is time penalty + discounting vs. collision risk -- and
    this function says where the crossover sits. A crossover near 0 means the
    policy should (almost) never risk the fast route; near 1 means risk is
    irrelevant. Tune ``collision_penalty`` so it lands somewhere informative.
    """
    vf = fast_speed or fast.design_speed_mps
    vs = slow_speed or slow.design_speed_mps
    tf, ts = fast.length_m / vf, slow.length_m / vs

    def ret(t_s: float, terminal: float) -> float:
        n = int(round(t_s / policy_dt))
        disc_steps = sum(gamma ** k for k in range(n))
        return -step_penalty * disc_steps + (gamma ** n) * terminal

    r_fast = ret(tf, goal_bonus)
    r_slow = ret(ts, goal_bonus)
    r_crash = ret(tf / 2.0, -collision_penalty)
    # p * r_crash + (1 - p) * r_fast = r_slow
    denom = r_fast - r_crash
    p = (r_fast - r_slow) / denom if denom > 0 else 0.0
    return BreakEven(tf, ts, r_fast, r_slow, r_crash, max(0.0, min(1.0, p)))
