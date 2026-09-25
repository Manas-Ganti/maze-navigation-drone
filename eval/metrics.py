"""Aggregate episode records into the deliverable table. PURE (no Isaac, no torch).

Every number in the README's results table must come out of this module from
episodes that actually ran (the PROOF RULE). Rates carry Wilson 95% intervals
so a 128-episode checkpoint is not over-read.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Sequence

ROUTES = ("fast", "slow", "both", "none")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """episodes: dicts with outcome, time_s, route, path_length_m, min_clearance_m, mean_speed_mps."""
    n = len(episodes)
    succ = [e for e in episodes if e["outcome"] == "success"]
    k = len(succ)
    lo, hi = wilson(k, n)
    times = [e["time_s"] for e in succ]
    out: Dict[str, Any] = {
        "episodes": n,
        "success_rate": k / n if n else float("nan"),
        "success_ci_lo": lo,
        "success_ci_hi": hi,
        "collision_rate": sum(e["outcome"] == "collision" for e in episodes) / n if n else float("nan"),
        "timeout_rate": sum(e["outcome"] == "timeout" for e in episodes) / n if n else float("nan"),
        "time_to_goal_mean_s": statistics.fmean(times) if times else float("nan"),
        "time_to_goal_median_s": statistics.median(times) if times else float("nan"),
        "time_to_goal_std_s": statistics.pstdev(times) if len(times) > 1 else float("nan"),
        "mean_speed_success_mps": statistics.fmean(e["mean_speed_mps"] for e in succ) if succ else float("nan"),
        "min_clearance_success_m": statistics.fmean(e["min_clearance_m"] for e in succ) if succ else float("nan"),
    }
    for r in ROUTES:
        out[f"route_{r}"] = sum(e["route"] == r for e in episodes) / n if n else float("nan")
        out[f"success_route_{r}"] = sum(e["route"] == r for e in succ) / k if k else float("nan")
    # success rate PER route: is the fast route getting safer, not just more popular?
    for r in ("fast", "slow"):
        on = [e for e in episodes if e["route"] == r]
        out[f"success_rate_on_{r}"] = sum(e["outcome"] == "success" for e in on) / len(on) if on else float("nan")
        t = [e["time_s"] for e in on if e["outcome"] == "success"]
        out[f"time_on_{r}_s"] = statistics.fmean(t) if t else float("nan")
    return out


def _f(x: float, pct: bool = False, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    """rows: summarize() dicts plus 'label' (e.g. 'iter 500' or 'scripted fast')."""
    head = ("| checkpoint | episodes | success (95% CI) | collision | timeout | time-to-goal s (median) "
            "| fast route | slow route | success on fast |")
    sep = "|" + "---|" * 9
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['episodes']} | {_f(r['success_rate'], True)} "
            f"({_f(r['success_ci_lo'], True)}-{_f(r['success_ci_hi'], True)}) | {_f(r['collision_rate'], True)} "
            f"| {_f(r['timeout_rate'], True)} | {_f(r['time_to_goal_mean_s'])} ({_f(r['time_to_goal_median_s'])}) "
            f"| {_f(r['route_fast'], True)} | {_f(r['route_slow'], True)} | {_f(r['success_rate_on_fast'], True)} |"
        )
    return "\n".join(lines)
