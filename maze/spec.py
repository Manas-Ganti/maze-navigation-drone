"""Maze config schema: load + validate a hand-authored maze YAML.

PURE (numpy + yaml only). Every other module -- the Isaac scene builder, the
analytic collision check, the voxel grid, the route checks -- reads geometry
from a :class:`MazeSpec`, never from the USD stage, so the sim, the collision
ground truth and the validation checklist cannot disagree about where a wall is.

Frame
-----
Maze-local, metres. Origin at the south-west floor corner; +x east, +y north,
+z up. The flyable interior is exactly ``[0, X] x [0, Y] x [0, Z]`` with
``(X, Y, Z) = bounds``. Perimeter walls are generated OUTSIDE that box, so a
point's clearance to the bounds is simply its distance to the box faces.
There is no ceiling mesh: flying above ``Z`` counts as a collision (the
analytic ceiling) -- the perimeter walls are ``Z`` tall, so nothing outside the
maze is visible below it.

Primitive schema (one YAML mapping per primitive)
-------------------------------------------------
  id:     unique name (becomes the USD prim name)
  type:   box | cylinder
  pos:    [x, y, z] CENTRE of the primitive
  size:   [sx, sy, sz] FULL extents (box only)
  radius, height:   (cylinder only; axis is always vertical)
  yaw_deg: rotation about +z (box only; default 0). Yaw-only is deliberate:
          it keeps the analytic SDF exact and covers every structural element
          the spec asks for (walls, partitions, gaps, pillars, hanging blocks).
  color:  [r, g, b] in 0..1 -- per-branch colours are the visual route cue
  tag:    wall | partition | pillar | hanging | frame | decor (free text, used
          in plots and reports; ``decor`` primitives are visual-only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

Vec3 = Tuple[float, float, float]

PRIMITIVE_TYPES = ("box", "cylinder")
DEFAULT_COLOR: Vec3 = (0.6, 0.6, 0.6)


@dataclass(frozen=True)
class Primitive:
    id: str
    type: str
    pos: Vec3
    size: Optional[Vec3] = None          # box: full extents
    radius: Optional[float] = None       # cylinder
    height: Optional[float] = None       # cylinder
    yaw_deg: float = 0.0
    color: Vec3 = DEFAULT_COLOR
    tag: str = "wall"
    collision: bool = True               # False for visual-only decor

    @property
    def half_extents(self) -> Vec3:
        """Axis-aligned half extents in the primitive's own (yawed) frame."""
        if self.type == "box":
            assert self.size is not None
            return (self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0)
        assert self.radius is not None and self.height is not None
        return (self.radius, self.radius, self.height / 2.0)

    def min_thickness(self) -> float:
        """Thinnest dimension -- the tunnelling-relevant one."""
        if self.type == "box":
            assert self.size is not None
            return float(min(self.size))
        assert self.radius is not None and self.height is not None
        return float(min(2.0 * self.radius, self.height))


@dataclass(frozen=True)
class Region:
    """Axis-aligned box used as a route signature (maze-local frame)."""

    pos: Vec3
    size: Vec3

    def contains(self, p: Sequence[float]) -> bool:
        return all(abs(p[i] - self.pos[i]) <= self.size[i] / 2.0 for i in range(3))


@dataclass(frozen=True)
class Route:
    name: str
    kind: str                       # "fast" | "slow"
    waypoints: Tuple[Vec3, ...]
    design_speed_mps: float         # speed the scripted flythrough flies it at
    signature: Tuple[Region, ...]   # regions ONLY this route passes through
    note: str = ""

    @property
    def length_m(self) -> float:
        return float(
            sum(math.dist(a, b) for a, b in zip(self.waypoints[:-1], self.waypoints[1:]))
        )


@dataclass(frozen=True)
class MazeSpec:
    name: str
    description: str
    bounds: Vec3
    start_pos: Vec3
    start_yaw_deg: float
    goal_pos: Vec3
    goal_radius_m: float
    perimeter_thickness_m: float
    perimeter_color: Vec3
    floor_color: Vec3
    primitives: Tuple[Primitive, ...]
    routes: Tuple[Route, ...]
    source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def colliders(self) -> List[Primitive]:
        """Interior primitives that the drone can hit (perimeter excluded:
        the analytic bounds check covers it exactly)."""
        return [p for p in self.primitives if p.collision]

    def perimeter_walls(self) -> List[Primitive]:
        """Four walls OUTSIDE the flyable box, Z tall (visual + PhysX collider)."""
        X, Y, Z = self.bounds
        t = self.perimeter_thickness_m
        c = self.perimeter_color
        return [
            Primitive("perim_s", "box", (X / 2, -t / 2, Z / 2), (X + 2 * t, t, Z), color=c, tag="perimeter"),
            Primitive("perim_n", "box", (X / 2, Y + t / 2, Z / 2), (X + 2 * t, t, Z), color=c, tag="perimeter"),
            Primitive("perim_w", "box", (-t / 2, Y / 2, Z / 2), (t, Y, Z), color=c, tag="perimeter"),
            Primitive("perim_e", "box", (X + t / 2, Y / 2, Z / 2), (t, Y, Z), color=c, tag="perimeter"),
        ]

    def route(self, kind: str) -> Route:
        matches = [r for r in self.routes if r.kind == kind]
        if len(matches) != 1:
            raise KeyError(f"{self.name}: expected exactly one '{kind}' route, found {len(matches)}")
        return matches[0]

    @property
    def center_offset(self) -> Vec3:
        """Maze-local -> env-local offset: the maze is centred on the env origin in x-y."""
        return (-self.bounds[0] / 2.0, -self.bounds[1] / 2.0, 0.0)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _vec3(value: Any, what: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{what} must be a 3-list, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _primitive(raw: Dict[str, Any], idx: int) -> Primitive:
    allowed = {"id", "type", "pos", "size", "radius", "height", "yaw_deg", "color", "tag", "collision"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"primitive #{idx} has unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
    pid = str(raw.get("id") or f"prim_{idx:03d}")
    ptype = str(raw.get("type", "box"))
    if ptype not in PRIMITIVE_TYPES:
        raise ValueError(f"primitive {pid}: type must be one of {PRIMITIVE_TYPES}, got {ptype!r}")
    pos = _vec3(raw["pos"], f"primitive {pid}.pos")
    color = _vec3(raw.get("color", DEFAULT_COLOR), f"primitive {pid}.color")
    common = dict(
        id=pid,
        type=ptype,
        pos=pos,
        color=color,
        tag=str(raw.get("tag", "wall")),
        collision=bool(raw.get("collision", True)),
    )
    if ptype == "box":
        if "size" not in raw or "radius" in raw or "height" in raw:
            raise ValueError(f"primitive {pid}: a box takes 'size' (and not radius/height)")
        size = _vec3(raw["size"], f"primitive {pid}.size")
        if min(size) <= 0:
            raise ValueError(f"primitive {pid}: size must be positive, got {size}")
        return Primitive(size=size, yaw_deg=float(raw.get("yaw_deg", 0.0)), **common)
    if "radius" not in raw or "height" not in raw or "size" in raw or "yaw_deg" in raw:
        raise ValueError(f"primitive {pid}: a cylinder takes 'radius' and 'height' (no size/yaw)")
    radius, height = float(raw["radius"]), float(raw["height"])
    if radius <= 0 or height <= 0:
        raise ValueError(f"primitive {pid}: radius/height must be positive")
    return Primitive(radius=radius, height=height, **common)


def _route(name: str, raw: Dict[str, Any]) -> Route:
    kind = str(raw.get("kind", name))
    if kind not in ("fast", "slow"):
        raise ValueError(f"route {name}: kind must be 'fast' or 'slow', got {kind!r}")
    wps = tuple(_vec3(w, f"route {name} waypoint") for w in raw["waypoints"])
    if len(wps) < 2:
        raise ValueError(f"route {name}: needs >= 2 waypoints")
    sig = tuple(
        Region(_vec3(r["pos"], f"route {name} signature pos"), _vec3(r["size"], f"route {name} signature size"))
        for r in raw.get("signature", [])
    )
    if not sig:
        raise ValueError(f"route {name}: needs >= 1 signature region (used to classify route choice)")
    return Route(
        name=name,
        kind=kind,
        waypoints=wps,
        design_speed_mps=float(raw["design_speed_mps"]),
        signature=sig,
        note=str(raw.get("note", "")),
    )


def maze_from_dict(data: Dict[str, Any], source: str = "") -> MazeSpec:
    prims = tuple(_primitive(p, i) for i, p in enumerate(data["primitives"]))
    ids = [p.id for p in prims]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate primitive ids: {dupes}")
    routes = tuple(_route(name, raw) for name, raw in data["routes"].items())
    per = data.get("perimeter", {})
    spec = MazeSpec(
        name=str(data["name"]),
        description=str(data.get("description", "")).strip(),
        bounds=_vec3(data["bounds"], "bounds"),
        start_pos=_vec3(data["start"]["pos"], "start.pos"),
        start_yaw_deg=float(data["start"].get("yaw_deg", 0.0)),
        goal_pos=_vec3(data["goal"]["pos"], "goal.pos"),
        goal_radius_m=float(data["goal"]["radius"]),
        perimeter_thickness_m=float(per.get("thickness", 0.3)),
        perimeter_color=_vec3(per.get("color", DEFAULT_COLOR), "perimeter.color"),
        floor_color=_vec3(data.get("floor_color", (0.25, 0.25, 0.27)), "floor_color"),
        primitives=prims,
        routes=routes,
        source=source,
        extra={k: v for k, v in data.items() if k not in _KNOWN_TOP_LEVEL},
    )
    spec.route("fast"), spec.route("slow")   # exactly one of each, or raise
    return spec


_KNOWN_TOP_LEVEL = {
    "name", "description", "bounds", "start", "goal", "perimeter", "floor_color", "primitives", "routes",
}


def load_maze(path: str | Path) -> MazeSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return maze_from_dict(data, source=str(path))
