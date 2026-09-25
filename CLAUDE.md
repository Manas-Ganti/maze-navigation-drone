# Drone Maze Speed-Navigation — Build Spec

## 1. Project summary

Train a single vision-based quadrotor navigation policy to fly through a
hand-built 3D maze from a fixed start to a fixed goal, learning to prefer
faster (riskier) routes over slower (safer) ones as training progresses.
Medium-scope robotics/RL portfolio project. No sim-to-real shift testing,
no domain-randomization sweep, no multi-agent — those are explicitly out
of scope for this build.

**Core deliverable:** a policy whose behavior visibly improves across
training checkpoints (slower/cautious → faster/skilled), demonstrated via
a metrics table (success rate, time-to-goal) and 2 short rendered
flythrough videos (early checkpoint vs. final checkpoint).

**Explicitly out of scope for this build:** domain randomization axes
(wind/latency/visual corruption/dynamics perturbation), procedural maze
generation, multi-agent/self-play, memory-buffer/long-context mechanisms.
Do not add these without an explicit spec change.

---

## 2. Platform

- Isaac Lab, using the built-in quadcopter demo as the dynamics/control
  base, OR Pegasus Simulator if PX4-style low-level control is preferred.
  Either is acceptable — do not spend build time re-deciding this.
- Training framework: PPO via RSL-RL or Stable-Baselines3 (match whatever
  the existing RLVR-portfolio environments use, for consistency).

---

## 3. Task definition

- **Action space (high-level):** velocity/waypoint commands (vx, vy, vz,
  yaw-rate), NOT raw thrust/motor output. Low-level attitude/thrust
  control is handled by a standard controller (PX4-style or the
  Isaac Lab default) and is not trained.
- **Observation space:**
  - Forward-facing RGB camera (fixed resolution, e.g. 84x84 or 128x128 —
    pick the smallest resolution that keeps corridors/gaps visually
    distinguishable; verify this during environment validation, §6).
  - Proprioception: linear velocity, angular velocity, orientation
    (quaternion or 6D rotation rep), and relative vector to next
    waypoint/goal.
  - No global map, no full maze layout handed to the policy, no memory
    buffer across steps beyond what a standard recurrent/frame-stack
    policy architecture provides if used.
- **Episode termination:** goal reached (success), collision (failure),
  or timeout (failure). Timeout horizon should be generous enough that
  even the slow-safe route comfortably completes within it during early
  training.

---

## 4. Reward design

Reward hacking and reward-term imbalance are the single biggest risk in
this project (see failure-mode analysis, §8) — treat this section as
requiring iteration, not a one-shot spec to implement and forget.

Components (all must be scale-normalized relative to each other, not just
summed as-authored):

1. **Potential-based distance shaping** — reward proportional to
   reduction in distance-to-goal (or distance-to-next-waypoint if using
   intermediate waypoints). Must be strictly potential-based
   (`shaped_reward = gamma * potential(s') - potential(s)`) so it doesn't
   distort the optimal policy, only accelerates learning it.
2. **Per-step time penalty** — small constant penalty per timestep, so
   faster completion accumulates less total penalty. This is what
   creates the speed incentive.
3. **Collision penalty** — meaningfully large negative reward + episode
   termination on collision. Tune this relative to (2): too small →
   reckless flying; too large → policy never attempts the fast/risky
   route. Expect to tune this iteratively against observed behavior,
   not just set once analytically.
4. **Goal-reached bonus** — terminal positive reward on success, larger
   in magnitude than the maximum possible accumulated time penalty for a
   slow-safe completion, so success always beats a faster failure.

**Validation requirement before training at scale:** log reward
component magnitudes (not just total reward) during initial short runs,
and confirm no single term dominates by more than ~1 order of magnitude
unless intentional.

---

## 5. Maze design (hand-built, config-driven — not GUI-authored)

### 5.1 Authoring approach

- Define each maze as a **data config** (YAML/JSON), listing every wall,
  pillar, platform, and gap as a primitive spec: `{type, position, size,
  rotation}`. Do not hand-place geometry directly in the Isaac Sim GUI —
  author from config via `isaaclab.sim` spawners
  (`sim_utils.CuboidCfg`, etc.) so layouts are version-controlled and
  regenerable.
- Build 2–3 distinct maze layouts, each its own config file, sharing one
  environment-construction script.

### 5.2 Structural elements per maze

- Bounded volume, e.g. 20m x 20m x 5m.
- **Corridors** of varying width — some wide (fast, forgiving), some
  narrow (require slowing/centering).
- **Low partition walls** (below ceiling height) the drone can optionally
  fly over as a vertical shortcut, vs. going around through the corridor.
- **Vertical gap shortcuts** — openings at a specific height connecting
  otherwise-distant corridor segments.
- **Hanging obstacles** (box/cylinder primitives suspended at various
  heights) inside corridors, forcing under/over/weave decisions — this
  is what punishes naive "fly fast down the middle" behavior.
- **Off-center pillars** in open rooms so open space isn't risk-free.
- Clearly marked start pad and goal pad (colored floor plane is enough).
- Optional: distinct wall materials/colors per corridor branch, to give
  the vision policy an actual visual cue for route identity and reduce
  visual-aliasing risk (see §8).

### 5.3 Route requirement (mandatory, per maze)

Each maze must contain at least two geometrically distinct start→goal
routes:
- **Slow-safe route:** wide corridors, generous clearance, ground-level
  throughout, no vertical shortcut required.
- **Fast-risky route:** shorter total path length, requiring at least one
  of: a narrow-gap squeeze, a vertical shortcut, or weaving between
  hanging obstacles.

**Hard requirement:** compute both route lengths geometrically from the
config (waypoint path length, not straight-line distance) before
training. Reject/redesign any layout where the fast/slow path-length
difference is under ~30% — too small a gap won't produce a visible
speed-improvement story.

---

## 6. Environment validation checklist (run before any training)

All of the following are cheap, scripted checks — do not skip them, they
catch the most common and most expensive-to-discover failures.

1. **Spawn validity:** confirm the drone's start pose does not intersect
   any collider (a common cause of physics-solver explosions at episode
   reset).
2. **Reachability:** confirm goal is reachable from start via a simple
   pathfinding/waypoint check against the maze config — not just visual
   inspection.
3. **Route length gap:** confirm fast vs. slow route path lengths differ
   by the required margin (§5.3).
4. **Scale/unit sanity:** confirm wall and gap dimensions are authored in
   consistent real-world units (meters) relative to the drone's collision
   radius — check narrow gaps are actually passable at the drone's
   physical size, not just visually plausible.
5. **Scripted flythrough:** fly a hardcoded/scripted controller (not the
   learned policy) through both the fast and slow routes once, at
   the speeds/gaps the maze is designed around. This is the primary
   check against wall-tunneling at speed and against collision-margin
   miscalibration, before spending any GPU-hours on a broken layout.
6. **Visual distinguishability spot-check:** render a few camera frames
   from mid-corridor in each branch and confirm a human can tell which
   corridor/branch they're looking at — if corridors are visually
   identical, the vision-based framing is undermined (§8).

Do not proceed to training until all six pass.

---

## 7. Training plan

- Train one policy per maze layout initially (simplest, lowest-risk
  path); optionally extend to training across all 2–3 layouts jointly if
  time permits, to get a weaker generalization signal — this is a
  stretch goal, not a requirement.
- Checkpoint regularly (e.g. every N updates) and retain checkpoints
  across training — these are what the improvement-curve demo depends
  on.
- Expect materially lower environment throughput than prior state-based
  RLVR environments (synthea, racetrack) due to camera rendering cost —
  budget wall-clock time accordingly and consider a smaller parallel-env
  count than usual as the default, not a fallback.

---

## 8. Known failure modes to actively guard against

(Carried from pre-build risk review — keep this section in the repo,
don't just treat it as planning scratch.)

- **Reward hacking via geometry exploits** — high-speed collisions
  "succeeding" due to tunneling through thin colliders. Mitigated by
  §6.5 scripted flythrough + adequate collider thickness/continuous
  collision detection.
- **Reward term imbalance** — collision penalty vs. time penalty
  miscalibrated, producing either reckless or permanently-cautious
  policies. Mitigated by §4's logging requirement and iterative tuning.
- **Sparse-reward exploration stall** — mitigated by potential-based
  shaping (§4.1).
- **Visual aliasing between corridors** — undermines the vision-based
  framing if the policy can't actually tell routes apart. Mitigated by
  §5.2's optional per-corridor materials and §6.6's spot-check.
- **Vision throughput bottleneck** — camera-based envs run far fewer
  parallel instances than state-based ones; this changes iteration speed
  materially. Plan for it (§7), don't discover it mid-project.
- **Checkpoint cherry-picking in the demo** — when rendering the
  early-vs-late comparison video, average over several rollouts per
  checkpoint before selecting which to render, so the comparison isn't
  a lucky/unlucky single sample.
- **Memorization vs. generalization ambiguity** — with only 2-3 hand
  layouts, don't claim general route-evaluation skill in the writeup;
  state the limitation explicitly.

---

## 9. Deliverables / definition of done

- [ ] 2-3 maze configs authored, all passing the §6 validation checklist
- [ ] Trained policy with logged reward-component breakdown showing no
      unintended term dominance
- [ ] Metrics table: success rate and mean time-to-goal at several
      training checkpoints (showing improvement over training)
- [ ] Route-choice check: confirm the final policy actually takes the
      fast route more often than early checkpoints did (not just faster
      execution of the same route — measure this explicitly)
- [ ] 2 rendered flythrough videos: early checkpoint vs. final checkpoint
- [ ] README: task description, one results table, the two videos/GIFs,
      explicit statement of scope limitations (§8's memorization caveat)