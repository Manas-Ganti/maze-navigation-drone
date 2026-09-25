"""Flythrough videos of a checkpoint -- early vs final (CLAUDE.md s9). VISUALS ONLY.

    python eval/record_video.py --headless --run results/runs/a_s42 --checkpoints 250 2000

For each checkpoint: fly ``video.takes_per_checkpoint`` episodes (same env,
start noise on), write EVERY take to disk, then publish the MEDIAN-REPRESENTATIVE
take -- the modal outcome, and within it the take whose time is closest to that
group's median. That is the s8 guard against cherry-picking: the selection rule
is fixed before looking, and all takes plus their stats are in manifest.json.

Each frame: an overview camera over the maze + a trail of the drone's path,
with the policy's own camera frame (what it actually sees, 64x64) inset.

Output: results/videos/<run>/iter_<N>/take_<k>.mp4, representative.mp4 / .gif, manifest.json.
Success RATES come from eval/evaluate.py, never from these clips.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.app import launch, make_env, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoints", type=int, nargs="+", required=True, help="e.g. an early one and the last")
    ap.add_argument("--takes", type=int, default=None, help="override video.takes_per_checkpoint")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        AppLauncher.add_app_launcher_args(ap)
    except ImportError:
        ap.add_argument("--headless", action="store_true")
    return ap.parse_args()


def pick_representative(takes: List[Dict[str, Any]]) -> int:
    """Modal outcome; within it, the take closest to the group's median time. Pure, tested."""
    modal = Counter(t["outcome"] for t in takes).most_common(1)[0][0]
    group = [i for i, t in enumerate(takes) if t["outcome"] == modal]
    med = statistics.median(takes[i]["time_s"] for i in group)
    return min(group, key=lambda i: (abs(takes[i]["time_s"] - med), i))


def main() -> None:
    args = parse_args()
    from training.config import apply_overrides, validate_config  # noqa: PLC0415
    from training.ppo_config import list_checkpoints  # noqa: PLC0415

    run_dir = Path(args.run)
    data = json.loads((run_dir / "resolved_config.json").read_text())
    data["env"]["num_envs"] = 1
    data["env"]["rerender_on_reset"] = True
    data["logging"]["wandb"]["enabled"] = False
    data["seed"] = int(data["seed"]) + 2000
    data = apply_overrides(data, args.set)
    validate_config(data)
    vid = data["video"]
    takes_n = args.takes or int(vid["takes_per_checkpoint"])

    app = launch(args, data, force_cameras=True)
    set_global_seed(int(data["seed"]))

    import imageio.v2 as imageio  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.envs import ViewerCfg  # noqa: PLC0415
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: PLC0415
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

    from eval.policies import CheckpointPolicy  # noqa: PLC0415
    from maze.spec import load_maze  # noqa: PLC0415

    maze = load_maze(data["maze"]["config"])
    off = maze.center_offset
    to_env = lambda p: tuple(float(p[i]) + off[i] for i in range(3))   # noqa: E731
    # VERIFY ON ARC: ViewerCfg fields (Isaac Lab 2.1); origin_type "env" = coordinates relative to env 0.
    viewer = ViewerCfg(eye=to_env(vid["overview_eye"]), lookat=to_env(vid["overview_lookat"]),
                       resolution=tuple(int(v) for v in vid["resolution"]), origin_type="env", env_index=0)
    env, _ = make_env(data, render_mode="rgb_array", viewer=viewer)
    env.record_episodes = True

    trail = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/Trail",
        markers={"dot": sim_utils.SphereCfg(radius=0.06, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.9, 1.0), emissive_color=(0.0, 0.4, 0.5)))},
    ))

    ckpts = list_checkpoints(run_dir)
    policy = CheckpointPolicy(RslRlVecEnvWrapper(env), data, deterministic=bool(data["eval"]["deterministic"]))
    every = int(vid["frame_every_n_steps"])
    fps = 1.0 / (env.policy_dt * every)
    scale = int(vid["fpv_inset_scale"])
    max_trail = int(vid["trail_points"])
    out_root = Path("results/videos") / run_dir.name

    def compose(frame: np.ndarray, fpv: np.ndarray | None, label: str) -> np.ndarray:
        img = Image.fromarray(frame[..., :3])
        if fpv is not None:
            inset = Image.fromarray(fpv).resize((fpv.shape[1] * scale, fpv.shape[0] * scale), Image.NEAREST)
            x = img.width - inset.width - 12
            ImageDraw.Draw(img).rectangle([x - 3, 9, x + inset.width + 2, 12 + inset.height], fill=(255, 255, 255))
            img.paste(inset, (x, 12))
            ImageDraw.Draw(img).text((x, 16 + inset.height), "policy camera", fill=(255, 255, 255))
        ImageDraw.Draw(img).text((12, 12), label, fill=(255, 255, 255))
        return np.asarray(img)

    for it in args.checkpoints:
        if it not in ckpts:
            raise SystemExit(f"no model_{it}.pt in {run_dir}; available {list(ckpts)}")
        policy.load(str(ckpts[it]))
        out = out_root / f"iter_{it}"
        out.mkdir(parents=True, exist_ok=True)
        takes: List[Dict[str, Any]] = []
        for k in range(takes_n):
            obs, _ = env.reset()
            env.drain_records()
            for _ in range(3):
                env.render()   # first frames after a reset can be blank
            path = out / f"take_{k + 1}.mp4"
            writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=16)
            pts: List[List[float]] = []
            rec = None
            for step in range(int(env.max_episode_length) + 2):
                obs, _, term, trunc, _ = env.step(policy.act(obs["policy"]))
                recs = env.drain_records()
                if recs:
                    rec = recs[0]
                    break   # env already auto-reset: the next frame would show the new spawn
                pts.append(env.robot.data.root_pos_w[0].tolist())
                trail.visualize(translations=torch.tensor(pts[-max_trail:], device=env.device))
                if step % every == 0:
                    fpv = None
                    if env.camera is not None:
                        fpv = env.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                    label = f"{maze.name}  iter {it}  take {k + 1}  t={step * env.policy_dt:4.1f}s"
                    writer.append_data(compose(env.render(), fpv, label))
            writer.close()
            takes.append({"file": path.name, "outcome": rec.outcome if rec else "unknown",
                          "time_s": rec.time_s if rec else float("nan"), "route": rec.route if rec else "unknown",
                          "min_clearance_m": rec.min_clearance_m if rec else float("nan")})
            print(f"[video] iter {it} take {k + 1}: {takes[-1]}", flush=True)

        sel = pick_representative(takes)
        shutil.copy(out / takes[sel]["file"], out / "representative.mp4")
        frames = imageio.mimread(str(out / "representative.mp4"), memtest=False)
        small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::2]]
        imageio.mimsave(str(out / "representative.gif"), small, duration=2 * 1000.0 / fps, loop=0)
        (out / "manifest.json").write_text(json.dumps({
            "checkpoint": str(ckpts[it]), "iteration": it, "takes": takes, "selected": sel,
            "rule": "modal outcome, then time closest to that group's median (fixed before recording)",
        }, indent=2))
        print(f"[video] iter {it}: representative = take {sel + 1} ({takes[sel]['outcome']}, "
              f"{takes[sel]['time_s']:.1f} s, route {takes[sel]['route']})")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
