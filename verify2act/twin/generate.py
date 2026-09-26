"""Generate a twin dataset in the robosuite layout the WM and critic trainers already read.

    <out>/episodes/ep_000123/frame_00000.jpg ...   one frame per state (initial + after every subtask)
    <out>/episodes/ep_000123/goal.jpg              final state, re-rendered with another camera/photometric draw
    <out>/episodes/ep_000123/states.json           block poses per frame
    <out>/transitions.jsonl                         one row per subtask (image_t, image_t1, action_text, lang_goal, ...)
    <out>/metadata.json

    MUJOCO_GL=egl python -m verify2act.twin.generate --out verify2act/output/twin/dofbot_v1 --num-episodes 20000 --workers 8

See TWIN_DATASET.md at the repo root for the full procedure (server setup, generation, training).
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from verify2act.twin import augment
from verify2act.twin.config import ASSETS_DIR, load_config
from verify2act.twin.scene import DofbotTwin, InvalidSubtask
from verify2act.twin.tasks import EVAL_TASKS, FAMILY_WEIGHTS, eval_goal, run_goal, sample_goal


def _prelude(t: DofbotTwin, rng: np.random.Generator, n: int) -> None:
    """Unrecorded random subtasks, so initial states include removed blocks and stacks."""
    for _ in range(n):
        cands = t.valid_subtasks()
        if not cands or sum(t.present.values()) <= 2:
            return
        try:
            t.apply(str(rng.choice(cands)), rng)
        except InvalidSubtask:
            pass


def _random_walk(t: DofbotTwin, rng: np.random.Generator, n: int, on_step) -> List[str]:
    done: List[str] = []
    for _ in range(n):
        cands = t.valid_subtasks()
        rng.shuffle(cands)
        for s in cands[:10]:
            try:
                t.apply(str(s), rng)
            except InvalidSubtask:
                continue
            done.append(str(s))
            on_step(str(s))
            break
    return done


def make_episode(t: DofbotTwin, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """One episode in memory (frames, subtasks, goal); None when the sampled task could not be executed."""
    ec = t.cfg["episodes"]
    random_walk = rng.random() < ec["random_walk_frac"]
    weights = ec.get("family_weights") or FAMILY_WEIGHTS
    family = None if random_walk else str(rng.choice(list(weights), p=_norm(list(weights.values()))))
    is_eval = family == "eval"               # eval scenes: all four blocks on the sheet, nothing moved yet
    n = 4 if is_eval or rng.random() < ec["p_n_blocks_4"] else int(rng.integers(ec["n_blocks"][0],
                                                                                   ec["n_blocks"][1] + 1))
    t.randomize_episode(rng)                 # first: the layout must be in this episode's camera view
    t.reset(rng, [str(c) for c in rng.permutation(t.colors)[:n]])
    eval_task = None
    if is_eval:
        # Uniform over the eval tasks: re-draw the scene (not the task) until it can host the task, so the tasks
        # needing room (2a/2b) are not under-represented.
        eval_task = str(rng.choice(sorted(EVAL_TASKS)))
        for _ in range(10):
            if eval_goal(t, rng, eval_task) is not None:
                break
            t.reset(rng)
        else:
            return None
    if not is_eval and rng.random() < ec["p_prelude"]:
        _prelude(t, rng, int(rng.integers(ec["prelude_steps"][0], ec["prelude_steps"][1] + 1)))
    photo = augment.sample_params(t.cfg, rng)

    def shot() -> np.ndarray:
        return augment.apply(t.render(rng), photo, rng)

    frames, states = [shot()], [t.get_state()]

    def on_step(_s: str) -> None:
        frames.append(shot())
        states.append(t.get_state())

    task = ""
    try:
        if random_walk:
            steps = _random_walk(t, rng, int(rng.integers(ec["random_walk_steps"][0], ec["random_walk_steps"][1] + 1)),
                                 on_step)
            goal_text, family, success = "", "random", False
        else:
            goal = eval_goal(t, rng, eval_task) if is_eval else sample_goal(t, rng, family=family)
            if goal is None:
                return None
            steps = run_goal(t, goal, rng, on_step)
            goal_text, family, success, task = goal.text, goal.family, True, goal.task
    except (InvalidSubtask, RuntimeError):
        return None
    if not steps:
        return None
    # Goal image: the final state after another return to the home pose (episode jitter), with new lighting and a new
    # photometric draw, so it is not pixel-identical to the last frame (the goal head's anchor and positive).
    t.randomize_appearance(rng)
    goal_cam = t._jitter(t.episode_camera, t.cfg["camera"]["jitter_episode"], rng)
    goal_img = augment.apply(t.render(rng, cam=goal_cam), augment.sample_params(t.cfg, rng), rng)
    return {"frames": frames, "states": states, "steps": steps, "goal_image": goal_img,
            "lang_goal": goal_text, "family": family, "task": task, "success": success}


def _norm(w: List[float]) -> np.ndarray:
    a = np.asarray(w, float)
    return a / a.sum()


def write_episode(out: Path, ep_id: str, ep: Dict[str, Any], fmt: str = "jpg") -> List[Dict[str, Any]]:
    """Frames as JPEG (quality 95) by default: they already carry the webcam's JPEG artefacts, and PNG is ~4x larger."""
    d = out / "episodes" / ep_id
    d.mkdir(parents=True, exist_ok=True)
    kw = {"quality": 95} if fmt == "jpg" else {}
    for i, f in enumerate(ep["frames"]):
        Image.fromarray(f).save(d / f"frame_{i:05d}.{fmt}", **kw)
    Image.fromarray(ep["goal_image"]).save(d / f"goal.{fmt}", **kw)
    (d / "states.json").write_text(json.dumps(ep["states"]))
    rows = []
    for i, s in enumerate(ep["steps"]):
        rows.append({
            "episode_id": ep_id, "timestep": i,
            "image_t": f"episodes/{ep_id}/frame_{i:05d}.{fmt}", "image_t1": f"episodes/{ep_id}/frame_{i + 1:05d}.{fmt}",
            "goal_image": f"episodes/{ep_id}/goal.{fmt}", "goal_image_source": "rerender",
            "action_text": s, "lang_goal": ep["lang_goal"], "family": ep["family"], "task": ep.get("task", ""),
            "episode_success": ep["success"], "state_t": i, "state_t1": i + 1,
            "policy_type": "twin_oracle", "num_steps": len(ep["steps"]),
        })
    return rows


def _worker(args: Tuple[str, Optional[str], int, int, int, str, str]) -> List[Dict[str, Any]]:
    out, cfg_path, start, count, seed, assets, fmt = args
    t = DofbotTwin(load_config(cfg_path), assets_dir=assets)
    rng = np.random.default_rng([seed, start])
    rows: List[Dict[str, Any]] = []
    idx, failures = start, 0
    while idx < start + count:
        ep = make_episode(t, rng)
        if ep is None:
            failures += 1
            if failures > 20 * count + 100:
                raise RuntimeError("too many failed episodes; check the scene config")
            continue
        rows.extend(write_episode(Path(out), f"ep_{idx:06d}", ep, fmt))
        idx += 1
    t.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk", type=int, default=50, help="episodes per worker task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--assets", default=str(ASSETS_DIR))
    ap.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--start-episode", type=int, default=0, help="append: first episode index (keeps existing rows)")
    ap.add_argument("--allow-flat", action="store_true", help="run without the real-image textures (flat colours)")
    args = ap.parse_args()

    needed = ["wood.png", "sheet.png"] + [f"{c}_{f}.png" for c in load_config(args.config)["blocks"]["colors"]
                                          for f in ("px", "nx", "py", "ny", "pz", "nz")]
    missing = [n for n in needed if not (Path(args.assets) / n).exists()]
    if missing and not args.allow_flat:
        raise SystemExit(f"missing textures in {args.assets}: {missing[:4]}{' ...' if len(missing) > 4 else ''}\n"
                         "build them with `python -m verify2act.twin.textures --frames <real frames>` (TWIN_DATASET.md)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(out), args.config, s, min(args.chunk, args.start_episode + args.num_episodes - s), args.seed,
             args.assets, args.image_format)
            for s in range(args.start_episode, args.start_episode + args.num_episodes, args.chunk)]
    t0 = time.time()
    mode = "a" if args.start_episode > 0 else "w"
    n_rows = 0
    count_family, count_task = collections.Counter(), collections.Counter()
    ctx = mp.get_context("spawn")        # one EGL context per process
    with ctx.Pool(args.workers) as pool, open(out / "transitions.jsonl", mode) as f:
        for i, rows in enumerate(pool.imap_unordered(_worker, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                if r["timestep"] == 0:
                    count_family[r["family"]] += 1
                    count_task[r["task"] or "-"] += 1
            n_rows += len(rows)
            print(f"[{i + 1}/{len(jobs)}] {n_rows} transitions, {time.time() - t0:.0f}s", flush=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    meta = {"generator": "verify2act.twin.generate", "git_commit": commit, "seed": args.seed,
            "num_episodes": args.start_episode + args.num_episodes, "transitions_written": n_rows,
            "episodes_by_family": dict(count_family), "episodes_by_eval_task": dict(count_task),
            "image_format": args.image_format,
            "assets": sorted(p.name for p in Path(args.assets).glob("*.png")),
            "config": load_config(args.config), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {n_rows} transitions in {time.time() - t0:.0f}s -> {out}")
    print(f"episodes by family: {dict(count_family)}")
    print(f"eval tasks: {dict(count_task)}")


if __name__ == "__main__":
    main()
