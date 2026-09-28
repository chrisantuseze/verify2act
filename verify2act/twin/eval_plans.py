"""Plan-ranking evaluation on held-out twin episodes with simulated ground truth.

For each start frame of a single-step goal (task2a/2b/3a and the side/stack families), every valid subtask is executed
in the twin (no failures) and labelled by the goal predicate. Each (world model, critic) config imagines every plan from
the start image and scores it with the language goal; we report
  top1     = P(the highest-scored plan achieves the goal)          (what the planner acts on)
  pairwise = P(score(correct plan) > score(wrong plan))            (0.5 = chance)
  acc@thr  = accuracy of accept/reject at the goal threshold

python -m verify2act.twin.eval_plans --goal-head verify2act/output/goal_head/v1/goal_head_best.pt
"""
import argparse, json, os, random, re
from types import SimpleNamespace

import numpy as np, torch
from PIL import Image

from verify2act.pipeline.inference import _build_critic
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.twin.goal_labels import on, side_of
from verify2act.twin.scene import DofbotTwin, InvalidSubtask

D = "verify2act/data/twin/dofbot_v1"
CALVIN = "verify2act/output/v2a_wm/calvin"
TWIN = "verify2act/output/v2a_wm/dofbot_twin"


def goal_predicate(text):
    m = re.search(r"the (\w+) block to the (left|right) of the (\w+) block", text)
    if m:
        c, side, b = m.groups()
        return lambda s: side_of(s, c, b, side)
    m = re.search(r"the (\w+) block on (?:top of )?the (\w+) block", text)
    if m:
        c, b = m.groups()
        return lambda s: on(s, c, b)
    return None


def val_episodes(val_frac=0.05):
    ev = sorted(os.listdir(f"{D}/episodes")); random.Random(0).shuffle(ev)
    return set(ev[:int(val_frac * len(ev))])


def build_frames(n, seed=0):
    val = val_episodes()
    first = {}
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        if r["timestep"] == 0 and r["episode_id"] in val and r["num_steps"] == 1 and goal_predicate(r["lang_goal"]):
            first[r["episode_id"]] = r
    rows = sorted(first.values(), key=lambda r: r["episode_id"])
    random.Random(seed).shuffle(rows)
    # eval tasks first, then side/stack families
    rows = [r for r in rows if r["task"]] + [r for r in rows if not r["task"]]
    tw = DofbotTwin(render=False); tw.cfg["placement"]["p_fail"] = 0.0
    frames = []
    for r in rows[:n]:
        s0 = json.load(open(f"{D}/episodes/{r['episode_id']}/states.json"))[0]
        pred = goal_predicate(r["lang_goal"])
        tw.set_state(s0)
        plans, labels = [], []
        for sub in tw.valid_subtasks():
            tw.set_state(s0)
            try:
                tw.apply(sub, np.random.default_rng(0))
            except (InvalidSubtask, RuntimeError):
                continue
            plans.append([sub]); labels.append(bool(pred(tw.get_state())))
        if any(labels) and not all(labels):
            img = np.array(Image.open(f"{D}/{r['image_t']}").convert("RGB"))
            frames.append((r["episode_id"], r["task"] or r["family"], r["lang_goal"], img, plans, labels))
    return frames


def load_wm(path, encoder=f"{CALVIN}/encoder/ckpt/delta_encoder_best.pt", dev="cuda"):
    return LatentWorldModel(device=dev, dynamics_weights_path=path, encoder_ckpt=encoder,
                            history_len=3, token_dim=128, num_latent_tokens=32, action_conditioning="cross_attn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v1/goal_head_best.pt")
    ap.add_argument("--twin-critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    ap.add_argument("--configs", default="twin+pooled,twin+goalhead,calvin+goalhead",
                    help="wm+critic pairs; wm in {calvin, twin, twin2 (--twin2-wm/--twin2-encoder)}")
    ap.add_argument("--twin2-wm", default="verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--twin2-encoder", default="verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_plans.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    frames = build_frames(a.n)
    print(f"{len(frames)} frames, {np.mean([len(f[4]) for f in frames]):.1f} plans/frame, "
          f"{np.mean([np.mean(f[5]) for f in frames]):.2f} correct frac", flush=True)
    wms, res = {}, {}
    for cfg in a.configs.split(","):
        wn, cn = cfg.split("+")
        if wn not in wms:
            wms[wn] = (load_wm(a.twin2_wm, a.twin2_encoder) if wn == "twin2" else
                       load_wm(f"{TWIN}/wm_old/ckpt/latent_dynamics_best_weights.pt" if wn == 'twin' else f"{CALVIN}/wm/ckpt/latent_dynamics_best_weights.pt"))
        critic = _build_critic(SimpleNamespace(critic_ckpt=a.twin_critic,
                                               goal_head_ckpt=a.goal_head if cn == "goalhead" else None), dev)
        thr = 0.5 if cn == "goalhead" else 0.05
        bp = BeamSearchPlanner(vlm_planner=None, world_model=wms[wn], critic=critic, beam_width=1, goal_threshold=thr,
                               temporal_threshold=-1e9, max_retries=1, max_replans=0, wm_mode="v2a_wm")
        per = []
        by_task = {}
        for name, task, goal, img, plans, labels in frames:
            sc = []
            for p in plans:
                s, *_ = bp._evaluate_trajectory(plan=p, current_image_np=img, language_goal=goal, decoder=None)
                sc.append(float(s) if s is not None and np.isfinite(s) else -1e9)
            sc, lab = np.array(sc), np.array(labels)
            pair = float(np.mean([[p > q for q in sc[~lab]] for p in sc[lab]]))
            top1 = float(lab[int(np.argmax(sc))])
            acc = float(np.mean((sc >= thr) == lab))
            per.append({"frame": name, "task": task, "pairwise": pair, "top1": top1, "acc_thr": acc})
            by_task.setdefault(task, []).append((pair, top1))
        res[cfg] = {"pairwise": float(np.mean([x["pairwise"] for x in per])), "top1": float(np.mean([x["top1"] for x in per])),
                    "acc_thr": float(np.mean([x["acc_thr"] for x in per])),
                    "by_task": {t: {"n": len(v), "pairwise": float(np.mean([x[0] for x in v])), "top1": float(np.mean([x[1] for x in v]))}
                                for t, v in sorted(by_task.items())}, "per_frame": per}
        print(f"== {cfg}: pairwise={res[cfg]['pairwise']:.3f} top1={res[cfg]['top1']:.3f} acc@thr={res[cfg]['acc_thr']:.3f}", flush=True)
        for t, v in res[cfg]["by_task"].items():
            print(f"     {t:8s} n={v['n']:3d} pairwise={v['pairwise']:.3f} top1={v['top1']:.3f}", flush=True)
        del critic
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
