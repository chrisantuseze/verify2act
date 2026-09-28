"""Re-score logged real planning frames with different (world model, critic) pairs.

For every real request frame whose executed history is empty (initial state, goal not yet met) enumerate the valid
candidate plans; the ones that satisfy the goal are "correct". The planner accepts a plan by its goal score, so a
useful critic gives correct plans a higher goal score than wrong ones. Reports per pair:
  pairwise = P(goal score of a correct plan > that of a wrong plan)   (0.5 = chance)
  margin   = mean(correct) - mean(wrong)
  tc       = mean temporal-head score over all imagined steps
"""
import argparse, glob, itertools, json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from verify2act.pipeline.inference import _build_critic
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.pipeline.world_model import LatentWorldModel

COLORS = ["red", "green", "blue", "yellow"]
CALVIN = "verify2act/output/v2a_wm/calvin"
TWIN = "verify2act/output/v2a_wm/dofbot_twin"


def single(c, rel, b=None):
    return f"pick and place {c} block into the bin" if rel == "bin" else f"pick and place {c} block {rel} {b} block"


def candidates(goal):
    g = goal.lower()
    if "left of the blue block" in g and "red" in g:          # task2a
        plans = [[single(c, "bin")] for c in COLORS]
        for c, b in itertools.permutations(COLORS, 2):
            plans += [[single(c, r, b)] for r in ("on", "to the left of", "to the right of")]
        good = {(single("red", "to the left of", "blue"),), (single("blue", "to the right of", "red"),)}
        return plans, [tuple(p) in good for p in plans]
    if "blue block and the yellow block into the bin" in g:   # task1a
        plans = [[single(a, "bin"), single(b, "bin")] for a, b in itertools.permutations(COLORS, 2)]
        return plans, [{p[0].split()[3], p[1].split()[3]} == {"blue", "yellow"} for p in plans]
    return None, None


def load_wm(wm, enc, dev):
    return LatentWorldModel(device=dev, dynamics_weights_path=wm, encoder_ckpt=enc, history_len=3, token_dim=128,
                            num_latent_tokens=32, action_conditioning="cross_attn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="verify2act/output/real")
    ap.add_argument("--twin-dir", default=None, help="evaluate on N twin task2a start frames instead of real logs")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="verify2act/output/real_eval/results.json")
    ap.add_argument("--twin-wm", default=f"{TWIN}/wm/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--twin-critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    a = ap.parse_args()
    dev = torch.device("cuda")
    enc = f"{CALVIN}/encoder/ckpt/delta_encoder_best.pt"
    wms = {"calvin": load_wm(f"{CALVIN}/wm/ckpt/latent_dynamics_best_weights.pt", enc, "cuda"),
           "twin": load_wm(a.twin_wm, enc, "cuda")}
    critics = {"calvin": _build_critic(SimpleNamespace(critic_ckpt="verify2act/output/contrastive/calvin/best_contrastive_critic.pt"), dev),
               "twin": _build_critic(SimpleNamespace(critic_ckpt=a.twin_critic), dev)}

    frames = []
    for f in sorted(glob.glob(f"{a.real}/*/imagination_logs/planning_call_*/request.json")):
        r = json.load(open(f))
        plans, good = candidates(r["goal"])
        if r.get("history") or plans is None:
            continue
        img = np.array(Image.open(f.replace("request.json", "request_image.png")).convert("RGB"))
        frames.append((f.split("real/")[1].split("/imagination")[0] + "/" + f.split("/")[-2], r["goal"], img, plans, good))
    if a.twin_dir:
        frames = []
        for line in open(f"{a.twin_dir}/transitions.jsonl"):
            r = json.loads(line)
            if r["timestep"] == 0 and r["task"] in ("task2a", "task1a") and len(frames) < a.n:
                plans, good = candidates(r["lang_goal"])
                img = np.array(Image.open(f"{a.twin_dir}/{r['image_t']}").convert("RGB"))
                frames.append((r["episode_id"], r["lang_goal"], img, plans, good))
    print(f"{len(frames)} initial frames")

    res = {}
    for wn, critn in [("calvin", "calvin"), ("twin", "twin"), ("twin", "calvin"), ("calvin", "twin")]:
        key = f"wm={wn}+critic={critn}"
        bp = BeamSearchPlanner(vlm_planner=None, world_model=wms[wn], critic=critics[critn], beam_width=1,
                               goal_threshold=0.05, temporal_threshold=-1e9, max_retries=1, max_replans=0,
                               wm_mode="v2a_wm")
        per = []
        for name, goal, img, plans, good in frames:
            gs, tcs = [], []
            for p in plans:
                score, _, all_scores, *_ = bp._evaluate_trajectory(plan=p, current_image_np=img, language_goal=goal,
                                                                  decoder=None)
                gs.append(float(score) if score is not None else float("nan"))
                tcs += [float(s[0]) for s in all_scores]
            gs, good_a = np.array(gs), np.array(good)
            pos, neg = gs[good_a], gs[~good_a]
            pair = float(np.mean([[p > n for n in neg] for p in pos]))
            per.append({"frame": name, "pairwise": pair, "margin": float(pos.mean() - neg.mean()),
                        "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()), "tc_mean": float(np.mean(tcs)),
                        "tc_pass_0.5": float(np.mean(np.array(tcs) >= 0.5))})
            print(f"{key:28s} {name:50s} pair={pair:.2f} margin={per[-1]['margin']:+.3f} pos={pos.mean():+.3f} neg={neg.mean():+.3f} tc={per[-1]['tc_mean']:.2f}")
        res[key] = {"per_frame": per, "pairwise": float(np.mean([x["pairwise"] for x in per])),
                    "margin": float(np.mean([x["margin"] for x in per])),
                    "tc_mean": float(np.mean([x["tc_mean"] for x in per])),
                    "tc_pass_0.5": float(np.mean([x["tc_pass_0.5"] for x in per]))}
        print(f"== {key}: pairwise={res[key]['pairwise']:.3f} margin={res[key]['margin']:+.4f} tc={res[key]['tc_mean']:.2f} tc>=0.5:{res[key]['tc_pass_0.5']:.2f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
