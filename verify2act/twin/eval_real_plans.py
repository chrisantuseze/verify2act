"""Real-frame plan ranking for task2a ("Put the red block to the left of the blue block").

Frames where the goal is not met (hand-checked on real_sheet.png): originals 1, 2, 3, 9, and the horizontal mirrors of
5, 6, 7, 8, 10 (red left of blue there, so the mirror has red right of blue). Candidates: all 40 single subtasks over
the four blocks; correct = red to the left of blue / blue to the right of red. Reports top1, the rank of the best
correct plan (1 = first) and pairwise.

python -m verify2act.twin.eval_real_plans --configs old+pooled,wm2+goalhead
"""
import argparse, glob, itertools, json

import numpy as np, torch
from PIL import Image, ImageOps
from types import SimpleNamespace

from verify2act.pipeline.inference import _build_critic
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.twin.eval_plans import CALVIN, TWIN, load_wm

COLORS = ["red", "green", "blue", "yellow"]
GOAL = "Put the red block to the left of the blue block"
GOOD = {"pick and place red block to the left of blue block", "pick and place blue block to the right of red block"}


def frames():
    fs = sorted(glob.glob("verify2act/output/real/*/imagination_logs/planning_call_*/request_image.png"))
    out = []
    for i in (1, 2, 3, 9):
        out.append((f"{i}", np.array(Image.open(fs[i]).convert("RGB"))))
    for i in (5, 6, 7, 8, 10):
        out.append((f"{i}-mirror", np.array(ImageOps.mirror(Image.open(fs[i]).convert("RGB")))))
    return out


def plans():
    ps = [f"pick and place {c} block into the bin" for c in COLORS]
    for c, b in itertools.permutations(COLORS, 2):
        ps += [f"pick and place {c} block {r} {b} block" for r in ("on", "to the left of", "to the right of")]
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="old+pooled,wm2+goalhead")
    ap.add_argument("--wm2", default="verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--wm2-encoder", default="verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v1/goal_head_last.pt")
    ap.add_argument("--critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_real_plans.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    fr, ps = frames(), plans()
    good = np.array([p in GOOD for p in ps])
    res = {}
    for cfg in a.configs.split(","):
        wn, cn = cfg.split("+")
        wm = (load_wm(a.wm2, a.wm2_encoder) if wn == "wm2" else
              load_wm(f"{TWIN}/wm/ckpt/latent_dynamics_best_weights.pt") if wn == "old" else
              load_wm(f"{CALVIN}/wm/ckpt/latent_dynamics_best_weights.pt"))
        crit = _build_critic(SimpleNamespace(critic_ckpt=a.critic, goal_head_ckpt=a.goal_head if cn == "goalhead" else None), dev)
        bp = BeamSearchPlanner(vlm_planner=None, world_model=wm, critic=crit, beam_width=1, goal_threshold=0.5,
                               temporal_threshold=-1e9, max_retries=1, max_replans=0, wm_mode="v2a_wm")
        per = []
        for name, img in fr:
            sc = []
            for p in ps:
                s, *_ = bp._evaluate_trajectory(plan=[p], current_image_np=img, language_goal=GOAL, decoder=None)
                sc.append(float(s) if s is not None and np.isfinite(s) else -1e9)
            sc = np.array(sc); order = np.argsort(-sc)
            rank = int(np.where(good[order])[0][0]) + 1
            pair = float(np.mean([[x > y for y in sc[~good]] for x in sc[good]]))
            per.append({"frame": name, "top1": ps[order[0]], "top1_ok": bool(good[order[0]]), "rank_correct": rank,
                        "pairwise": pair, "p_correct": float(sc[good].max()), "p_best_wrong": float(sc[~good].max())})
            print(f"{cfg:14s} {name:10s} rank={rank:2d} pair={pair:.2f} top1={ps[order[0]][15:]!r} "
                  f"P(correct)={sc[good].max():.2f} P(best wrong)={sc[~good].max():.2f}", flush=True)
        res[cfg] = {"top1": float(np.mean([x["top1_ok"] for x in per])), "mean_rank": float(np.mean([x["rank_correct"] for x in per])),
                    "pairwise": float(np.mean([x["pairwise"] for x in per])), "per_frame": per}
        print(f"== {cfg}: top1={res[cfg]['top1']:.2f} mean rank of correct={res[cfg]['mean_rank']:.1f}/40 "
              f"pairwise={res[cfg]['pairwise']:.3f}", flush=True)
        del wm, crit, bp
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
