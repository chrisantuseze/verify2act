"""Precondition-conflict evaluation on fresh twin scenes (tasks.conflict_episode, not in any training set).

Each scene has a goal whose direct subtask cannot work (the block is covered, the stacking reference is covered, or the
side spot is occupied). Two plans are imagined from the start image and scored with the language goal:
  naive     the direct one-step subtask (simulated with scene.apply(conflicts=True): the goal never holds)
  clearing  move the blocker first, then the subtask (tasks.run_clearing: the goal holds)
We report
  pairwise     = P(score(clearing) > score(naive))
  naive_rej    = P(naive score < thr)          (the planner rejects the failing plan)
  clearing_acc = P(clearing score >= thr)

MUJOCO_GL=egl python -m verify2act.twin.eval_conflict --wms wm2=<weights.pt>,wm3=<weights.pt>
"""
import argparse, json, os
from types import SimpleNamespace

import numpy as np, torch

from verify2act.pipeline.inference import _build_critic
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.twin import augment
from verify2act.twin.config import load_config
from verify2act.twin.eval_plans import load_wm
from verify2act.twin.scene import DofbotTwin, InvalidSubtask
from verify2act.twin.tasks import conflict_episode, run_clearing

CONFIG = "verify2act/configs/twin/dofbot_twin_conflict.yaml"
ENCODER = "verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt"


def build_scenes(n, seed):
    t = DofbotTwin(load_config(CONFIG))
    rng = np.random.default_rng([seed, 7919])
    kinds = ["covered", "ref_covered", "occupied"]
    scenes = []
    while len(scenes) < n:
        conflict = kinds[len(scenes) % len(kinds)]
        t.randomize_episode(rng)
        t.reset(rng, [str(c) for c in rng.permutation(t.colors)[:int(rng.integers(3, 5))]])
        setup = conflict_episode(t, rng, conflict)
        if setup is None:
            continue
        goal, naive = setup
        if len(goal.moves) != 1:        # else the one-step naive plan could not reach the goal even without the conflict
            continue
        s0 = t.get_state()
        photo = augment.sample_params(t.cfg, rng)
        img = augment.apply(t.render(rng), photo, rng)
        try:
            outcome = t.apply(naive, rng, conflicts=True)
            if not outcome or goal.check(t):
                continue
            img_naive = augment.apply(t.render(rng), photo, rng)         # the true (simulated) outcomes, for the
            t.set_state(s0)                                                # critic alone, without the WM
            clearing = run_clearing(t, goal, rng, lambda s: None)
            img_clearing = augment.apply(t.render(rng), photo, rng)
        except (InvalidSubtask, RuntimeError):
            continue
        if len(clearing) < 2:
            continue
        scenes.append({"conflict": conflict, "outcome": outcome, "goal": goal.text, "img": img,
                       "naive": [naive], "clearing": clearing, "img_naive": img_naive, "img_clearing": img_clearing})
    t.close()
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--wms", required=True, help="name=weights.pt,... (twin AE encoder)")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v2/goal_head_last.pt")
    ap.add_argument("--twin-critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    ap.add_argument("--samples", type=int, default=1, help="WM samples per plan; score = best sample")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_conflict.json")
    a = ap.parse_args()
    scenes = build_scenes(a.n, a.seed)
    print(f"{len(scenes)} scenes: " + ", ".join(f"{k} {sum(s['conflict'] == k for s in scenes)}"
                                                for k in ("covered", "ref_covered", "occupied")), flush=True)
    critic = _build_critic(SimpleNamespace(critic_ckpt=a.twin_critic, goal_head_ckpt=a.goal_head), torch.device("cuda"))
    # The critic on the true outcome frames: separates critic leniency from WM errors.
    res = {}
    wm0 = load_wm(a.wms.split(",")[0].split("=", 1)[1], a.encoder)
    true = []
    with torch.no_grad():
        for sc in scenes:
            p = {}
            for k in ("naive", "clearing"):
                wm0.initialize_history(sc[f"img_{k}"])
                p[k] = float(critic.goal_sim_from_text_with_uncertainty(critic.encode_features(wm0.get_history()[:, -1]),
                                                                        sc["goal"])[0])
            true.append({"conflict": sc["conflict"], "P_naive": p["naive"], "P_clearing": p["clearing"],
                         "naive_rej": float(p["naive"] < a.thr), "clearing_acc": float(p["clearing"] >= a.thr)})
    del wm0
    res["true_frames"] = {k: float(np.mean([r[k] for r in true])) for k in ("naive_rej", "clearing_acc")} | {
        "by_conflict": {c: {k: float(np.mean([r[k] for r in true if r["conflict"] == c])) for k in ("naive_rej", "clearing_acc")}
                        for c in ("covered", "ref_covered", "occupied")}, "per_scene": true}
    print("== true frames (no WM): naive_rej={naive_rej:.3f} clearing_acc={clearing_acc:.3f}".format(**res["true_frames"])
          + "  " + "  ".join(f"{c}: rej {v['naive_rej']:.2f} acc {v['clearing_acc']:.2f}"
                             for c, v in res["true_frames"]["by_conflict"].items()), flush=True)
    for spec in a.wms.split(","):
        name, path = spec.split("=", 1)
        wm = load_wm(path, a.encoder)
        bp = BeamSearchPlanner(vlm_planner=None, world_model=wm, critic=critic, beam_width=1, goal_threshold=a.thr,
                               temporal_threshold=-1e9, max_retries=1, max_replans=0, wm_mode="v2a_wm")

        def score(plan, img, goal):
            best = -1e9
            for _ in range(a.samples):
                s, *_ = bp._evaluate_trajectory(plan=plan, current_image_np=img, language_goal=goal, decoder=None)
                if s is not None and np.isfinite(s):
                    best = max(best, float(s))
            return best

        per = []
        for sc in scenes:
            sn, scl = score(sc["naive"], sc["img"], sc["goal"]), score(sc["clearing"], sc["img"], sc["goal"])
            per.append({"conflict": sc["conflict"], "outcome": sc["outcome"], "goal": sc["goal"], "naive": sc["naive"],
                        "clearing": sc["clearing"], "score_naive": sn, "score_clearing": scl,
                        "pairwise": float(scl > sn), "naive_rej": float(sn < a.thr), "clearing_acc": float(scl >= a.thr)})

        def summary(rows):
            return {k: float(np.mean([r[k] for r in rows])) for k in ("pairwise", "naive_rej", "clearing_acc")} | \
                   {"n": len(rows)}

        res[name] = summary(per) | {"by_conflict": {k: summary([r for r in per if r["conflict"] == k])
                                                    for k in ("covered", "ref_covered", "occupied")}, "per_scene": per}
        r = res[name]
        print(f"== {name}: pairwise={r['pairwise']:.3f} naive_rej={r['naive_rej']:.3f} "
              f"clearing_acc={r['clearing_acc']:.3f}", flush=True)
        for k, v in r["by_conflict"].items():
            print(f"     {k:12s} n={v['n']:3d} pairwise={v['pairwise']:.3f} naive_rej={v['naive_rej']:.3f} "
                  f"clearing_acc={v['clearing_acc']:.3f}", flush=True)
        del wm, bp
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
