"""Real-frame plan ranking with simulated labels, and a real-to-sim bridge.

The block poses and camera of every real request frame (and its mirror image) are fitted with fit_camera.py
(``camera_fit/fits.json``, ``camera_fit_mirror/fits.json``). For each scene and each single-step eval goal not yet met
(task2a, task2b, task3a), every valid subtask is executed in the twin from the fitted state and labelled with the goal
predicate. Each config imagines every plan and scores it; top1 / pairwise as in eval_plans.py.

Input modes:  real = the real image;  sim = a twin render of the fitted scene with the fitted camera (real-to-sim).

python -m verify2act.twin.eval_real2sim --configs real:wm2+goalhead,sim:wm2+goalhead,real:old+pooled
"""
import argparse, json, os
from types import SimpleNamespace

import numpy as np, torch
from PIL import Image

from verify2act.pipeline.inference import _build_critic
from verify2act.pipeline.planner import BeamSearchPlanner
from verify2act.twin.eval_plans import TWIN, load_wm
from verify2act.twin.fit_camera import cam_from_params
from verify2act.twin.goal_labels import EVAL_GOALS
from verify2act.twin.scene import SHEET_THICKNESS, DofbotTwin, InvalidSubtask

GOALS = {t: g for t, g in zip(["task1a", "task1b", "task1c", "task1d", "task2a", "task2b", "task3a"], EVAL_GOALS)}


def scenes(tw):
    out = []
    for fit_path in ("verify2act/output/twin/camera_fit/fits.json", "verify2act/output/twin/camera_fit_mirror/fits.json"):
        for i, fr in enumerate(json.load(open(fit_path))["frames"]):
            if fr["rms"] > 1.5:
                continue
            state = {c: {"present": c in fr["blocks"], "pos": [fr["blocks"][c][0], fr["blocks"][c][1],
                                                                 SHEET_THICKNESS + tw.size[2] / 2] if c in fr["blocks"] else [0, 0, 0],
                         "yaw": fr["blocks"][c][2] if c in fr["blocks"] else 0.0} for c in tw.colors}
            cam = cam_from_params(np.array(fr["params"]))
            cam = {"pos": np.asarray(cam["pos"]), "lookat": np.asarray(cam["lookat"]), "fovy": cam["fovy"], "roll": cam["roll"]}
            name = ("m" if "mirror" in fit_path else "") + f"{i:02d}"
            out.append((name, fr["frame"], state, cam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="real:wm2+goalhead,sim:wm2+goalhead,real:old+pooled")
    ap.add_argument("--tasks", default="task2a,task2b,task3a")
    ap.add_argument("--wm2", default="verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--wm2-encoder", default="verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v1/goal_head_last.pt")
    ap.add_argument("--critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    ap.add_argument("--samples", type=int, default=1, help="WM samples per plan (planner max_retries; score = max)")
    ap.add_argument("--save-renders", default="verify2act/output/real_eval/real2sim_renders")
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_real2sim.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tw = DofbotTwin(); tw.cfg["placement"]["p_fail"] = 0.0
    os.makedirs(a.save_renders, exist_ok=True)
    cases = []
    for name, frame, state, cam in scenes(tw):
        tw.set_state(state)
        render = tw.render(cam=cam)
        Image.fromarray(render).save(f"{a.save_renders}/{name}.png")
        real = np.array(Image.open(frame).convert("RGB"))
        subs = tw.valid_subtasks()
        for t in a.tasks.split(","):
            text, pred = GOALS[t]
            tw.set_state(state)
            if pred(tw.get_state()):
                continue
            plans, labels = [], []
            for s in subs:
                tw.set_state(state)
                try:
                    tw.apply(s, np.random.default_rng(0))
                except (InvalidSubtask, RuntimeError):
                    continue
                plans.append(s); labels.append(bool(pred(tw.get_state())))
            if any(labels):
                cases.append((name, t, text, {"real": real, "sim": render}, plans, np.array(labels)))
    print(f"{len(cases)} cases from real scenes; {np.mean([len(c[4]) for c in cases]):.1f} plans each, "
          f"{np.mean([c[5].mean() for c in cases]):.3f} correct frac", flush=True)
    from verify2act.critic.goal_head import GoalScorer
    crit = _build_critic(SimpleNamespace(critic_ckpt=a.critic), dev)
    scorer = GoalScorer(a.goal_head, dev) if "goalhead" in a.configs else None
    res, wms = {}, {}
    for cfg in a.configs.split(","):
        mode, rest = cfg.split(":"); wn, cn = rest.split("+")
        if wn not in wms:
            wms[wn] = load_wm(a.wm2, a.wm2_encoder) if wn == "wm2" else load_wm(f"{TWIN}/wm_old/ckpt/latent_dynamics_best_weights.pt")
        crit.goal_scorer = scorer if cn == "goalhead" else None
        bp = BeamSearchPlanner(vlm_planner=None, world_model=wms[wn], critic=crit, beam_width=1, goal_threshold=0.5,
                               temporal_threshold=-1e9, max_retries=a.samples, max_replans=0, wm_mode="v2a_wm")
        per = []
        for name, t, text, imgs, plans, lab in cases:
            sc = []
            for p in plans:
                s, *_ = bp._evaluate_trajectory(plan=[p], current_image_np=imgs[mode], language_goal=text, decoder=None)
                sc.append(float(s) if s is not None and np.isfinite(s) else -1e9)
            sc = np.array(sc)
            per.append({"scene": name, "task": t, "top1": float(lab[int(np.argmax(sc))]),
                        "pairwise": float(np.mean([[x > y for y in sc[~lab]] for x in sc[lab]])),
                        "top1_plan": plans[int(np.argmax(sc))], "correct": [p for p, l in zip(plans, lab) if l]})
        by = {}
        for x in per:
            by.setdefault(x["task"], []).append(x)
        res[cfg] = {"top1": float(np.mean([x["top1"] for x in per])), "pairwise": float(np.mean([x["pairwise"] for x in per])),
                    "by_task": {t: {"n": len(v), "top1": float(np.mean([x["top1"] for x in v])),
                                    "pairwise": float(np.mean([x["pairwise"] for x in v]))} for t, v in sorted(by.items())},
                    "per_case": per}
        print(f"== {cfg}: top1={res[cfg]['top1']:.3f} pairwise={res[cfg]['pairwise']:.3f}  " +
              "  ".join(f"{t}: n={v['n']} top1={v['top1']:.2f} pw={v['pairwise']:.2f}" for t, v in res[cfg]["by_task"].items()),
              flush=True)
        del bp
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
