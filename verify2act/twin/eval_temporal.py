"""Temporal-consistency gate (HEAD2, theta_c) of the twin critic, checked against the twin WMs.

The planner requeries a step when the critic's MC temporal similarity tc = cos(head2(e_t), head2(e_t+1)) is below
theta_c or its MC std is >= 0.08 (critic/inference.check_rollout_consistency). The gate should pass real transitions and
the WM's imagined ones, and reject implausible ones. On held-out twin single-step episodes (eval_plans.build_frames):
  true      F_t -> F_t1 (the recorded next frame)
  wm:<name>   F_t -> WM(F_t, correct action)       wrong:<name>  F_t -> WM(F_t, a wrong valid action)
  swap      F_t -> the start frame of another episode   (implausible: a different scene)
  shuffle   F_t -> F_t with its 256 patches permuted     (implausible: scrambled layout)
  noise     F_t -> F_t + N(0, (s * std F_t)^2)           (implausible: feature-level hallucination)
and on real frames (consecutive real planning calls, eval_real_pairs.real_pairs): real before -> real after, and
real before -> WM imagination of the executed subtasks. For each condition: mean tc / std, the pass rate at --theta-c,
and a theta sweep; AUROC(plausible = true + wm vs implausible = swap + shuffle + noise).

MUJOCO_GL=egl python -m verify2act.twin.eval_temporal --wms wm3=<weights.pt>[,rla=<weights.pt>:rla]
"""
import argparse, json, os, random
from types import SimpleNamespace

import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from verify2act.critic.inference import check_rollout_consistency
from verify2act.pipeline.inference import _build_critic
from verify2act.twin.eval_plans import D, build_frames, load_wm
from verify2act.twin.eval_real_pairs import real_pairs

T = "verify2act/output/v2a_wm/dofbot_twin"
THETAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load(spec, encoder):
    """name=path[:rla] -> (name, WM)."""
    name, path = spec.split("=", 1)
    return name, load_wm(path, encoder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wms", required=True, help="name=weights.pt[:rla],...")
    ap.add_argument("--encoder", default=f"{T}/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--critic", default="verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--samples", type=int, default=3, help="WM samples per transition")
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--theta-c", type=float, default=0.5)
    ap.add_argument("--runs", nargs="+", default=["verify2act/output/real/run_*", "run_2026*"])
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_temporal.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    critic = _build_critic(SimpleNamespace(critic_ckpt=a.critic, goal_head_ckpt=None), dev).eval()
    wms = [load(s, a.encoder) for s in a.wms.split(",")]
    feat_wm = wms[0][1]

    def F_of(img):
        feat_wm.initialize_history(img)
        return feat_wm.get_history()[:, -1].clone()

    def tc(F0, F1):
        m, s = critic.temporal_sim_with_uncertainty(critic.encode_features(F0), critic.encode_features(F1))
        return m.item(), s.item()

    def imagine(wm, img, steps):
        wm.initialize_history(img)
        for step in steps:
            F, _ = wm.imagine(None, step)
        return F

    first = {}
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        if r["timestep"] == 0:
            first[r["episode_id"]] = r
    frames = build_frames(a.n)
    print(f"{len(frames)} twin frames", flush=True)
    rng = random.Random(0)
    torch.manual_seed(0)
    S = {}
    add = lambda k, v: S.setdefault(k, []).append(v)
    load_img = lambda p: np.array(Image.open(f"{D}/{p}").convert("RGB"))
    eps = [f[0] for f in frames]
    with torch.no_grad():
        for ep, task, goal, img, plans, labels in frames:
            r = first[ep]
            F0 = F_of(img)
            add("true", tc(F0, F_of(load_img(r["image_t1"]))))
            other = rng.choice([e for e in eps if e != ep])
            add("swap", tc(F0, F_of(load_img(first[other]["image_t"]))))
            add("shuffle", tc(F0, F0[:, torch.randperm(F0.shape[1], device=F0.device)]))
            add("noise", tc(F0, F0 + a.noise * F0.std() * torch.randn_like(F0)))
            wrong = rng.choice([p[0] for p, l in zip(plans, labels) if not l])
            for name, wm in wms:
                for _ in range(a.samples):
                    add(f"wm:{name}", tc(F0, imagine(wm, img, [r["action_text"]])))
                    add(f"wrong:{name}", tc(F0, imagine(wm, img, [wrong])))
        pairs = real_pairs(a.runs)
        print(f"{len(pairs)} real pairs", flush=True)
        for pname, goal, before, done, after in pairs:
            F0 = F_of(before)
            add("real:true", tc(F0, F_of(after)))
            for name, wm in wms:
                for _ in range(a.samples):
                    add(f"real:wm:{name}", tc(F0, imagine(wm, before, done)))

    def passes(v, th):
        return float(np.mean([check_rollout_consistency(m, th, uncertainty=s).action == "continue" for m, s in v]))

    res = {"theta_c": a.theta_c, "conditions": {}}
    print(f"\n{'condition':18s} {'n':>4s} {'tc mean':>8s} {'tc p05':>7s} {'unc':>6s} {'pass@' + str(a.theta_c):>9s}   sweep "
          + " ".join(f"{t:4.1f}" for t in THETAS))
    for k, v in S.items():
        m = np.array([x[0] for x in v]); s = np.array([x[1] for x in v])
        row = {"n": len(v), "tc_mean": float(m.mean()), "tc_p05": float(np.percentile(m, 5)), "unc_mean": float(s.mean()),
               "pass": passes(v, a.theta_c), "sweep": {str(t): passes(v, t) for t in THETAS}}
        res["conditions"][k] = row
        print(f"{k:18s} {len(v):4d} {row['tc_mean']:8.3f} {row['tc_p05']:7.3f} {row['unc_mean']:6.3f} {row['pass']:9.2f}   "
              + " ".join(f"{row['sweep'][str(t)]:.2f}" for t in THETAS))
    bad = [x[0] for k in ("swap", "shuffle", "noise") for x in S[k]]
    for name, _ in wms:
        good = [x[0] for x in S["true"] + S[f"wm:{name}"]]
        auc = roc_auc_score([1] * len(good) + [0] * len(bad), good + bad)
        res[f"auroc_plausible_vs_implausible:{name}"] = float(auc)
        print(f"AUROC plausible (true + wm:{name}) vs implausible (swap/shuffle/noise): {auc:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res | {"raw": S}, open(a.out, "w"), indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
