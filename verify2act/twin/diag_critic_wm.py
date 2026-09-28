"""Separate critic quality from WM quality on twin task2a episodes (1 subtask each).

  gt     : goal-head AUROC, ground-truth final frame (achieved) vs start frame (not achieved)       -> critic alone
  imag   : goal-head AUROC, WM-imagined final frame (correct action) vs start frame                 -> critic + WM
  act    : P(goal score of imagined frame with the correct action > with a wrong action)            -> WM uses the action?
  sim_ok / sim_bad : cosine(critic embedding of the imagined frame, of the ground-truth final frame),
                     correct vs wrong action                                                        -> WM accuracy
"""
import argparse, json, random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from verify2act.pipeline.inference import _build_critic, preprocess_image_for_critic
from verify2act.twin.eval_real import CALVIN, TWIN, load_wm, single, COLORS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin-dir", default="verify2act/data/twin/dofbot_v1")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="verify2act/output/real_eval/diag_critic_wm.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    enc = f"{CALVIN}/encoder/ckpt/delta_encoder_best.pt"
    pairs = {"calvin": (f"{CALVIN}/wm/ckpt/latent_dynamics_best_weights.pt",
                        "verify2act/output/contrastive/calvin/best_contrastive_critic.pt"),
             "twin": (f"{TWIN}/wm_old/ckpt/latent_dynamics_best_weights.pt",
                      "verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt")}
    rows = []
    for line in open(f"{a.twin_dir}/transitions.jsonl"):
        r = json.loads(line)
        if r["task"] == "task2a" and r["timestep"] == 0 and r["num_steps"] == 1:
            rows.append(r)
        if len(rows) >= a.n * 3:
            break
    random.Random(0).shuffle(rows)
    rows = rows[-a.n:]                      # later episodes: not all from the start of the file
    goal = rows[0]["lang_goal"]
    rng = random.Random(1)
    wrong_pool = [single(c, "bin") for c in COLORS] + [single(c, r, b) for c in COLORS for b in COLORS if c != b
                                                       for r in ("on", "to the left of", "to the right of")]
    wrong_pool = [w for w in wrong_pool if w != "pick and place red block to the left of blue block"]
    res = {}
    load = lambda p: np.array(Image.open(f"{a.twin_dir}/{p}").convert("RGB"))
    for name, (wmp, cp) in pairs.items():
        wm = load_wm(wmp, enc, "cuda")
        critic = _build_critic(SimpleNamespace(critic_ckpt=cp), dev)
        emb = lambda img: critic.encode(preprocess_image_for_critic(img).to(dev))
        goal_s = lambda e: critic.goal_sim_from_text_with_uncertainty(e, goal)[0].item()

        def imagine(img, act):
            wm.initialize_history(img)
            h = wm.get_history().clone()
            F, _ = wm.imagine(None, act)
            wm.set_history(h)
            return critic.encode_features(F)

        s_t0, s_gt, s_im, act_win, sim_ok, sim_bad = [], [], [], [], [], []
        for r in rows:
            img0, img1 = load(r["image_t"]), load(r["image_t1"])
            e0, e1 = emb(img0), emb(img1)
            good = imagine(img0, r["action_text"])
            s_t0.append(goal_s(e0)); s_gt.append(goal_s(e1)); s_im.append(goal_s(good))
            sim_ok.append(torch.cosine_similarity(good.mu, e1.mu).item())
            for w in rng.sample(wrong_pool, 4):
                bad = imagine(img0, w)
                act_win.append(s_im[-1] > goal_s(bad))
                sim_bad.append(torch.cosine_similarity(bad.mu, e1.mu).item())
        y = [0] * len(rows) + [1] * len(rows)
        res[name] = {"gt_auroc": roc_auc_score(y, s_t0 + s_gt), "imag_auroc": roc_auc_score(y, s_t0 + s_im),
                     "action_pairwise": float(np.mean(act_win)), "sim_correct": float(np.mean(sim_ok)),
                     "sim_wrong": float(np.mean(sim_bad)),
                     "goal_mean": {"start": float(np.mean(s_t0)), "gt_final": float(np.mean(s_gt)),
                                   "imagined": float(np.mean(s_im))}}
        print(name, json.dumps(res[name]), flush=True)
        del wm, critic
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
