"""Where does the WM lose the goal signal? On held-out twin single-step episodes (goal met only after the step), score
with the spatial goal head:
  start  : F_t                                  (goal not met)
  true   : F_t1                                 (critic ceiling)
  ae     : F_t + dec(enc(F_t1 - F_t))           (ceiling for any WM on this delta autoencoder)
  wm     : WM(F_t, correct action)
  wrong  : WM(F_t, a random other valid action)
AUROC(x vs start) and P(wm > wrong) per condition.

python -m verify2act.twin.diag_wm_chain --encoder <delta_encoder.pt> --wm <latent_dynamics.pt>
"""
import argparse, json, random

import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from verify2act.critic.goal_head import GoalScorer
from verify2act.latent_wm.delta_encoder import DeltaDecoder, DeltaEncoder
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.twin.eval_plans import D, build_frames

C = "verify2act/data/twin/dino_features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="verify2act/output/v2a_wm/calvin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--wm", default="verify2act/output/v2a_wm/dofbot_twin/wm/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v1/goal_head_best.pt")
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    dev = torch.device("cuda")
    gs = GoalScorer(a.goal_head, dev)
    ck = torch.load(a.encoder, map_location="cpu")
    enc = DeltaEncoder(dino_channels=1024, model_channels=512, token_dim=128, num_latent_tokens=32, num_blocks=4)
    dec = DeltaDecoder(token_dim=128, model_channels=512, dino_channels=1024, num_patches=256, num_blocks=4)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"]); enc, dec = enc.to(dev).eval(), dec.to(dev).eval()
    wm = LatentWorldModel(device="cuda", dynamics_weights_path=a.wm, encoder_ckpt=a.encoder, history_len=3,
                          token_dim=128, num_latent_tokens=32, action_conditioning="cross_attn")
    frames = build_frames(a.n)
    first = {}
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        if r["timestep"] == 0:
            first[r["episode_id"]] = r
    feat = lambda p: torch.load(f"{C}/{p.replace('/', '_')}.pt").float().to(dev)[None]
    rng = random.Random(0)
    S = {k: [] for k in ("start", "true", "ae", "wm", "wrong")}
    with torch.no_grad():
        for ep, task, goal, img, plans, labels in frames:
            r = first[ep]
            Ft, Ft1 = feat(r["image_t"]), feat(r["image_t1"])
            wrong = rng.choice([p[0] for p, l in zip(plans, labels) if not l])

            def imag(act):
                wm.initialize_history(img)
                F, _ = wm.imagine(None, act)
                return F
            x = {"start": Ft, "true": Ft1, "ae": Ft + dec(enc(Ft1 - Ft)), "wm": imag(r["action_text"]), "wrong": imag(wrong)}
            for k, v in x.items():
                S[k].append(gs(v, [goal]).item())
    st = S["start"]
    for k in ("true", "ae", "wm", "wrong"):
        print(f"{k:6s} mean p={np.mean(S[k]):.3f}  auroc vs start={roc_auc_score([0] * len(st) + [1] * len(st), st + S[k]):.3f}")
    print(f"start  mean p={np.mean(st):.3f}")
    print(f"P(wm > wrong) = {np.mean(np.array(S['wm']) > np.array(S['wrong'])):.3f}   (n={len(st)})")


if __name__ == "__main__":
    main()
