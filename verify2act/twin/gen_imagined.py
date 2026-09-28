"""Imagined-feature dataset for training the goal head on what it will score at planning time.

For random frames of *training* episodes (held-out episodes of eval_plans / train_goal_head excluded) and a few valid
subtasks per frame (the executed one when available + random alternatives), the twin simulates the subtask (no
failures) to get the true next state, and the WM imagines the next DINO features from the cached features of the
frame. Saves feats [N, 256, 1024] fp16 + next states (for state-based goal labels).

python -m verify2act.twin.gen_imagined --wm <dyn.pt> --encoder <ae.pt> --out verify2act/data/twin/imagined_wm2.pt
"""
import argparse, json, os, random

import numpy as np, torch

from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.twin.eval_plans import D, val_episodes
from verify2act.twin.scene import DofbotTwin, InvalidSubtask

C = "verify2act/data/twin/dino_features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--encoder", default="verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--actions-per-frame", type=int, default=3)
    ap.add_argument("--out", default="verify2act/data/twin/imagined_wm2.pt")
    a = ap.parse_args()
    wm = LatentWorldModel(device="cuda", dynamics_weights_path=a.wm, encoder_ckpt=a.encoder, history_len=3,
                          token_dim=128, num_latent_tokens=32, action_conditioning="cross_attn")
    val = val_episodes()
    eps = sorted(e for e in os.listdir(f"{D}/episodes") if e not in val)
    executed = {}
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        executed[(r["episode_id"], r["timestep"])] = r["action_text"]
    rng = random.Random(0); nrng = np.random.default_rng(0)
    tw = DofbotTwin(render=False); tw.cfg["placement"]["p_fail"] = 0.0
    feats, states, meta = [], [], []
    for e in rng.sample(eps, a.frames):
        st = json.load(open(f"{D}/episodes/{e}/states.json"))
        k = rng.randrange(len(st))
        tw.set_state(st[k])
        subs = tw.valid_subtasks()
        acts = rng.sample(subs, min(len(subs), a.actions_per_frame))
        ex = executed.get((e, k))
        if ex and ex not in acts:
            acts[0] = ex
        F0 = torch.load(f"{C}/episodes_{e}_frame_{k:05d}.jpg.pt").float().cuda()[None]
        for act in acts:
            tw.set_state(st[k])
            try:
                tw.apply(act, nrng)
            except (InvalidSubtask, RuntimeError):
                continue
            wm._history = F0.unsqueeze(1).repeat(1, 3, 1, 1)
            wm._history_mask = torch.zeros(1, 3, dtype=torch.bool, device=F0.device); wm._history_mask[:, -1] = True
            with torch.no_grad():
                F1, _ = wm.imagine(None, act)
            feats.append(F1[0].half().cpu()); states.append(tw.get_state()); meta.append((e, k, act))
        if len(meta) % 1000 < a.actions_per_frame:
            print(len(meta), flush=True)
    torch.save({"feats": torch.stack(feats), "states": states, "meta": meta}, a.out)
    print("saved", len(meta), a.out)


if __name__ == "__main__":
    main()
