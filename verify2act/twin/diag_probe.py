"""Step-1 diagnostics. (A) does CLIP / the critic's text projection separate swapped left/right prompts?
(B) can left/right be linearly read from mean-pooled DINOv2 patch tokens vs the spatial patch grid?"""
import json, os, random, sys
from types import SimpleNamespace
import numpy as np, torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

D = "verify2act/data/twin/dofbot_v1"
C = "verify2act/data/twin/dino_features"
CK = "verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt"

def clip_check():
    from verify2act.pipeline.inference import _build_critic
    critic = _build_critic(SimpleNamespace(critic_ckpt=CK), torch.device("cpu"))
    pairs = [("Put the red block to the left of the blue block", "Put the blue block to the left of the red block"),
             ("Put the red block to the left of the blue block", "Put the red block to the right of the blue block"),
             ("Put the green block to the right of the yellow block", "Put the green block to the left of the yellow block"),
             ("Put the red block to the left of the blue block", "Stack the blue block on top of the yellow block"),
             ("Put the blue block and the yellow block into the bin", "Put the red block and the green block into the bin")]
    critic._load_clip(torch.device("cpu"))
    for a, b in pairs:
        tok = critic._clip_tokenizer([a, b], truncate=True)
        with torch.no_grad():
            e = critic._clip_model.encode_text(tok).float()
            p = F.normalize(critic.clip_goal_proj(e), dim=-1)
        print("clip=%.4f proj=%.4f | %s || %s" % (F.cosine_similarity(e[0], e[1], dim=0), (p[0] * p[1]).sum(), a, b))

def frames(n, seed=0):
    eps = sorted(os.listdir(f"{D}/episodes"))
    random.Random(seed).shuffle(eps)
    out = []
    for e in eps:
        st = json.load(open(f"{D}/episodes/{e}/states.json"))
        k = random.Random(e).randrange(len(st))
        s = st[k]
        for c, b in (("red", "blue"), ("green", "yellow")):
            if s[c]["present"] and s[b]["present"] and abs(s[c]["pos"][2] - s[b]["pos"][2]) < 0.01:
                dy = s[c]["pos"][1] - s[b]["pos"][1]
                if abs(dy) > 0.02:
                    out.append((f"{C}/episodes_{e}_frame_{k:05d}.jpg.pt", (c, b), int(dy > 0)))
                    break
        if len(out) >= n:
            break
    return out

def probe(n=3000):
    fr = frames(n)
    X = np.stack([torch.load(p).float().numpy() for p, _, _ in fr])     # [N,256,1024]
    y = np.array([l for _, _, l in fr]); pair = np.array([pr[0] == "red" for _, pr, _ in fr])
    print("N", len(y), "pos frac", y.mean())
    tr = np.arange(len(y)) < int(0.75 * len(y)); te = ~tr
    def fit(Z, name):
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Z[tr], y[tr])
        acc = (clf.predict(Z[te]) == y[te])
        print(f"{name}: test acc {acc.mean():.3f}  (red/blue {acc[pair[te]].mean():.3f}, green/yellow {acc[~pair[te]].mean():.3f})")
    fit(X.mean(1), "mean-pooled [1024]")
    pca = PCA(32).fit(X[tr].reshape(-1, 1024)[::7])
    Z = pca.transform(X.reshape(-1, 1024)).reshape(len(y), 256, 32)
    fit(Z.reshape(len(y), -1), "spatial grid PCA32 [256x32]")
    Zp = Z.reshape(len(y), 16, 16, 32).reshape(len(y), 4, 4, 4, 4, 32).mean((2, 4))
    fit(Zp.reshape(len(y), -1), "4x4 grid PCA32 [16x32]")

if __name__ == "__main__":
    if "clip" in sys.argv: clip_check()
    if "probe" in sys.argv: probe()
