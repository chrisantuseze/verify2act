"""Nonlinear left/right probe on cached DINOv2 patch tokens: pair-conditioned small transformer over the 16x16 grid
vs an MLP on the mean-pooled tokens. Label: block c is left (+y) of block b, for a random ordered pair of table blocks."""
import json, os, random
import numpy as np, torch, torch.nn as nn

D = "verify2act/output/twin/dofbot_v1"; C = "verify2act/output/twin/dino_features"
COL = ["red", "green", "blue", "yellow"]

def samples(n, seed=0):
    eps = sorted(os.listdir(f"{D}/episodes")); random.Random(seed).shuffle(eps); out = []
    for e in eps:
        st = json.load(open(f"{D}/episodes/{e}/states.json"))
        for k in random.Random(e).sample(range(len(st)), min(2, len(st))):
            s = st[k]; tab = [c for c in COL if s[c]["present"] and s[c]["pos"][2] < 0.025]
            prs = [(a, b) for a in tab for b in tab if a != b and abs(s[a]["pos"][1] - s[b]["pos"][1]) > 0.02]
            if prs:
                a, b = random.Random(e + str(k)).choice(prs)
                out.append((f"{C}/episodes_{e}_frame_{k:05d}.jpg.pt", COL.index(a), COL.index(b), int(s[a]["pos"][1] > s[b]["pos"][1])))
        if len(out) >= n: break
    return out

class Spatial(nn.Module):
    def __init__(s, d=192, pos=True):
        super().__init__(); s.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        s.pos = nn.Parameter(torch.zeros(1, 256, d)) if pos else None; s.q = nn.Embedding(8, d)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        s.out = nn.Linear(d, 1)
    def forward(s, x, a, b):
        h = s.inp(x) + (s.pos if s.pos is not None else 0)
        q = (s.q(a) + s.q(b + 4)).unsqueeze(1)
        return s.out(s.tf(torch.cat([q, h], 1))[:, 0]).squeeze(-1)

class Pooled(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.q = nn.Embedding(8, 1024)
        s.m = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(s, x, a, b): return s.m(x.mean(1) + s.q(a) + s.q(b + 4)).squeeze(-1)

def run(model, X, A, B, Y, tr, te, dev, epochs=15):
    model = model.to(dev); opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            loss = nn.functional.binary_cross_entropy_with_logits(model(X[j].to(dev).float(), A[j].to(dev), B[j].to(dev)), Y[j].to(dev).float())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); acc = []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            j = te[i:i + 256]; acc.append(((model(X[j].to(dev).float(), A[j].to(dev), B[j].to(dev)) > 0).long().cpu() == Y[j]))
    return torch.cat(acc).float().mean().item()

if __name__ == "__main__":
    torch.manual_seed(0); fr = samples(12000)
    X = torch.stack([torch.load(p) for p, *_ in fr]); A, B, Y = (torch.tensor([f[i] for f in fr]) for i in (1, 2, 3))
    n = len(Y); tr = torch.arange(int(0.85 * n)); te = torch.arange(int(0.85 * n), n); dev = torch.device("cuda:0")
    print("N", n, "pos", Y.float().mean().item(), flush=True)
    print("pooled MLP      acc %.3f" % run(Pooled(), X, A, B, Y, tr, te, dev), flush=True)
    print("spatial no-pos  acc %.3f" % run(Spatial(pos=False), X, A, B, Y, tr, te, dev), flush=True)
    print("spatial +pos    acc %.3f" % run(Spatial(pos=True), X, A, B, Y, tr, te, dev), flush=True)
