"""Train SpatialGoalHead on twin frames relabelled with state-derived goal labels (verify2act/twin/goal_labels.py).

Features: the cached DINOv2 patch tokens (verify2act/data/twin/dino_features). Evaluation:
  - twin val (held-out episodes): AUROC over sampled goals, flipped-side / swapped-argument accuracy,
    and per eval task AUROC of episode start vs final frame (same protocol as diag_critic_tasks.py)
  - real: the robot request frames in output/real, labelled by hand for task2a, plus their horizontal mirrors.

python -m verify2act.twin.train_goal_head --out verify2act/output/goal_head/v1
"""
import argparse, collections, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from verify2act.critic.goal_head import ClipTokenEncoder, SpatialGoalHead
from verify2act.twin.goal_labels import EVAL_GOALS, sample_goals

D = "verify2act/data/twin/dofbot_v1"
C = "verify2act/data/twin/dino_features"
TASK_IDX = {"task1a": 0, "task1b": 1, "task1c": 2, "task1d": 3, "task2a": 4, "task2b": 5, "task3a": 6}
REAL_FEATS = "verify2act/output/goal_head/real_feats.pt"


def load_frames(val_frac=0.05, limit=None, extra=None):
    """extra: optional [M, 256, 1024] tensor appended after the frames (allocated once, no concat copy)."""
    eps = sorted(os.listdir(f"{D}/episodes"))[:limit]
    task = {}
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        if r["task"] and r["episode_success"]:
            task[r["episode_id"]] = r["task"]
    items = []
    for e in eps:
        st = json.load(open(f"{D}/episodes/{e}/states.json"))
        for k, s in enumerate(st):
            items.append((e, k, len(st), s))
    t0 = time.time()
    M = 0 if extra is None else len(extra)
    X = torch.empty(len(items) + M, 256, 1024, dtype=torch.float16)

    def load(j):
        e, k = items[j][:2]
        X[j] = torch.load(f"{C}/episodes_{e}_frame_{k:05d}.jpg.pt")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, range(len(items))))
    if M:
        X[len(items):] = extra
    print(f"loaded {len(items)} frames {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)
    # held-out episodes: drawn from the dofbot_v1 ids only (ep_0xxxxx), so the split is the same for dofbot_v1c;
    # supplementary episodes (ep_1xxxxx, dofbot_conflict) are all training
    rng = random.Random(0); ev = sorted(e for e in set(eps) if e < "ep_100000"); rng.shuffle(ev)
    val_eps = set(ev[:int(val_frac * len(ev))])
    is_val = torch.tensor([it[0] in val_eps for it in items])
    return X, items, is_val, task


def make_real_feats(device):
    """DINOv2 patch tokens of the real request frames and their mirrors (same transform as cache_utils)."""
    import glob
    from PIL import Image, ImageOps
    from torchvision import transforms
    fs = sorted(glob.glob("verify2act/output/real/*/imagination_logs/planning_call_*/request_image.png"))
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True).eval().to(device)
    xf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ims = [Image.open(f).convert("RGB") for f in fs]
    with torch.no_grad():
        x = torch.stack([xf(i) for i in ims] + [xf(ImageOps.mirror(i)) for i in ims]).to(device)
        feats = torch.cat([dino.forward_features(x[i:i + 8])["x_norm_patchtokens"] for i in range(0, len(x), 8)])
    del dino
    # hand labels for "red left of blue" on the originals (see real_sheet.png); None = ambiguous
    red_left = {5: 1, 6: 1, 7: 1, 8: 1, 10: 1, 0: 0, 2: 0, 3: 0, 9: 0}
    out = {"files": fs, "feats": feats.half().cpu(), "red_left": red_left}
    Path(REAL_FEATS).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, REAL_FEATS)
    return out


@torch.no_grad()
def score(head, enc, X, texts, device, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        txt, pad = enc(texts[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(head(X[i:i + bs].to(device), txt, pad).float().cpu())
    return torch.cat(out)


def eval_real(head, enc, real, device):
    f = real["feats"]; n = f.shape[0] // 2; lab = real["red_left"]
    L = "Put the red block to the left of the blue block"; R = "Put the red block to the right of the blue block"
    s_orig = torch.sigmoid(score(head, enc, f[:n], [L] * n, device))
    s_flip = torch.sigmoid(score(head, enc, f[n:], [L] * n, device))
    r_flip = torch.sigmoid(score(head, enc, f[n:], [R] * n, device))
    pos = [i for i, y in lab.items() if y == 1]; neg = [i for i, y in lab.items() if y == 0]
    y = [1] * len(pos) + [0] * (len(neg) + len(pos))
    sc = [s_orig[i].item() for i in pos] + [s_orig[i].item() for i in neg] + [s_flip[i].item() for i in pos]
    res = {"real_auroc_task2a": roc_auc_score(y, sc),
           "real_pair_orig_gt_flip": float(np.mean([s_orig[i] > s_flip[i] for i in pos])),
           "real_rightof_on_flip_pos": float(r_flip[pos].mean()),
           "real_p_pos": float(s_orig[pos].mean()), "real_p_neg": float(s_orig[neg].mean())}
    # every real frame has all four blocks on the table: bin / stack eval goals are all false there
    others = [t for t, _ in EVAL_GOALS if "left" not in t and "right" not in t]
    so = torch.sigmoid(score(head, enc, f[:n].repeat_interleave(len(others), 0), others * n, device))
    res["real_false_goal_mean_p"] = float(so.mean())
    return res


def evaluate(head, enc, X, items, is_val, task, device, ae=None):
    head.eval()
    vi = torch.nonzero(is_val).squeeze(1).tolist()
    rng = random.Random(123); texts, labels, idx, kinds = [], [], [], []
    for i in vi:
        s = items[i][3]
        for t, y in sample_goals(rng, s, 8):
            texts.append(t); labels.append(y); idx.append(i)
    sc = score(head, enc, X[idx], texts, device)
    res = {"val_auroc": roc_auc_score(labels, sc.numpy()),
           "val_acc": float(((sc > 0).long().numpy() == np.array(labels)).mean())}
    # per eval task: start vs final frame, eval-task text
    per = collections.defaultdict(lambda: ([], []))
    first = {}
    for i in vi:
        e, k, n, _ = items[i]
        if e in task:
            if k == 0: per[task[e]][0].append(i)
            if k == n - 1: per[task[e]][1].append(i)
    for t, (a, b) in sorted(per.items()):
        txt = EVAL_GOALS[TASK_IDX[t]][0]
        s = score(head, enc, X[a + b], [txt] * (len(a) + len(b)), device).numpy()
        res[f"auroc_{t}"] = roc_auc_score([0] * len(a) + [1] * len(b), s)
        if ae is not None:
            with torch.no_grad():
                prev = X[[i - 1 for i in b]].to(device).float()
                xb = prev + ae(X[b].to(device).float() - prev)
            s_ae = score(head, enc, torch.cat([X[a].to(device).float(), xb]), [txt] * (len(a) + len(b)), device).numpy()
            res[f"ae_auroc_{t}"] = roc_auc_score([0] * len(a) + [1] * len(b), s_ae)
    head.train()
    return res


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify2act/output/goal_head/v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=96, help="frames per step (x goals-per-frame pairs)")
    ap.add_argument("--goals-per-frame", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--feat-noise", type=float, default=0.0, help="gaussian noise std on patch tokens (robustness)")
    ap.add_argument("--token-drop", type=float, default=0.0, help="fraction of patch tokens zeroed per sample")
    ap.add_argument("--ae-ckpt", default=None, help="delta autoencoder: replace --ae-frac of frames (k >= 1) by "
                    "F_{k-1} + dec(enc(F_k - F_{k-1})), i.e. what a WM on this autoencoder can at best produce")
    ap.add_argument("--ae-frac", type=float, default=0.5)
    ap.add_argument("--init", default=None, help="goal head checkpoint to start from")
    ap.add_argument("--imagined", default=None, help="gen_imagined.py output: WM-imagined features + true next states, "
                    "added to the training pool (training episodes only)")
    ap.add_argument("--imagined-repeat", type=int, default=1, help="how often each imagined sample appears per epoch")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default=D, help="twin dataset dir (e.g. verify2act/data/twin/dofbot_v1c)")
    args = ap.parse_args()
    D = args.dataset
    torch.manual_seed(0); random.seed(0)
    dev = torch.device(args.device); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    real = torch.load(REAL_FEATS) if os.path.exists(REAL_FEATS) else make_real_feats(dev)
    im = torch.load(args.imagined, weights_only=False) if args.imagined else None
    X, items, is_val, task = load_frames(limit=args.limit_episodes, extra=None if im is None else im["feats"])
    if im is not None:
        del im["feats"]
        items = items + [(f"{e}#im", -1, 0, st) for (e, k, act), st in zip(im["meta"], im["states"])]
        is_val = torch.cat([is_val, torch.zeros(len(im["meta"]), dtype=torch.bool)])
        print(f"+{len(im['meta'])} imagined samples", flush=True)
    enc = ClipTokenEncoder(dev)
    head = SpatialGoalHead(d=args.d, layers=args.layers).to(dev)
    if args.init:
        head.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state_dict"])
    ae = None
    if args.ae_ckpt:
        from verify2act.latent_wm.delta_encoder import DeltaDecoder, DeltaEncoder
        ck = torch.load(args.ae_ckpt, map_location="cpu")
        ae_e = DeltaEncoder(dino_channels=1024, model_channels=512, token_dim=128, num_latent_tokens=32, num_blocks=4)
        ae_d = DeltaDecoder(token_dim=128, model_channels=512, dino_channels=1024, num_patches=256, num_blocks=4)
        ae_e.load_state_dict(ck["encoder"]); ae_d.load_state_dict(ck["decoder"])
        ae_e, ae_d = ae_e.to(dev).eval(), ae_d.to(dev).eval()
        ae = lambda d: ae_d(ae_e(d))
    opt = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.05)
    tr = torch.nonzero(~is_val).squeeze(1)
    if args.imagined and args.imagined_repeat > 1:
        extra = torch.arange(len(items) - len(im["meta"]), len(items))
        tr = torch.cat([tr] + [extra] * (args.imagined_repeat - 1))
    steps_per_epoch = len(tr) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    rng = random.Random(1); best = -1; hist = []
    print("real before training:", eval_real(head, enc, real, dev), flush=True)
    for ep in range(args.epochs):
        perm = tr[torch.randperm(len(tr))]; t0 = time.time(); tot = 0.0
        for st in range(steps_per_epoch):
            b = perm[st * args.batch:(st + 1) * args.batch].tolist()
            texts, labels, idx = [], [], []
            for i in b:
                for t, y in sample_goals(rng, items[i][3], args.goals_per_frame):
                    texts.append(t); labels.append(y); idx.append(i)
            x = X[idx].to(dev, non_blocking=True).float()
            if ae is not None:
                sel = [j for j, i in enumerate(idx) if items[i][1] >= 1 and rng.random() < args.ae_frac]
                if sel:
                    prev = X[[idx[j] - 1 for j in sel]].to(dev).float()
                    with torch.no_grad():
                        x[sel] = prev + ae(x[sel] - prev)
            if args.feat_noise > 0:
                x = x + args.feat_noise * torch.randn_like(x)
            if args.token_drop > 0:
                x = x * (torch.rand(x.shape[:2], device=dev) > args.token_drop).unsqueeze(-1)
            txt, pad = enc(texts)
            y = torch.tensor(labels, device=dev, dtype=torch.float32)
            pw = ((1 - y).sum() / y.sum().clamp(min=1)).clamp(max=5.0)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = head(x, txt, pad)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
            if st % 200 == 0:
                print(f"ep {ep} step {st}/{steps_per_epoch} loss {loss.item():.4f}", flush=True)
        res = {"epoch": ep, "train_loss": tot / steps_per_epoch, "time": time.time() - t0}
        res.update(evaluate(head, enc, X, items, is_val, task, dev, ae)); res.update(eval_real(head, enc, real, dev))
        res = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}
        hist.append(res); print(json.dumps(res), flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        ck = {"config": head.config, "state_dict": head.state_dict(), "epoch": ep, "metrics": res}
        torch.save(ck, out / "goal_head_last.pt")
        if res["val_auroc"] > best:
            best = res["val_auroc"]; torch.save(ck, out / "goal_head_best.pt")


if __name__ == "__main__":
    main()
