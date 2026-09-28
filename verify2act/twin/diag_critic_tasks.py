"""Goal-head AUROC on ground-truth twin frames per eval task: episode start (not achieved) vs final frame (achieved)."""
import collections, json, random, sys
from types import SimpleNamespace
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from verify2act.pipeline.inference import _build_critic, preprocess_image_for_critic

D = "verify2act/data/twin/dofbot_v1"
ck = sys.argv[1] if len(sys.argv) > 1 else "verify2act/output/contrastive/dofbot_twin/best_contrastive_critic.pt"
dev = torch.device("cuda")
critic = _build_critic(SimpleNamespace(critic_ckpt=ck), dev)
by = collections.defaultdict(list)
last = {}
for line in open(f"{D}/transitions.jsonl"):
    r = json.loads(line)
    if r["task"]:
        by[r["episode_id"]].append(r)
eps = list(by.values())
random.Random(0).shuffle(eps)
per = collections.defaultdict(lambda: ([], []))
with torch.no_grad():
    for ep in eps:
        t = ep[0]["task"]
        if len(per[t][0]) >= 40 or not ep[-1]["episode_success"]:
            continue
        g = ep[0]["lang_goal"]
        sc = lambda p: critic.goal_sim_from_text_with_uncertainty(critic.encode(preprocess_image_for_critic(
            np.array(Image.open(f"{D}/{p}").convert("RGB"))).to(dev)), g)[0].item()
        per[t][0].append(sc(ep[0]["image_t"])); per[t][1].append(sc(ep[-1]["image_t1"]))
for t, (n, p) in sorted(per.items()):
    print(t, len(n), "auroc=%.3f" % roc_auc_score([0] * len(n) + [1] * len(p), n + p), "start=%.2f final=%.2f" % (np.mean(n), np.mean(p)))
