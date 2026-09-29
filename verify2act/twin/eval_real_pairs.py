"""Imagination quality on real frames without hand labels: consecutive planning calls of a real run give a real
(before, executed subtasks, after) triple. The WM imagines the executed subtasks from the before frame; we compare its
DINO features with those of the real after frame.

  cos_changed  mean per-patch cosine(imagined, real after) over the patches that changed most between before and after
               (top ``--changed-frac``); the "copy" baseline (imagined = before) shows how much of the change is captured
  cos_all      the same over all patches
  P_real / P_imag  goal-head P(goal) on the real after frame / on the imagined one

Real runs whose robot misexecuted the subtask are included as they are: the WM predicts the intended effect, so the
contact sheet (before | after | decoded after | decoded imagination samples) shows which pairs are fair comparisons.

python -m verify2act.twin.eval_real_pairs --wms wm2=<weights.pt>,wm3=<weights.pt>
"""
import argparse, glob, json, os

import numpy as np, torch
from PIL import Image, ImageDraw

from verify2act.critic.goal_head import GoalScorer
from verify2act.robot.backend import load_feature_decoder
from verify2act.twin.eval_plans import load_wm

T = "verify2act/output/v2a_wm/dofbot_twin"


def real_pairs(patterns):
    """(name, goal, before image, executed subtasks, after image) from consecutive planning calls."""
    pairs = []
    for run in sorted({p for pat in patterns for p in glob.glob(pat)}):
        calls = sorted(glob.glob(f"{run}/imagination_logs/planning_call_*"))
        for a, b in zip(calls, calls[1:]):
            qa, qb = json.load(open(f"{a}/request.json")), json.load(open(f"{b}/request.json"))
            done = qb["history"][len(qa["history"]):]
            if not done or qb["history"][:len(qa["history"])] != qa["history"]:
                continue
            img = lambda c: np.array(Image.open(f"{c}/request_image.png").convert("RGB"))
            pairs.append((f"{os.path.basename(run)}:{os.path.basename(a)[-2:]}->{os.path.basename(b)[-2:]}",
                          qa["goal"], img(a), done, img(b)))
    return pairs


def flat(F):
    """(1, C, H, W) or (1, N, C) features -> (N, C)."""
    F = F.float()[0]
    return F.flatten(1).T if F.dim() == 3 and F.shape[0] > F.shape[-1] else F.reshape(-1, F.shape[-1])


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["verify2act/output/real/run_*", "run_2026*"])
    ap.add_argument("--wms", required=True, help="name=weights.pt,... (twin AE encoder)")
    ap.add_argument("--encoder", default=f"{T}/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v2/goal_head_last.pt")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--changed-frac", type=float, default=0.1)
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_real_pairs.json")
    ap.add_argument("--sheet", default="verify2act/output/visualizations/dofbot_twin/real_pairs.png")
    a = ap.parse_args()
    dev = torch.device("cuda")
    pairs = real_pairs(a.runs)
    print(f"{len(pairs)} real pairs: " + ", ".join(f"{n} ({len(s)} steps)" for n, _, _, s, _ in pairs), flush=True)
    gs = GoalScorer(a.goal_head, dev)
    dec = load_feature_decoder(f"{T}/decoder", dev)
    to_img = lambda F: Image.fromarray((dec.decode(F.float()).clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
                                        * 127.5 + 127.5).astype(np.uint8))
    specs = [s.split("=", 1) for s in a.wms.split(",")]
    S = 192
    cols = 3 + a.samples
    sheet = Image.new("RGB", (S * cols, (S + 30) * len(pairs) * len(specs)), "white")
    draw = ImageDraw.Draw(sheet)
    res = {}
    for w, (name, path) in enumerate(specs):
        wm = load_wm(path, a.encoder)
        torch.manual_seed(0)
        per = []
        with torch.no_grad():
            for i, (pn, goal, before, steps, after) in enumerate(pairs):
                wm.initialize_history(after)
                F1 = wm.get_history()[:, -1].clone()
                wm.initialize_history(before)
                h0 = wm.get_state()
                F0 = h0[0][:, -1]
                f0, f1 = flat(F0), flat(F1)
                change = 1 - cos(f0, f1)
                idx = torch.topk(change, max(1, int(a.changed_frac * len(change)))).indices
                y = (w * len(pairs) + i) * (S + 30)
                for c, (im, lab) in enumerate([(Image.fromarray(before), "before"), (Image.fromarray(after), "real after"),
                                               (to_img(F1), f"decoded after P={gs(F1, [goal]).item():.2f}")]):
                    sheet.paste(im.resize((S, S)), (c * S, y + 30))
                    draw.text((c * S + 2, y + 16), lab, fill="black")
                draw.text((2, y + 2), f"{name} {pn}: {' ; '.join(steps)}"[:150], fill="black")
                samples = []
                for k in range(a.samples):
                    wm.set_state(h0)
                    for s in steps:
                        Fk, _ = wm.imagine(None, s)
                    fk = flat(Fk)
                    samples.append({"cos_changed": cos(fk[idx], f1[idx]).mean().item(), "cos_all": cos(fk, f1).mean().item(),
                                    "P_imag": gs(Fk, [goal]).item()})
                    sheet.paste(to_img(Fk).resize((S, S)), ((3 + k) * S, y + 30))
                    draw.text(((3 + k) * S + 2, y + 16), f"WM {k} cos {samples[-1]['cos_changed']:.2f} "
                              f"P={samples[-1]['P_imag']:.2f}", fill="black")
                row = {"pair": pn, "steps": steps, "P_real": gs(F1, [goal]).item(),
                       "copy_cos_changed": cos(f0[idx], f1[idx]).mean().item(), "copy_cos_all": cos(f0, f1).mean().item()}
                for k in ("cos_changed", "cos_all", "P_imag"):
                    row[k] = float(np.mean([s[k] for s in samples]))
                    row[k + "_best"] = float(np.max([s[k] for s in samples]))
                per.append(row)
                print(f"  {name} {pn}: cos_changed {row['cos_changed']:.3f} (best {row['cos_changed_best']:.3f}, "
                      f"copy {row['copy_cos_changed']:.3f})  cos_all {row['cos_all']:.3f} (copy {row['copy_cos_all']:.3f})"
                      f"  P real {row['P_real']:.2f} imag {row['P_imag']:.2f}", flush=True)
        keys = ("cos_changed", "cos_changed_best", "copy_cos_changed", "cos_all", "copy_cos_all", "P_real", "P_imag")
        res[name] = {k: float(np.mean([r[k] for r in per])) for k in keys} | {"per_pair": per}
        print(f"== {name}: " + " ".join(f"{k}={res[name][k]:.3f}" for k in keys), flush=True)
        del wm
        torch.cuda.empty_cache()
    for p in (a.out, a.sheet):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    sheet.save(a.sheet)
    print(a.sheet)


if __name__ == "__main__":
    main()
