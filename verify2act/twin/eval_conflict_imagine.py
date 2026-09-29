"""Imagination quality on fresh precondition-conflict scenes (eval_conflict.build_scenes): does the WM imagine the
failing naive subtask as failing, and how close is each imagined outcome to the true simulated one?

For each scene and plan (naive / clearing), per WM, over ``--samples`` WM samples:
  cos_changed  mean per-patch cosine(imagined, true outcome) over the patches that changed most between the start
               frame and the true outcome (top ``--changed-frac``); ``copy`` (imagined = start) is the no-change baseline
  cos_all      the same over all patches
  closer_own   P(imagined naive is closer to the true naive outcome than to the true clearing outcome), and vice versa,
               on the union of both plans' changed patches: whether the WM imagines the right outcome, not just a change
  P_imag       goal-head P(goal) of the imagined outcome (P_true: of the true outcome)
The contact sheet shows start | true naive | naive per WM | true clearing | clearing per WM (decoded, sample 0).

MUJOCO_GL=egl python -m verify2act.twin.eval_conflict_imagine --wms wm2=<weights.pt>,wm3=<weights.pt>
"""
import argparse, json, os

import numpy as np, torch
from PIL import Image, ImageDraw

from verify2act.critic.goal_head import GoalScorer
from verify2act.robot.backend import load_feature_decoder
from verify2act.twin.eval_conflict import ENCODER, build_scenes
from verify2act.twin.eval_plans import load_wm
from verify2act.twin.eval_real_pairs import cos, flat

T = "verify2act/output/v2a_wm/dofbot_twin"
KINDS = ("covered", "ref_covered", "occupied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--wms", required=True, help="name=weights.pt,... (twin AE encoder)")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v3/goal_head_last.pt")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--changed-frac", type=float, default=0.1)
    ap.add_argument("--sheet-rows", type=int, default=5, help="scenes per conflict type on the contact sheet")
    ap.add_argument("--out", default="verify2act/output/real_eval/eval_conflict_imagine.json")
    ap.add_argument("--sheet", default="verify2act/output/visualizations/dofbot_twin/conflict_imagine.png")
    a = ap.parse_args()
    dev = torch.device("cuda")
    scenes = build_scenes(a.n, a.seed)
    print(f"{len(scenes)} scenes: " + ", ".join(f"{k} {sum(s['conflict'] == k for s in scenes)}" for k in KINDS), flush=True)
    gs = GoalScorer(a.goal_head, dev)
    dec = load_feature_decoder(f"{T}/decoder", dev)
    to_img = lambda F: Image.fromarray((dec.decode(F.float()).clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
                                        * 127.5 + 127.5).astype(np.uint8))
    specs = [s.split("=", 1) for s in a.wms.split(",")]
    on_sheet = [i for k in KINDS for i in [j for j, s in enumerate(scenes) if s["conflict"] == k][:a.sheet_rows]]
    S = 192
    cols = 1 + 2 * (1 + len(specs))
    sheet = Image.new("RGB", (S * cols, (S + 30) * len(on_sheet)), "white")
    draw = ImageDraw.Draw(sheet)
    res = {}
    for w, (name, path) in enumerate(specs):
        wm = load_wm(path, a.encoder)
        torch.manual_seed(0)
        per = []
        with torch.no_grad():
            for i, sc in enumerate(scenes):
                F = {}
                for k in ("naive", "clearing"):
                    wm.initialize_history(sc[f"img_{k}"])
                    F[k] = wm.get_history()[:, -1].clone()
                wm.initialize_history(sc["img"])
                h0 = wm.get_state()
                F0 = h0[0][:, -1]
                f0, ft = flat(F0), {k: flat(v) for k, v in F.items()}
                n_top = max(1, int(a.changed_frac * len(f0)))
                idx = {k: torch.topk(1 - cos(f0, ft[k]), n_top).indices for k in ft}
                both = torch.unique(torch.cat([idx["naive"], idx["clearing"]]))
                row = {"conflict": sc["conflict"], "outcome": sc["outcome"], "goal": sc["goal"]}
                for k, other in (("naive", "clearing"), ("clearing", "naive")):
                    smp = []
                    for s in range(a.samples):
                        wm.set_state(h0)
                        for step in sc[k]:
                            Fk, _ = wm.imagine(None, step)
                        fk = flat(Fk)
                        smp.append({"cos_changed": cos(fk[idx[k]], ft[k][idx[k]]).mean().item(),
                                    "cos_all": cos(fk, ft[k]).mean().item(),
                                    "closer_own": float(cos(fk[both], ft[k][both]).mean() > cos(fk[both], ft[other][both]).mean()),
                                    "P_imag": gs(Fk, [sc["goal"]]).item()})
                        if s == 0 and i in on_sheet:
                            y = on_sheet.index(i) * (S + 30)
                            x = (1 + (0 if k == "naive" else 1 + len(specs)) + 1 + w) * S
                            sheet.paste(to_img(Fk).resize((S, S)), (x, y + 30))
                            draw.text((x + 2, y + 16), f"{name} {k} cos {smp[-1]['cos_changed']:.2f} P={smp[-1]['P_imag']:.2f}",
                                      fill="black")
                    for m in ("cos_changed", "cos_all", "closer_own", "P_imag"):
                        row[f"{k}_{m}"] = float(np.mean([s_[m] for s_ in smp]))
                    row[f"{k}_copy_cos_changed"] = cos(f0[idx[k]], ft[k][idx[k]]).mean().item()
                    row[f"{k}_P_true"] = gs(F[k], [sc["goal"]]).item()
                if w == 0 and i in on_sheet:
                    y = on_sheet.index(i) * (S + 30)
                    draw.text((2, y + 2), f"[{sc['conflict']}/{sc['outcome']}] {sc['goal']}  |  naive: {sc['naive'][0]}  |  "
                                          f"clearing: {' ; '.join(sc['clearing'])}"[:190], fill="black")
                    sheet.paste(Image.fromarray(sc["img"]).resize((S, S)), (0, y + 30))
                    draw.text((2, y + 16), "start", fill="black")
                    for k, x in (("naive", S), ("clearing", (2 + len(specs)) * S)):
                        sheet.paste(Image.fromarray(sc[f"img_{k}"]).resize((S, S)), (x, y + 30))
                        draw.text((x + 2, y + 16), f"true {k} P={row[f'{k}_P_true']:.2f}", fill="black")
                per.append(row)
        keys = [f"{k}_{m}" for k in ("naive", "clearing")
                for m in ("cos_changed", "copy_cos_changed", "cos_all", "closer_own", "P_imag", "P_true")]
        summ = lambda rows: {m: float(np.mean([r[m] for r in rows])) for m in keys} | {"n": len(rows)}
        res[name] = summ(per) | {"by_conflict": {c: summ([r for r in per if r["conflict"] == c]) for c in KINDS},
                                 "per_scene": per}
        for label, r in [("all", res[name])] + list(res[name]["by_conflict"].items()):
            print(f"== {name} {label:11s} n={r['n']:3d}  " + "  ".join(
                f"{k}: cos_chg {r[k + '_cos_changed']:.3f} (copy {r[k + '_copy_cos_changed']:.3f}) "
                f"closer_own {r[k + '_closer_own']:.2f} P imag/true {r[k + '_P_imag']:.2f}/{r[k + '_P_true']:.2f}"
                for k in ("naive", "clearing")), flush=True)
        del wm
        torch.cuda.empty_cache()
    for p in (a.out, a.sheet):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    sheet.save(a.sheet)
    print(a.sheet)


if __name__ == "__main__":
    main()
