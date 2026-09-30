"""Diffusion WM (InstructPix2Pix + LoRA) on the real scenes' twin re-renders (what it sees with --real2sim).

For each real2sim case (eval_real2sim per_case: render id, task, correct plans) the WM imagines the correct subtask and
one wrong valid-looking subtask (a correct plan of another case, with a different target) from the render. The imagined
images go through DINOv2 and the twin goal head: pairwise = P(P_goal(correct) > P_goal(wrong)), and P(goal) of each.
The goal head is not part of the diffusion variant (it has no critic, as in the sim); it is only a yardstick here.
Configs are adapter[+decoder] pairs, e.g. the CALVIN LoRA vs the twin fine-tune, CALVIN-tuned VAE decoder vs stock.

python -m verify2act.twin.eval_diffusion --configs twin=<unet_lora>,twin_stockvae=<unet_lora>:none
"""
import argparse, json, os, random

import numpy as np, torch
from PIL import Image, ImageDraw

from verify2act.critic.goal_head import GoalScorer
from verify2act.pipeline.world_model import DiffusionWorldModel
from verify2act.twin.eval_plans import load_wm
from verify2act.twin.eval_real2sim import GOALS

R = "verify2act/output/real_eval"
T = "verify2act/output/v2a_wm/dofbot_twin"
CALVIN_DEC = "verify2act/output/diffusion_wm/calvin/decoder/checkpoint-5000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="name=adapter_dir[:decoder_dir|:none],... (default decoder: CALVIN)")
    ap.add_argument("--cases", default=f"{R}/final5_eval_real2sim_wm3_ghv3_k1.json")
    ap.add_argument("--renders", default=f"{R}/final5_renders_wm3_k1")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v3/goal_head_last.pt")
    ap.add_argument("--limit", type=int, default=None, help="first N cases only (smoke test)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--image-guidance", type=float, default=2.8)
    ap.add_argument("--text-guidance", type=float, default=7.5)
    ap.add_argument("--out", default=f"{R}/eval_diffusion.json")
    ap.add_argument("--sheet", default="verify2act/output/visualizations/dofbot_twin/eval_diffusion.png")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cases = json.load(open(a.cases))["real:wm2+goalhead"]["per_case"]
    rng = random.Random(0)
    for c in cases:
        others = [p for o in cases for p in o["correct"] if p not in c["correct"] and o["scene"] != c["scene"]]
        c["wrong"] = rng.choice(others)
    cases = cases[:a.limit]
    gs = GoalScorer(a.goal_head, dev)
    feat = load_wm(f"{T}/wm3/ckpt/latent_dynamics_best_weights.pt", f"{T}/encoder/ckpt/delta_encoder_best.pt")

    def P(img, goal):
        feat.initialize_history(img)
        return gs(feat.get_history()[:, -1], [goal]).item()

    S = 160
    specs = [s.split("=", 1) for s in a.configs.split(",")]
    sheet_rows = cases[:12]
    sheet = Image.new("RGB", (S * (1 + 2 * len(specs)), (S + 28) * len(sheet_rows)), "white")
    draw = ImageDraw.Draw(sheet)
    res = {}
    for ci, (name, spec) in enumerate(specs):
        adapter, _, decoder = spec.partition(":")
        decoder = None if decoder == "none" else (decoder or CALVIN_DEC)
        wm = DiffusionWorldModel(adapter_dir=adapter, decoder_dir=decoder, vae_model="runwayml/stable-diffusion-v1-5",
                                 vae_subfolder="vae", device="cuda", torch_dtype=torch.float16,
                                 num_inference_steps=a.steps, image_guidance_scale=a.image_guidance,
                                 guidance_scale=a.text_guidance, seed=0)
        rows = []
        for i, c in enumerate(cases):
            img = np.array(Image.open(f"{a.renders}/{c['scene']}.png").convert("RGB"))
            goal = GOALS[c["task"]][0]
            ok, bad = wm.imagine(img, c["correct"][0]), wm.imagine(img, c["wrong"])
            row = {"scene": c["scene"], "task": c["task"], "P_correct": P(ok, goal), "P_wrong": P(bad, goal),
                   "P_start": P(img, goal)}
            rows.append(row)
            if c in sheet_rows:
                y = sheet_rows.index(c) * (S + 28)
                if ci == 0:
                    sheet.paste(Image.fromarray(img).resize((S, S)), (0, y + 28))
                    draw.text((2, y + 2), f"{c['scene']} {c['task']}: {c['correct'][0]}  | wrong: {c['wrong']}"[:150], fill="black")
                for j, (im, p) in enumerate(((ok, row["P_correct"]), (bad, row["P_wrong"]))):
                    x = (1 + 2 * ci + j) * S
                    sheet.paste(Image.fromarray(im).resize((S, S)), (x, y + 28))
                    draw.text((x + 2, y + 15), f"{name} {'ok' if j == 0 else 'wrong'} P={p:.2f}", fill="black")
        pc, pw = np.array([r["P_correct"] for r in rows]), np.array([r["P_wrong"] for r in rows])
        res[name] = {"pairwise": float(np.mean(pc > pw)), "P_correct": float(pc.mean()), "P_wrong": float(pw.mean()),
                     "acc@0.5_correct": float(np.mean(pc >= 0.5)), "n": len(rows), "per_case": rows}
        print(f"== {name}: pairwise {res[name]['pairwise']:.3f}  P correct/wrong {pc.mean():.2f}/{pw.mean():.2f}  "
              f"correct>=0.5 {res[name]['acc@0.5_correct']:.2f}  (n={len(rows)})", flush=True)
        del wm
        torch.cuda.empty_cache()
    for p in (a.out, a.sheet):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    sheet.save(a.sheet)
    print(a.sheet)


if __name__ == "__main__":
    main()
