"""Contact sheet: input frame | decoded DINO features | decoded WM imagination for a few actions (+ goal-head P).

python -m verify2act.twin.viz_imagine --wm <latent_dynamics.pt> --encoder <delta_encoder.pt> --out sheet.png
Frames: the real request frames (output/real) and a few held-out twin task2a start frames.
"""
import argparse, glob, json

import numpy as np, torch
from PIL import Image, ImageDraw

from verify2act.critic.goal_head import GoalScorer
from verify2act.pipeline.world_model import LatentWorldModel
from verify2act.robot.backend import load_feature_decoder
from verify2act.twin.eval_plans import D, val_episodes

GOAL = "Put the red block to the left of the blue block"
ACTIONS = ["pick and place red block to the left of blue block", "pick and place red block to the right of blue block",
           "pick and place red block into the bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="verify2act/output/v2a_wm/dofbot_twin/wm2/ckpt/latent_dynamics_best_weights.pt")
    ap.add_argument("--encoder", default="verify2act/output/v2a_wm/dofbot_twin/encoder/ckpt/delta_encoder_best.pt")
    ap.add_argument("--decoder-dir", default="verify2act/output/v2a_wm/dofbot_twin/decoder")
    ap.add_argument("--goal-head", default="verify2act/output/goal_head/v1/goal_head_last.pt")
    ap.add_argument("--n-twin", type=int, default=4)
    ap.add_argument("--out", default="verify2act/output/visualizations/dofbot_twin/imagine_sheet.png")
    a = ap.parse_args()
    dev = torch.device("cuda")
    dec = load_feature_decoder(a.decoder_dir, dev)
    gs = GoalScorer(a.goal_head, dev)
    wm = LatentWorldModel(device="cuda", dynamics_weights_path=a.wm, encoder_ckpt=a.encoder, history_len=3,
                          token_dim=128, num_latent_tokens=32, action_conditioning="cross_attn")
    imgs = [(f.split("real/")[1][:22], np.array(Image.open(f).convert("RGB")))
            for f in sorted(glob.glob("verify2act/output/real/*/imagination_logs/planning_call_*/request_image.png"))]
    val = val_episodes()
    for line in open(f"{D}/transitions.jsonl"):
        r = json.loads(line)
        if r["task"] == "task2a" and r["timestep"] == 0 and r["episode_id"] in val and len(imgs) < 11 + a.n_twin:
            imgs.append(("twin " + r["episode_id"], np.array(Image.open(f"{D}/{r['image_t']}").convert("RGB"))))
    to_img = lambda t: Image.fromarray((dec.decode(t.float()).clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
                                       .astype(np.uint8))
    cols = 2 + len(ACTIONS)
    S = 224
    sheet = Image.new("RGB", (S * cols, (S + 30) * len(imgs)), "white"); d = ImageDraw.Draw(sheet)
    with torch.no_grad():
        for r_, (name, img) in enumerate(imgs):
            y = r_ * (S + 30)
            sheet.paste(Image.fromarray(img).resize((S, S)), (0, y + 30))
            wm.initialize_history(img)
            h = wm.get_state()
            F0 = h[0][:, -1]
            sheet.paste(to_img(F0).resize((S, S)), (S, y + 30))
            d.text((2, y + 2), f"{name}", fill="black")
            d.text((S + 2, y + 2), f"decoded  P={gs(F0, [GOAL]).item():.2f}", fill="black")
            for j, act in enumerate(ACTIONS):
                wm.set_state(h)
                F1, _ = wm.imagine(None, act)
                sheet.paste(to_img(F1).resize((S, S)), (S * (2 + j), y + 30))
                d.text((S * (2 + j) + 2, y + 2), f"{act.replace('pick and place ', '')[:30]}", fill="black")
                d.text((S * (2 + j) + 2, y + 14), f"P(goal)={gs(F1, [GOAL]).item():.2f}", fill="black")
    sheet.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
