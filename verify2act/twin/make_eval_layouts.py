"""Fixed start layouts for the real-robot eval, so every variant is run from the same scenes.

For each eval task, --n layouts of all four blocks on the sheet (twin ``reset``), kept only if
  * every block is fully in view (margin --margin px) from *every* home camera pose fitted to real frames,
  * the task's goal does not already hold,
  * the task can be done from there: its intended plan (EVAL_PLANS) runs in the twin and reaches the goal.
Per layout the sheet shows (a) a to-scale top view of the US-Letter sheet with each block's centre in cm from the far
edge and the left edge of the sheet (as seen from the robot) and its rotation, the tag end marked, and (b) the arm-camera
view rendered from the base home pose, to compare with the live camera before starting the episode.

python -m verify2act.twin.make_eval_layouts --n 5 --main 3 --out verify2act/real_eval_layouts
"""
import argparse, json, math, os, re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from verify2act.twin.config import load_config
from verify2act.twin.eval_real2sim import GOALS
from verify2act.twin.scene import DofbotTwin, InvalidSubtask

TASKS = ["task1a", "task1b", "task1c", "task1d", "task2a", "task2b", "task3a"]
EVAL_PLANS = {   # the intended plan per task, used only to check the layout is solvable
    "task1a": ["pick and place blue block into the bin", "pick and place yellow block into the bin"],
    "task1b": ["pick and place green block into the bin", "pick and place blue block into the bin"],
    "task1c": ["pick and place red block into the bin", "pick and place yellow block into the bin"],
    "task1d": ["pick and place red block into the bin", "pick and place green block into the bin",
               "pick and place blue block into the bin"],
    "task2a": ["pick and place red block to the left of blue block"],
    "task2b": ["pick and place green block to the right of yellow block"],
    "task3a": ["pick and place blue block on yellow block"],
}
RGB = {"red": (185, 35, 30), "green": (30, 95, 60), "blue": (35, 105, 200), "yellow": (238, 205, 45)}
PX_PER_CM = 22


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def in_view_all(t, poses, margin):
    for p in poses:
        t.episode_camera = p
        if not all(t.in_view(*t.pose(c), margin=margin) for c in t.colors):
            return False
    return True


def top_view(t, state, title):
    """To-scale top view of the sheet: x (away from the robot) is up, y (robot's left) is left."""
    sx, sy = t.cfg["sheet"]["size"]
    W, H = int(sy * 100 * PX_PER_CM), int(sx * 100 * PX_PER_CM)
    pad_t, pad_b, pad_l, pad_r = 70, 90, 95, 30
    img = Image.new("RGB", (W + pad_l + pad_r, H + pad_t + pad_b), "white")
    d = ImageDraw.Draw(img)
    f, fs = font(18), font(14)
    to_px = lambda x, y: (pad_l + (sy / 2 - y) * 100 * PX_PER_CM, pad_t + (sx / 2 - x) * 100 * PX_PER_CM)
    d.rectangle([pad_l, pad_t, pad_l + W, pad_t + H], fill=(236, 237, 238), outline="black", width=2)
    for cm in range(1, int(sy * 100) + 1):   # grid: every cm light, every 5 cm darker, from the far-left corner
        x = pad_l + cm * PX_PER_CM
        d.line([x, pad_t, x, pad_t + H], fill=(200, 200, 200) if cm % 5 else (150, 150, 150))
        if cm % 5 == 0:
            d.text((x - 8, pad_t - 22), str(cm), fill="black", font=fs)
    for cm in range(1, int(sx * 100) + 1):
        y = pad_t + cm * PX_PER_CM
        d.line([pad_l, y, pad_l + W, y], fill=(200, 200, 200) if cm % 5 else (150, 150, 150))
        if cm % 5 == 0:
            d.text((pad_l - 30, y - 8), str(cm), fill="black", font=fs)
    d.text((pad_l, 8), title, fill="black", font=f)
    d.text((pad_l, pad_t - 44), "cm from the LEFT edge  →", fill=(90, 90, 90), font=fs)
    d.text((4, pad_t + H // 2 - 40), "cm\nfrom\nFAR\nedge\n↓", fill=(90, 90, 90), font=fs)
    d.text((pad_l + W // 2 - 110, pad_t + H + 12), "▲ ROBOT (near edge of the sheet)", fill="black", font=f)
    hl, hw = t.size[0] / 2, t.size[1] / 2
    for c in t.colors:
        s = state[c]
        x, y, yaw = s["pos"][0], s["pos"][1], s["yaw"]
        cs, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        corners = [(x + cs * a - sn * b, y + sn * a + cs * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        d.polygon([to_px(*p) for p in corners], fill=RGB[c], outline="black")
        # tag end = block -x end: a black bar across it
        e1, e2 = (x - cs * hl * 0.8 - sn * hw, y - sn * hl * 0.8 + cs * hw), (x - cs * hl * 0.8 + sn * hw, y - sn * hl * 0.8 - cs * hw)
        d.line([to_px(*e1), to_px(*e2)], fill="black", width=5)
        cx, cy = to_px(x, y)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill="white", outline="black")
    return img


def rows_text(t, state):
    sx, sy = t.cfg["sheet"]["size"]
    out = []
    for c in t.colors:
        x, y, yaw = state[c]["pos"][0], state[c]["pos"][1], state[c]["yaw"]
        yaw = (yaw + 180) % 360 - 180
        tag = "tag end toward robot" if abs(yaw) < 90 else "tag end away from robot"
        rot = (yaw + 90) % 180 - 90   # long-axis angle from the sheet's long side, CCW seen from above (tilt to the left)
        out.append(f"{c:6s}  far edge {100 * (sx / 2 - x):4.1f} cm   left edge {100 * (sy / 2 - y):4.1f} cm   "
                   f"rotated {rot:+5.1f}° {'(far end to the left)' if rot > 0 else '(far end to the right)' if rot < 0 else ''}"
                   f"   {tag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="layouts per task")
    ap.add_argument("--main", type=int, default=3, help="the first N are the eval set, the rest spares")
    ap.add_argument("--margin", type=float, default=25.0, help="px from the image border, for every fitted home pose")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="verify2act/real_eval_layouts")
    a = ap.parse_args()
    cfg = load_config("verify2act/configs/twin/dofbot_twin.yaml")
    t = DofbotTwin(cfg)
    t.cfg["placement"]["p_fail"] = 0.0
    base = t.base_camera
    poses = [base] + [{"pos": np.array(p["pos"], float), "lookat": np.array(p["lookat"], float),
                       "fovy": base["fovy"], "roll": float(p.get("roll", 0.0))} for p in cfg["camera"]["poses"]]
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    all_layouts = {}
    for task in TASKS:
        text, pred = GOALS[task]
        layouts = []
        while len(layouts) < a.n:
            t.episode_camera = base
            t.reset(rng, list(t.colors))
            if not in_view_all(t, poses, a.margin):
                continue
            s0 = t.get_state()
            if pred(s0):
                continue
            try:
                for step in EVAL_PLANS[task]:
                    t.apply(step, np.random.default_rng(0))
                solved = pred(t.get_state())
            except (InvalidSubtask, RuntimeError):
                solved = False
            t.set_state(s0)
            if not solved:
                continue
            m = re.match(r"pick and place (\w+) block to the (left|right) of (\w+) block", EVAL_PLANS[task][0])
            if m:   # vary the start: alternate layouts where the block starts on the wrong side / the right side
                c, side, b = m.groups()
                wrong = (s0[c]["pos"][1] - s0[b]["pos"][1]) * (1 if side == "left" else -1) < 0
                if wrong != (len(layouts) % 2 == 0):
                    continue
            layouts.append(s0)
        all_layouts[task] = []
        for i, s0 in enumerate(layouts):
            lid = f"{task}-L{i + 1}"
            role = "EVAL" if i < a.main else "SPARE"
            t.set_state(s0)
            t.episode_camera = base
            render = Image.fromarray(t.render())
            top = top_view(t, s0, "")
            rows = rows_text(t, s0)
            cam = render.resize((int(render.width * top.height / render.height * 0.62),
                                 int(top.height * 0.62)))
            W = top.width + cam.width + 40
            H = top.height + 30 * len(rows) + 110
            sheet = Image.new("RGB", (W, H), "white")
            sheet.paste(top, (0, 70))
            d = ImageDraw.Draw(sheet)
            d.text((20, 8), f"{lid}  [{role}]", fill="black", font=font(24))
            d.text((20, 40), f"goal: \"{text}\"", fill="black", font=font(19))
            d.text((top.width + 20, 110), "expected arm-camera view (home pose):", fill="black", font=font(18))
            sheet.paste(cam, (top.width + 20, 140))
            d.text((top.width + 20, 150 + cam.height),
                   "Black bar = tag end. White dot = block centre.\nPositions: centre of each block, measured from\n"
                   "the sheet's far edge (away from the robot) and\nits left edge (robot's left = camera image left).\n"
                   "Rotation: angle of the block's long side from the\nsheet's long side.",
                   fill=(60, 60, 60), font=font(15))
            for k, r in enumerate(rows):
                d.text((20, top.height + 80 + 30 * k), r, fill="black", font=font(17))
            os.makedirs(f"{a.out}/{task}", exist_ok=True)
            sheet.save(f"{a.out}/{task}/{lid}.png")
            all_layouts[task].append({"id": lid, "role": role, "goal": text, "state": s0, "placement": rows})
        print(task, [l["id"] for l in all_layouts[task]], flush=True)
    t.close()
    json.dump(all_layouts, open(f"{a.out}/layouts.json", "w"), indent=1)
    print(f"{a.out}/layouts.json")


if __name__ == "__main__":
    main()
