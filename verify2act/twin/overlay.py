"""Check (and hand-tune) the twin camera against a real frame: the projected sheet outline and block boxes are drawn on
the real image next to a twin render. Camera flags override the YAML; ``--pitch/--dist`` place the camera by the pitch
angle and distance from the look-at point.

    MUJOCO_GL=egl python -m verify2act.twin.overlay --frame real.png --out overlay.png
    MUJOCO_GL=egl python -m verify2act.twin.overlay --frame real.png --pitch 65 --dist 0.40 --fovy 34 \\
        --blocks '{"red": [-0.03, 0.0, 2], "yellow": [0.09, 0.0, -3]}'

``--blocks`` places blocks on the table at (x, y, yaw_deg) so their outlines can be matched to the real blocks.
"""

import argparse
import json
import math

import cv2
import numpy as np
import yaml
from PIL import Image

from verify2act.twin.config import load_config
from verify2act.twin.scene import SHEET_THICKNESS, DofbotTwin

_BGR = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 80, 0), "yellow": (0, 220, 255)}


def camera_override(args) -> dict:
    cam = {}
    if args.pitch is not None or args.dist is not None:
        if args.pitch is None or args.dist is None:
            raise SystemExit("--pitch and --dist go together")
        look = np.array(args.lookat if args.lookat else load_config()["camera"]["lookat"], float)
        p = math.radians(args.pitch)
        cam["pos"] = (look - args.dist * np.array([math.cos(p), 0.0, -math.sin(p)])).round(4).tolist()
        cam["lookat"] = look.tolist()
    else:
        if args.pos:
            cam["pos"] = args.pos
        if args.lookat:
            cam["lookat"] = args.lookat
    if args.fovy is not None:
        cam["fovy"] = args.fovy
    if args.roll is not None:
        cam["roll"] = args.roll
    return cam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pos", type=float, nargs=3)
    ap.add_argument("--lookat", type=float, nargs=3)
    ap.add_argument("--pitch", type=float, help="degrees below horizontal")
    ap.add_argument("--dist", type=float, help="camera to look-at distance (m)")
    ap.add_argument("--fovy", type=float)
    ap.add_argument("--roll", type=float)
    ap.add_argument("--sheet", type=float, nargs=2, help="sheet size x y (m)")
    ap.add_argument("--blocks", default=None, help='JSON {"color": [x, y, yaw_deg], ...}')
    args = ap.parse_args()

    overrides = {"camera": camera_override(args)}
    if args.sheet:
        overrides["sheet"] = {"size": args.sheet}
    cfg = load_config(args.config, overrides)
    t = DofbotTwin(cfg)
    for i, c in enumerate(t.colors):
        t.present[c] = False
        t._set_pose(c, t._park_pos(i), 0.0)
    blocks = json.loads(args.blocks) if args.blocks else {}
    for c, (x, y, yaw) in blocks.items():
        t.present[c] = True
        t._set_pose(c, (x, y, SHEET_THICKNESS + t.size[2] / 2), yaw)
    render = t.render(cam=t.base_camera)

    real = cv2.imread(args.frame)
    if real is None:
        raise SystemExit(f"cannot read {args.frame}")
    real = cv2.resize(real, (t.width, t.height))
    sx, sy = cfg["sheet"]["size"]
    ox, oy = cfg["sheet"]["offset"]
    sheet = np.array([[ox + sx / 2, oy + sy / 2, 0], [ox + sx / 2, oy - sy / 2, 0],
                      [ox - sx / 2, oy - sy / 2, 0], [ox - sx / 2, oy + sy / 2, 0]])
    uv = t.project(sheet)
    cv2.polylines(real, [np.nan_to_num(uv).astype(np.int32)], True, (255, 0, 255), 2)
    for c in blocks:
        xyz, yaw = t.pose(c)
        corners = t.project(t.block_corners(xyz, yaw))
        for face in (corners[:4], corners[4:]):
            cv2.polylines(real, [np.nan_to_num(face).astype(np.int32)], True, _BGR.get(c, (255, 255, 255)), 1)
    both = np.concatenate([real[..., ::-1], render], axis=1)
    Image.fromarray(both).save(args.out)
    print(f"wrote {args.out}")
    print(yaml.safe_dump({"camera": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in overrides["camera"].items()}}, default_flow_style=None).strip())
    t.close()


if __name__ == "__main__":
    main()
