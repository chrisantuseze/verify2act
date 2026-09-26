"""Fit the arm camera's home pose from logged real frames: no checkerboard needed.

The known geometry fixes the scale: blocks are 30x30x60 mm lying on a long side, and the sheet is US Letter in
portrait (216 mm wide, its far edge at x = +139.5 mm in the world frame). For every frame the camera (position, yaw,
pitch, roll, fovy) and each visible block's (x, y, yaw) are fitted by least squares on image moments (centroid, size,
second moments) of the projected silhouettes vs the colour masks. Moments of a polygon vary smoothly with the
parameters, unlike pixel IoU. The camera's median over frames is the base pose; the spread is the home-pose jitter.

    MUJOCO_GL=egl python -m verify2act.twin.fit_camera --frames 'verify2act/output/real/*/imagination_logs/*/request_image.png' \\
        --out-dir verify2act/output/twin/camera_fit
"""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from verify2act.twin.config import DEFAULT_CONFIG, load_config
from verify2act.twin.scene import SHEET_THICKNESS, DofbotTwin, camera_frame
from verify2act.twin.textures import sheet_mask

# Looser than textures.HSV_RANGES: the silhouette needs the dark side faces too.
SIL_HSV = {
    "red": [((0, 100, 40), (9, 255, 255)), ((165, 100, 40), (180, 255, 255))],
    "green": [((38, 40, 12), (95, 255, 220))],
    "blue": [((92, 90, 35), (132, 255, 255))],
    "yellow": [((16, 90, 90), (40, 255, 255))],
}
MIN_AREA = 600


def real_masks(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    sheet = sheet_mask(bgr)
    near = cv2.dilate(sheet, np.ones((41, 41), np.uint8))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blocks = {}
    for c, ranges in SIL_HSV.items():
        m = np.zeros(bgr.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        m &= near
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < MIN_AREA:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blocks[c] = ((labels == big) * 255).astype(np.uint8)
    return sheet, blocks


def clip_poly(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sutherland–Hodgman clip of a polygon to the image rectangle."""
    def clip(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            if inside(b):
                if not inside(a):
                    out.append(inter(a, b))
                out.append(b)
            elif inside(a):
                out.append(inter(a, b))
        return out

    def at_x(x0):
        return lambda a, b: a + (b - a) * ((x0 - a[0]) / (b[0] - a[0]))

    def at_y(y0):
        return lambda a, b: a + (b - a) * ((y0 - a[1]) / (b[1] - a[1]))

    pts = [np.asarray(p, float) for p in poly]
    for inside, inter in ((lambda p: p[0] >= 0, at_x(0)), (lambda p: p[0] <= w, at_x(w)),
                          (lambda p: p[1] >= 0, at_y(0)), (lambda p: p[1] <= h, at_y(h))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return np.array(pts, float).reshape(-1, 2)


def features(m: dict) -> np.ndarray:
    a = max(m["m00"], 1e-6)
    return np.array([m["m10"] / a, m["m01"] / a, math.sqrt(a),
                     math.sqrt(max(m["mu20"] / a, 0)), math.sqrt(max(m["mu02"] / a, 0)), m["mu11"] / a / 20.0])


def poly_features(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    p = clip_poly(poly, w, h)
    if len(p) < 3:
        return np.zeros(6)
    return features(cv2.moments(p.astype(np.float32)))


def mask_features(mask: np.ndarray) -> np.ndarray:
    return features(cv2.moments(mask, binaryImage=True))


_SCALE = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 3.0])


def cam_from_params(p: np.ndarray) -> dict:
    """p = [x, y, z, yaw, pitch, roll, fovy] -> twin camera dict (look-at point on the table)."""
    x, y, z, yaw, pitch, roll, fovy = p
    yw, pt = math.radians(yaw), math.radians(pitch)
    d = np.array([math.cos(pt) * math.cos(yw), math.cos(pt) * math.sin(yw), -math.sin(pt)])
    pos = np.array([x, y, z])
    return {"pos": pos, "lookat": pos + d * (z / math.sin(pt)), "fovy": float(fovy), "roll": float(roll)}


class FrameFit:
    def __init__(self, twin: DofbotTwin, bgr: np.ndarray):
        self.t = twin
        self.h, self.w = bgr.shape[:2]
        sheet, self.blocks = real_masks(bgr)
        self.colors = sorted(self.blocks)
        # Distance to the real sheet boundary, ignoring where the sheet simply leaves the image.
        edge = cv2.morphologyEx(sheet, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge[:3], edge[-3:], edge[:, :3], edge[:, -3:] = 0, 0, 0, 0
        self.sheet_dt = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 5)
        self.f_sheet = mask_features(sheet)[:3]
        self.f_blocks = {c: mask_features(m) for c, m in self.blocks.items()}
        cfg = twin.cfg
        sx, sy = cfg["sheet"]["size"]
        self.sheet_poly3d = np.array([[sx / 2, sy / 2, 0], [sx / 2, -sy / 2, 0], [-sx / 2, -sy / 2, 0],
                                      [-sx / 2, sy / 2, 0]])
        self.zc = SHEET_THICKNESS + twin.size[2] / 2

    def silhouette(self, cam: dict, x: float, y: float, yaw: float) -> np.ndarray:
        uv = self.t.project(self.t.block_corners((x, y, self.zc), yaw), cam)
        if np.isnan(uv).any():
            return np.zeros((0, 2))
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)

    def sheet_edge_residuals(self, cam: dict, n: int = 16, scale: float = 6.0) -> np.ndarray:
        """Chamfer term: points along each sheet edge -> distance (px) to the real sheet boundary. Points outside the
        image (the near part of the sheet, under the robot) contribute 0."""
        res = np.zeros(4 * n)
        for i in range(4):
            a, b = self.sheet_poly3d[i], self.sheet_poly3d[(i + 1) % 4]
            pts = a + (b - a) * np.linspace(0.02, 0.98, n)[:, None]
            uv = self.t.project(pts, cam)
            ok = ~np.isnan(uv).any(axis=1)
            ok &= (uv[:, 0] > 4) & (uv[:, 0] < self.w - 5) & (uv[:, 1] > 4) & (uv[:, 1] < self.h - 5)
            if ok.any():
                d = map_coordinates(self.sheet_dt, [uv[ok, 1], uv[ok, 0]], order=1)
                res[i * n:(i + 1) * n][ok] = np.minimum(d, 60.0) / scale
        return res

    def sheet_poly(self, cam: dict) -> np.ndarray:
        # Densify the sheet edges before projecting so clipping of the part behind the camera stays sane.
        pts = []
        for i in range(4):
            a, b = self.sheet_poly3d[i], self.sheet_poly3d[(i + 1) % 4]
            pts.extend(a + (b - a) * s for s in np.linspace(0, 1, 12, endpoint=False))
        uv = self.t.project(np.array(pts), cam)
        return uv[~np.isnan(uv).any(axis=1)]

    def residuals(self, x: np.ndarray) -> np.ndarray:
        cam = cam_from_params(x[:7])
        # Edge chamfer + a weak centroid/size term (edges alone could be zeroed by moving the sheet out of view).
        f_sheet = poly_features(self.sheet_poly(cam), self.w, self.h)[:3]
        res = [self.sheet_edge_residuals(cam), 0.5 * (f_sheet - self.f_sheet) / _SCALE[:3]]
        for i, c in enumerate(self.colors):
            bx, by, byaw = x[7 + 3 * i:10 + 3 * i]
            res.append((poly_features(self.silhouette(cam, bx, by, byaw), self.w, self.h) - self.f_blocks[c]) / _SCALE)
        return np.concatenate(res)

    def backproject(self, cam: dict, u: float, v: float, z: float) -> np.ndarray:
        R = camera_frame(np.asarray(cam["pos"]), np.asarray(cam["lookat"]), cam["roll"])
        f = (self.h / 2) / math.tan(math.radians(cam["fovy"]) / 2)
        d = R @ np.array([(u - self.w / 2) / f, -(v - self.h / 2) / f, -1.0])
        s = (z - cam["pos"][2]) / d[2]
        return np.asarray(cam["pos"]) + s * d

    def init_blocks(self, cam: dict) -> List[float]:
        out = []
        for c in self.colors:
            m = cv2.moments(self.blocks[c], binaryImage=True)
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            ang = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])      # image long axis
            p0 = self.backproject(cam, u, v, self.zc)
            p1 = self.backproject(cam, u + 15 * math.cos(ang), v + 15 * math.sin(ang), self.zc)
            yaw = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            yaw = (yaw + 90) % 180 - 90
            out += [p0[0], p0[1], yaw]
        return out

    def fit(self, cam0: np.ndarray, rng: np.random.Generator, restarts: int = 12, fovy: Optional[float] = None):
        lo = [-0.5, -0.3, 0.12, -60, 25, -15, 20] + [-0.4, -0.4, -400] * len(self.colors)
        hi = [0.3, 0.3, 0.8, 60, 89, 15, 80] + [0.5, 0.4, 400] * len(self.colors)
        if fovy is not None:                     # same lens in every frame: only the pose varies
            cam0 = np.array(cam0, float)
            cam0[6] = fovy
            lo[6], hi[6] = fovy - 1e-3, fovy + 1e-3
        best = None
        for k in range(restarts):
            c = np.array(cam0, float)
            if k:
                c += rng.normal(0, [0.04, 0.03, 0.06, 6, 8, 2, 0 if fovy is not None else 8])
                c = np.clip(c, np.array(lo[:7]) + 1e-3, np.array(hi[:7]) - 1e-3)
            x0 = np.concatenate([c, self.init_blocks(cam_from_params(c))])
            x0 = np.clip(x0, np.array(lo) + 1e-3, np.array(hi) - 1e-3)
            try:
                r = least_squares(self.residuals, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=400)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return best


def cam_params_from_config(cfg: dict) -> np.ndarray:
    cam = cfg["camera"]
    pos, look = np.array(cam["pos"], float), np.array(cam["lookat"], float)
    d = look - pos
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
    return np.array([*pos, yaw, pitch, cam.get("roll", 0.0), cam["fovy"]])


def draw(bgr: np.ndarray, fit: FrameFit, x: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    cam = cam_from_params(x[:7])
    sp = clip_poly(fit.sheet_poly(cam), fit.w, fit.h)
    if len(sp) >= 3:
        cv2.polylines(out, [sp.astype(np.int32)], True, (255, 0, 255), 2)
    col = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 128, 0), "yellow": (0, 255, 255)}
    for i, c in enumerate(fit.colors):
        bx, by, byaw = x[7 + 3 * i:10 + 3 * i]
        uv = fit.t.project(fit.t.block_corners((bx, by, fit.zc), byaw), cam)
        for face in (uv[:4], uv[4:]):
            cv2.polylines(out, [face.astype(np.int32)], True, col[c], 1)
        for a, b in zip(uv[:4], uv[4:]):
            cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), col[c], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out-dir", default="verify2act/output/twin/camera_fit")
    ap.add_argument("--config", default=None)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--max-rms", type=float, default=1.5, help="frames with a worse fit are left out of the median")
    ap.add_argument("--exclude", nargs="*", default=[], help="substrings of frames to leave out of the median")
    ap.add_argument("--write", action="store_true", help="write the fitted camera into the config YAML")
    ap.add_argument("--fovy", type=float, default=None,
                    help="fix the vertical FOV (deg). Default: two passes, free per frame, then fixed at their median")
    args = ap.parse_args()

    cfg = load_config(args.config)
    twin = DofbotTwin(cfg, render=False)
    rng = np.random.default_rng(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted({p for pat in args.frames for p in glob.glob(pat)})
    cam0 = cam_params_from_config(cfg)
    fovy = args.fovy
    if fovy is None:
        free = []
        for path in frames:
            r = FrameFit(twin, cv2.imread(path)).fit(cam0, rng, args.restarts)
            if np.sqrt(np.mean(r.fun ** 2)) <= args.max_rms:
                free.append(r.x[6])
        fovy = float(np.median(free))
        print(f"pass 1: fovy per frame {np.round(free, 1).tolist()} -> fixed at the median {fovy:.2f}")
    results = []
    for i, path in enumerate(frames):
        bgr = cv2.imread(path)
        fit = FrameFit(twin, bgr)
        r = fit.fit(cam0, rng, args.restarts, fovy=fovy)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        cam = cam_from_params(r.x[:7])
        blocks = {c: [round(float(v), 4) for v in r.x[7 + 3 * k:10 + 3 * k]] for k, c in enumerate(fit.colors)}
        results.append({"frame": path, "rms": rms, "params": r.x[:7].tolist(), "blocks": blocks,
                        "pos": cam["pos"].tolist(), "lookat": cam["lookat"].tolist()})
        cv2.imwrite(str(out / f"fit_{i:02d}.png"), draw(bgr, fit, r.x))
        p = r.x[:7]
        print(f"[{i:2d}] rms {rms:5.2f}  pos {np.round(p[:3], 3)}  yaw {p[3]:6.1f}  pitch {p[4]:5.1f}  "
              f"roll {p[5]:5.1f}  fovy {p[6]:5.1f}  blocks {len(fit.colors)}  {path}")

    good = [r for r in results if r["rms"] <= args.max_rms and not any(s in r["frame"] for s in args.exclude)]
    if not good:
        raise SystemExit("no frame fitted well; loosen --max-rms or check the masks in the fit images")
    P = np.array([r["params"] for r in good])
    med = np.median(P, axis=0)
    base = cam_from_params(med)
    poses = []
    for r in good:
        c = cam_from_params(np.array(r["params"]))
        poses.append({"pos": np.round(c["pos"], 4).tolist(), "lookat": np.round(c["lookat"], 4).tolist(),
                      "roll": round(float(c["roll"]), 2)})
    camera = {"pos": np.round(base["pos"], 4).tolist(), "lookat": np.round(base["lookat"], 4).tolist(),
              "fovy": round(float(med[6]), 2), "roll": round(float(med[5]), 2), "poses": poses}
    (out / "fits.json").write_text(json.dumps({"frames": results, "used": [r["frame"] for r in good],
                                               "camera": camera}, indent=2))
    spread = P[:, :6].std(axis=0)
    print(f"\n{len(good)}/{len(results)} frames used. Pose spread (std): pos {np.round(spread[:3], 3).tolist()} m, "
          f"yaw/pitch/roll {np.round(spread[3:6], 1).tolist()} deg")
    if args.write:
        _write_camera(Path(args.config) if args.config else DEFAULT_CONFIG, camera)
    else:
        print(yaml.safe_dump({"camera": camera}, default_flow_style=None, sort_keys=False).strip())


def _write_camera(path: Path, camera: dict) -> None:
    """Replace pos/lookat/fovy/roll and the poses list in the YAML's camera block, keeping everything else (and the
    comments outside the replaced lines)."""
    lines = path.read_text().splitlines()
    start = lines.index("camera:")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" "))
    body = [l for l in lines[start + 1:end]
            if not l.strip().startswith(("pos:", "lookat:", "fovy:", "roll:", "poses:", "- {pos:"))]
    new = [f"  pos: {camera['pos']}", f"  lookat: {camera['lookat']}", f"  fovy: {camera['fovy']}",
           f"  roll: {camera['roll']}",
           "  # Home poses fitted to real frames (fit_camera.py); each episode samples one, then adds jitter_episode.",
           "  poses:"]
    new += [f"    - {{pos: {p['pos']}, lookat: {p['lookat']}, roll: {p['roll']}}}" for p in camera["poses"]]
    body = [l for l in body if "Home poses fitted" not in l]
    k = next((i for i, l in enumerate(body) if not l.strip().startswith("#")), len(body))   # leading comments stay on top
    out = lines[:start + 1] + body[:k] + new + body[k:] + [""] + lines[end:]
    path.write_text("\n".join(out).replace("\n\n\n", "\n\n").rstrip("\n") + "\n")
    print(f"wrote the camera block of {path}")


if __name__ == "__main__":
    main()
