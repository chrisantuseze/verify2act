"""Albedo textures for the twin, cut from real camera frames ("paste the real objects on the sim blocks").

v1 (automatic, from any logged frames): a wood strip above the sheet, the sheet's colour, and one colour patch per block
(HSV mask inside the sheet, shading divided out so MuJoCo's lighting is not applied twice) used on every face.
Tags: the real blocks carry a printed fiducial on one 30x30 end face. Each block gets a random ArUco marker on its -x
end face (``nx``); ``DofbotTwin`` flips blocks by 180 deg at random, so the tag faces the camera about half the time.
v2 (``--faces faces.json``): rectified per-face photos instead. Format::

    {"red": {"nx": {"image": "red_end.jpg", "corners": [[x, y], [x, y], [x, y], [x, y]]}, ...}, ...}

Corners go clockwise from the face's top-left as it should appear on the texture. Faces are ``px nx py ny pz nz``
(block frame: x = long side, z = up); faces without a photo get the colour patch.

    python -m verify2act.twin.textures --frames verify2act/output/real/*/imagination_logs/*/request_image.png
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from verify2act.twin.config import ASSETS_DIR

TEX = 256
FACES = ("px", "nx", "py", "ny", "pz", "nz")

# OpenCV HSV (H in 0..180). Only searched inside the sheet, so the reddish wood does not count as red.
HSV_RANGES = {
    "red": [((0, 110, 50), (8, 255, 255)), ((168, 110, 50), (180, 255, 255))],
    "green": [((40, 50, 20), (90, 255, 200))],
    "blue": [((95, 100, 50), (130, 255, 255))],
    "yellow": [((18, 100, 110), (38, 255, 255))],
}


def sheet_mask(bgr: np.ndarray) -> np.ndarray:
    """Filled convex hull of the largest bright, unsaturated region (the sheet)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    if n < 2:
        return np.zeros(white.shape, np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.argwhere(labels == big)[:, ::-1].astype(np.int32)
    out = np.zeros(white.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    return out


def block_mask(bgr: np.ndarray, color: str, inside: np.ndarray) -> np.ndarray:
    """Largest connected component of the colour inside the sheet hull (eroded, to stay off edges)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(bgr.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[color]:
        m |= cv2.inRange(hsv, lo, hi)
    m &= inside
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 400:
        return np.zeros_like(m)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cv2.erode((labels == big).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))


def albedo_patch(bgr: np.ndarray, mask: np.ndarray, size: int = TEX) -> Optional[np.ndarray]:
    """Square texture: the brightest (top-face) pixels' grain around their median colour, shading removed."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2][ys, xs]
    top = v >= np.percentile(v, 50)
    ys, xs = ys[top], xs[top]
    median = np.median(bgr[ys, xs].astype(np.float32), axis=0)
    # The largest square window fully inside the top-face pixels gives the grain.
    top_mask = np.zeros(mask.shape, np.uint8)
    top_mask[ys, xs] = 1
    dist = cv2.distanceTransform(top_mask, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    r = max(4, int(dist[cy, cx] / 1.5))
    crop = bgr[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
    low = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(2.0, r / 2))
    grain = crop / np.maximum(low, 1.0)
    tile = cv2.resize(grain, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(tile * median, 0, 255).astype(np.uint8)


def wood_texture(frames: List[np.ndarray], size: int = 512) -> Optional[np.ndarray]:
    """Rows above the sheet in every frame (pure wood), mirror-stacked into a square."""
    strips = []
    for bgr in frames:
        sm = sheet_mask(bgr)
        rows = np.nonzero(sm.any(axis=1))[0]
        top = int(rows.min()) if len(rows) else bgr.shape[0]
        if top < 12:
            continue
        strip = bgr[:top - 4]
        # A block poking out above the sheet would be tiled all over the table. The wood is reddish but darker and
        # less saturated than the red block, hence the stricter red test.
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        blocky = sum(int((cv2.inRange(hsv, lo, hi) > 0).sum())
                     for c in ("green", "blue", "yellow") for lo, hi in HSV_RANGES[c])
        blocky += int((cv2.inRange(hsv, (0, 150, 110), (8, 255, 255)) > 0).sum())
        blocky += int((cv2.inRange(hsv, (168, 150, 110), (180, 255, 255)) > 0).sum())
        if blocky < 20:
            strips.append(strip)
    if not strips:
        return None
    strip = max(strips, key=len)
    tile = np.concatenate([strip, strip[::-1]], axis=0)
    reps = int(np.ceil(size / tile.shape[0]))
    tall = np.concatenate([tile] * reps, axis=0)[:max(size, tile.shape[0])]
    return cv2.resize(tall, (size, size), interpolation=cv2.INTER_AREA)


def sheet_texture(frames: List[np.ndarray], size: int = 256, seed: int = 0) -> Optional[np.ndarray]:
    """Median sheet colour (excluding blocks) with faint paper noise; lighting gradients come from the renderer."""
    px = []
    for bgr in frames:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255)) & sheet_mask(bgr)
        px.append(bgr[white > 0])
    px = np.concatenate(px) if px else np.zeros((0, 3))
    if len(px) < 1000:
        return None
    # Brightest half: the well-lit paper the renderer's key light reproduces.
    lum = px.mean(axis=1)
    col = np.median(px[lum >= np.percentile(lum, 50)].astype(np.float32), axis=0)
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 3.0, (size, size, 1)).astype(np.float32), (0, 0), 1.2)[..., None]
    return np.clip(col + noise, 0, 255).astype(np.uint8)


def tag_face(patch: np.ndarray, rng: np.random.Generator, size: int = TEX) -> np.ndarray:
    """Block-coloured end face with a random ArUco marker on a white square, as on the real blocks."""
    face = cv2.resize(patch, (size, size))
    white = int(size * 0.80)
    marker = int(white * 0.78)
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(aruco, int(rng.integers(50)), marker)
    o = (size - white) // 2
    face[o:o + white, o:o + white] = 235
    m = (size - marker) // 2
    face[m:m + marker, m:m + marker] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(face, (0, 0), 0.8)


def rectify_face(image_path: str, corners: List[List[float]], size: int = TEX) -> np.ndarray:
    src = np.array(corners, np.float32)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (size, size))


def build(frames: List[str], out_dir: Path, faces: Optional[Dict] = None, tags: bool = True,
          seed: int = 0) -> Dict[str, str]:
    """Write ``wood.png``, ``sheet.png`` and ``<color>.png`` or ``<color>_<face>.png`` into ``out_dir``."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [cv2.imread(f) for f in frames]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise SystemExit("no readable frames")
    report = {}
    for name, tex in (("wood", wood_texture(imgs)), ("sheet", sheet_texture(imgs))):
        if tex is not None:
            cv2.imwrite(str(out_dir / f"{name}.png"), tex)
            report[name] = "ok"
        else:
            report[name] = "not found (flat colour from the YAML)"
    for color in HSV_RANGES:
        patches = []
        for bgr in imgs:
            p = albedo_patch(bgr, block_mask(bgr, color, sheet_mask(bgr)))
            if p is not None:
                patches.append(p)
        if not patches:
            report[color] = "not found (flat colour from the YAML)"
            continue
        # Patch closest to the per-pixel median over frames: one good sample, not a blur of several.
        med = np.median(np.stack(patches), axis=0)
        patch = min(patches, key=lambda p: float(np.abs(p.astype(np.float32) - med).mean()))
        face_spec = (faces or {}).get(color, {})
        if face_spec or tags:
            for f in FACES:
                spec = face_spec.get(f)
                if spec:
                    img = rectify_face(spec["image"], spec["corners"])
                elif tags and f == "nx":
                    img = tag_face(patch, rng)
                else:
                    img = patch
                cv2.imwrite(str(out_dir / f"{color}_{f}.png"), img)
            (out_dir / f"{color}.png").unlink(missing_ok=True)
            extra = [f"faces {sorted(face_spec)}"] if face_spec else []
            extra += ["random tag on nx"] if tags and "nx" not in face_spec else []
            report[color] = f"ok ({len(patches)} frames) + " + ", ".join(extra)
        else:
            cv2.imwrite(str(out_dir / f"{color}.png"), patch)
            for f in FACES:
                (out_dir / f"{color}_{f}.png").unlink(missing_ok=True)
            report[color] = f"ok ({len(patches)} frames)"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="+", required=True, help="real frames (globs allowed)")
    ap.add_argument("--faces", default=None, help="JSON with rectified per-face photos (see the module doc)")
    ap.add_argument("--out", default=str(ASSETS_DIR))
    ap.add_argument("--no-tags", action="store_true", help="plain end faces")
    ap.add_argument("--seed", type=int, default=0, help="tag ids")
    args = ap.parse_args()
    frames = sorted({p for pat in args.frames for p in glob.glob(pat)})
    faces = json.loads(Path(args.faces).read_text()) if args.faces else None
    for k, v in build(frames, Path(args.out), faces, tags=not args.no_tags, seed=args.seed).items():
        print(f"{k:7s} {v}")


if __name__ == "__main__":
    main()
