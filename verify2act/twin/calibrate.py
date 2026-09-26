"""Checkerboard calibration of the arm camera at the home pose -> the ``camera:`` block of the twin YAML.

1. Intrinsics: 10–20 photos of a checkerboard at varied angles and distances (``--images``).
2. Home pose: one photo with the arm at the home/observation pose and the board lying flat **at the sheet centre**
   (``--home``). The board centre becomes the world origin; world x is the camera's viewing direction projected on the
   table (away from the robot), z is up, so the board's own orientation on the table does not matter.

    python -m verify2act.twin.calibrate --images calib/*.jpg --home calib/home.jpg --board 9 6 --square 0.024

``--board`` counts inner corners (columns rows). Paste the printed YAML over ``camera:`` in dofbot_twin.yaml (keep the
jitter entries). The renderer assumes a centred principal point; a large offset is reported.
"""

import argparse
import glob
import math
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def find_corners(path: str, board: Tuple[int, int]):
    img = cv2.imread(path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None, gray.shape[::-1]
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return corners, gray.shape[::-1]


def board_points(board: Tuple[int, int], square: float) -> np.ndarray:
    """Board corners in the board frame, centred on the board."""
    cols, rows = board
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    pts[:, :2] -= pts[:, :2].mean(axis=0)
    return pts


def home_pose(R: np.ndarray, t: np.ndarray) -> dict:
    """OpenCV board->camera pose -> twin camera (pos, lookat, roll) in the world frame defined in the module doc."""
    C = (-R.T @ t).ravel()                                   # camera centre in the board frame
    x_cv, y_cv, z_cv = R[0], R[1], R[2]                      # camera axes in the board frame
    if C[2] < 0:                                             # board z points into the table: rotate 180 deg about x
        flip = np.diag([1.0, -1.0, -1.0])
        C, x_cv, y_cv, z_cv = flip @ C, flip @ x_cv, flip @ y_cv, flip @ z_cv
    zw = np.array([0.0, 0.0, 1.0])
    xw = np.array([z_cv[0], z_cv[1], 0.0])
    xw /= np.linalg.norm(xw)
    yw = np.cross(zw, xw)
    W = np.stack([xw, yw, zw])                               # board -> world rotation (rows = world axes)
    pos, fwd, right = W @ C, W @ z_cv, W @ x_cv
    lookat = pos + fwd * (-pos[2] / fwd[2])
    r0 = np.cross(fwd, zw)
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, fwd)
    roll = math.degrees(math.atan2(float(right @ u0), float(right @ r0)))
    return {"pos": pos.round(4).tolist(), "lookat": lookat.round(4).tolist(), "roll": round(roll, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="intrinsics photos (globs allowed)")
    ap.add_argument("--home", required=True, help="board flat at the sheet centre, arm at the home pose")
    ap.add_argument("--board", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"), help="inner corners")
    ap.add_argument("--square", type=float, required=True, help="square size (m)")
    args = ap.parse_args()

    board = tuple(args.board)
    obj = board_points(board, args.square)
    paths: List[str] = sorted({p for pat in args.images for p in glob.glob(pat)} | {args.home})
    objpoints, imgpoints, size = [], [], None
    for p in paths:
        corners, sz = find_corners(p, board)
        size = size or sz
        if corners is None:
            print(f"  no board: {p}")
            continue
        objpoints.append(obj)
        imgpoints.append(corners)
    if len(objpoints) < 3:
        raise SystemExit("need the board in at least 3 images")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    print(f"intrinsics from {len(objpoints)} images, RMS reprojection {rms:.3f} px")

    corners, _ = find_corners(args.home, board)
    if corners is None:
        raise SystemExit(f"no board in the home image {args.home}")
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    if not ok:
        raise SystemExit("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    cam = home_pose(R, tvec)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    w, h = size
    cam.update({"fovy": round(math.degrees(2 * math.atan(h / 2 / fy)), 3), "width": int(w), "height": int(h),
                "intrinsics": {"fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2), "cy": round(cy, 2)}})
    off = math.hypot(cx - w / 2, cy - h / 2)
    if off > 10:
        print(f"note: principal point is {off:.0f} px off-centre; the twin renders with a centred one")
    if abs(fx - fy) / fy > 0.02:
        print(f"note: fx/fy = {fx / fy:.3f}; the twin renders square pixels")
    print(f"distortion k1..: {np.round(dist.ravel(), 4).tolist()} (the twin renders without distortion)")
    print(yaml.safe_dump({"camera": cam}, default_flow_style=None, sort_keys=False).strip())


if __name__ == "__main__":
    main()
