"""Real-to-sim bridge: re-render a real arm-camera frame in the digital twin.

The WM trained on twin data imagines far better from twin renders than from real frames (on real inputs it often
removes the moved block without placing it). This fits the camera pose and each block's (x, y, yaw) to the frame's
colour masks (``fit_camera.FrameFit``, fixed fovy) and renders the twin from that state and camera, so the planner
imagines in the twin domain. Blocks are assumed to lie on the table (stacks are not reconstructed); a block whose
colour is not found is treated as in the bin.

    r2s = Real2Sim(); render, info = r2s(real_rgb)
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from verify2act.twin.config import load_config
from verify2act.twin.fit_camera import FrameFit, cam_from_params, cam_params_from_config
from verify2act.twin.scene import SHEET_THICKNESS, DofbotTwin


class Real2Sim:
    def __init__(self, config: Optional[str] = None, restarts: int = 6, max_rms: float = 2.5, seed: int = 0):
        cfg = load_config(config)
        self.twin = DofbotTwin(cfg)
        self.cam0 = cam_params_from_config(cfg)
        self.fovy = float(cfg["camera"]["fovy"])
        self.restarts, self.max_rms = restarts, max_rms
        self.rng = np.random.default_rng(seed)

    def fit(self, rgb: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        fit = FrameFit(self.twin, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        r = fit.fit(self.cam0, self.rng, self.restarts, fovy=self.fovy)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        cam = cam_from_params(r.x[:7])
        zc = SHEET_THICKNESS + self.twin.size[2] / 2
        state = {}
        for i, c in enumerate(self.twin.colors):
            if c in fit.colors:
                k = fit.colors.index(c)
                bx, by, byaw = r.x[7 + 3 * k:10 + 3 * k]
                state[c] = {"present": True, "pos": [float(bx), float(by), zc], "yaw": float(byaw)}
            else:
                state[c] = {"present": False, "pos": [0.0, 0.0, 0.0], "yaw": 0.0}
        return state, cam, rms

    def __call__(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """(twin render at the real frame's resolution, info). The render is None when the fit is poor."""
        t0 = time.time()
        state, cam, rms = self.fit(rgb)
        info = {"rms": rms, "state": state, "fit_s": round(time.time() - t0, 2)}
        if rms > self.max_rms:
            return None, info
        self.twin.set_state(state)
        render = self.twin.render(cam=cam)
        if render.shape[:2] != rgb.shape[:2]:
            render = cv2.resize(render, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        return render, info
