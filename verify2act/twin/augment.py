"""Photometric realism for twin renders: the cheap webcam look (white balance, exposure, soft focus, noise, vignetting,
JPEG). The camera parameters are drawn once per episode (same camera, same room) and the noise once per frame."""

import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


def sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    a = cfg["augment"]
    return {
        "brightness": rng.uniform(*a["brightness"]),
        "contrast": rng.uniform(*a["contrast"]),
        "saturation": rng.uniform(*a["saturation"]),
        "gamma": rng.uniform(*a["gamma"]),
        "wb": rng.uniform(*a["wb_gain"], size=3),
        "blur": rng.uniform(*a["blur_sigma"]),
        "noise": rng.uniform(*a["noise_std"]),
        "vignette": rng.uniform(*a["vignette"]),
        "jpeg": int(rng.integers(a["jpeg_quality"][0], a["jpeg_quality"][1] + 1)),
    }


def apply(img: np.ndarray, p: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """RGB uint8 -> RGB uint8."""
    x = img.astype(np.float32) / 255.0
    x = x * p["wb"]
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * p["saturation"]
    x = (x - 0.5) * p["contrast"] + 0.5 + p["brightness"]
    x = np.clip(x, 0, 1) ** p["gamma"]
    if p["vignette"] > 0:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x = x * (1 - p["vignette"] * np.clip(r2 / 2, 0, 1))[..., None]
    if p["blur"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), p["blur"])
    x = x + rng.normal(0, p["noise"], x.shape).astype(np.float32)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    # The server receives base64 JPEG from the Jetson: bake in the same kind of artefacts.
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=p["jpeg"])
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
