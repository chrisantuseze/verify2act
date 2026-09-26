#!/usr/bin/env python3
"""Verify2Act server for the real DOFBOT. Runs on the lab GPU PC, not on the Jetson.

It connects OUT to the Jetson's rosbridge with roslibpy (no ROS install needed here), listens on
/v2a/request and answers on /v2a/response (see protocol.py). The Jetson keeps the episode loop and
skill execution; this process only runs the models.

    Jetson:  roscore, arm_driver, camera, ..., roslaunch rosbridge_server rosbridge_websocket.launch
    Lab PC:  conda activate verify2act
             python -m verify2act.robot.server --jetson-ip 192.168.0.8 [--wm-mode v2a_wm|rla_wm|diffusion_wm|vlm_only]

Offline check without the Jetson (loads the models, plans once on an image file):
    python -m verify2act.robot.server --offline-image frame.jpg --goal "Put the blue block and the yellow block into the bin"
"""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verify2act.robot.protocol import MSG_TYPE, REQUEST_TOPIC, RESPONSE_TOPIC, decode_image_rgb

logging.getLogger("twisted").setLevel(logging.WARNING)
logger = logging.getLogger("v2a_robot_server")


class Server:
    def __init__(self, backend, jetson_ip: str, port: int = 9090, connect_timeout: float = 15.0):
        import roslibpy

        self._roslibpy = roslibpy
        self.backend = backend
        self.lock = threading.Lock()   # one model call at a time (single GPU)

        logger.info("Connecting to rosbridge ws://%s:%d ...", jetson_ip, port)
        self.ros = roslibpy.Ros(host=jetson_ip, port=port)
        try:
            self.ros.run(timeout=connect_timeout)
        except Exception:
            pass   # roslibpy raises on timeout; report it below with the actionable message
        if not self.ros.is_connected:
            raise ConnectionError(f"Could not reach rosbridge at {jetson_ip}:{port}. Is "
                                  "`roslaunch rosbridge_server rosbridge_websocket.launch` running on the Jetson?")
        self.pub = roslibpy.Topic(self.ros, RESPONSE_TOPIC, MSG_TYPE)
        self.pub.advertise()
        self.sub = roslibpy.Topic(self.ros, REQUEST_TOPIC, MSG_TYPE)
        self.sub.subscribe(self.on_request)
        logger.info("Ready: listening on %s, answering on %s", REQUEST_TOPIC, RESPONSE_TOPIC)

    def on_request(self, msg: dict):
        # roslibpy callbacks run on the websocket thread; work elsewhere so pings stay responsive.
        threading.Thread(target=self.handle, args=(msg["data"],), daemon=True).start()

    def handle(self, raw: str):
        reply = handle_request(self.backend, raw, self.lock)
        if reply is not None:
            self.pub.publish(self._roslibpy.Message({"data": json.dumps(reply)}))

    def run(self):
        try:
            while self.ros.is_connected:
                time.sleep(1.0)
            logger.error("Lost the rosbridge connection.")
        except KeyboardInterrupt:
            pass
        finally:
            for fn in (self.sub.unsubscribe, self.pub.unadvertise, self.ros.terminate):
                try:
                    fn()
                except Exception:
                    pass


def handle_request(backend, raw: str, lock: threading.Lock):
    """Decode one request, run it, and return the reply dict (None if the request is not JSON)."""
    try:
        req = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring non-JSON request: %r", raw[:200])
        return None
    rid, op = req.get("id"), req.get("op")
    t0 = time.time()
    try:
        if op == "ping":
            out = dict(getattr(backend, "settings", {}))   # never takes the model lock
        else:
            with lock:
                out = dispatch(backend, op, req)
        reply = {"id": rid, "ok": True, **out}
    except Exception as e:
        logger.exception("op %s failed", op)
        reply = {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}
    logger.info("%-6s %.2fs", op, time.time() - t0)
    return reply


def dispatch(backend, op: str, r: dict) -> dict:
    if op == "reset":
        backend.reset(r["session"])
        return {}
    if op == "plan":
        return backend.plan(
            session=r.get("session", "default"),
            image_rgb=decode_image_rgb(r["image"]),
            goal=r["goal"],
            history=r.get("history"),
            obj_labels=r.get("obj_labels"),
            horizon=r.get("horizon"),
            feedback=r.get("feedback") or "",
        )
    raise ValueError(f"unknown op '{op}'")


# Per-variant checkpoints (CALVIN, as in commands/v2a_wm.sh), used when the flag is not given.
_V2A_CALVIN = "verify2act/output/v2a_wm/calvin"
MODE_DEFAULTS = {
    "v2a_wm": {"latent_wm_ckpt": f"{_V2A_CALVIN}/wm/ckpt/latent_dynamics_best_weights.pt",
               "wm_decoder_dir": f"{_V2A_CALVIN}/decoder"},
    "rla_wm": {"latent_wm_ckpt": "verify2act/output/rla_wm/calvin/wm/ckpt/latent_dynamics_best.pt",
               "wm_decoder_dir": f"{_V2A_CALVIN}/decoder"},
    "diffusion_wm": {"wm_decoder_dir": "verify2act/output/diffusion_wm/calvin/decoder/checkpoint-5000"},
    "vlm_only": {},
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Verify2Act real-robot server (lab PC side)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--jetson-ip", default="192.168.0.8")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--offline-image", default=None, help="skip rosbridge: plan once on this image and exit")
    ap.add_argument("--goal", default="Put the blue block and the yellow block into the bin",
                    help="goal for --offline-image")
    ap.add_argument("--output-dir", default=None,
                    help="per-session imagination logs (default: verify2act/output/real/<wm-mode>)")

    ap.add_argument("--wm-mode", choices=list(MODE_DEFAULTS), default="v2a_wm",
                    help="verification variant (one per server process); diffusion_wm is the sim's 'diffusion' mode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--critic-ckpt", default="verify2act/output/contrastive/calvin/best_contrastive_critic.pt")
    ap.add_argument("--latent-wm-ckpt", default=None, help="v2a_wm / rla_wm dynamics (default: per --wm-mode)")
    ap.add_argument("--encoder-ckpt", default=f"{_V2A_CALVIN}/encoder/ckpt/delta_encoder_best.pt",
                    help="v2a_wm / rla_wm delta encoder (the sim's rla_wm run uses the v2a_wm one too)")
    ap.add_argument("--wm-decoder-dir", default=None,
                    help="FeatureDecoder dir (v2a_wm / rla_wm) or VAE decoder dir (diffusion_wm); default: per --wm-mode")
    ap.add_argument("--history-len", type=int, default=3)
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--num-latent-tokens", type=int, default=32)
    ap.add_argument("--action-conditioning", choices=["cross_attn", "adaln"], default="cross_attn")
    # diffusion_wm (InstructPix2Pix + LoRA), defaults as in inference_calvin.py
    ap.add_argument("--wm-model", default="timbrooks/instruct-pix2pix")
    ap.add_argument("--wm-adapter-dir", default="verify2act/output/diffusion_wm/calvin/wm/best/unet_lora")
    ap.add_argument("--vae-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--vae-subfolder", default="vae")
    ap.add_argument("--wm-steps", type=int, default=30)
    ap.add_argument("--wm-image-guidance", type=float, default=2.8)
    ap.add_argument("--wm-text-guidance", type=float, default=7.5)
    ap.add_argument("--wm-seed", type=int, default=None)

    ap.add_argument("--prompt-config", default="verify2act/configs/prompts/dofbot/planner.yaml")
    ap.add_argument("--planner-model", default="gemini-2.5-flash", help="the model the sim runs use")
    ap.add_argument("--planner-max-tokens", type=int, default=8192)
    ap.add_argument("--planner-temperature", type=float, default=0.2)
    ap.add_argument("--planner-call-delay", type=float, default=0.0)
    ap.add_argument("--gcp-project", default="verify2act",
                    help="call Gemini through Vertex AI in this GCP project (gcloud ADC login, as the sim runs do); "
                         "pass '' to use the GEMINI_API_KEY AI Studio key instead")

    ap.add_argument("--horizon", type=int, default=4, help="max subtasks per plan (request may override)")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--theta-c", type=float, default=0.5, help="temporal-head threshold (as in the sim v2a_wm runs)")
    ap.add_argument("--theta-p", type=float, default=0.05, help="goal-head threshold (as in the sim v2a_wm runs)")
    ap.add_argument("--max-retries", type=int, default=2, help="WM re-samples on a requery")
    ap.add_argument("--max-replans", type=int, default=2, help="reflect -> replan budget per planning call (sim default)")
    args = ap.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"verify2act/output/real/{args.wm_mode}"
    for key, value in MODE_DEFAULTS[args.wm_mode].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from verify2act.robot.backend import Verify2ActBackend
    backend = Verify2ActBackend.from_args(args)

    if args.offline_image:
        import numpy as np
        from PIL import Image
        img = np.asarray(Image.open(args.offline_image).convert("RGB"))
        print(json.dumps(backend.plan("offline", img, args.goal), indent=2))
        return 0

    Server(backend, args.jetson_ip, args.port).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
