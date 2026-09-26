"""Kinematic MuJoCo twin of the DOFBOT block scene, seen from the arm camera at the home pose.

No arm is simulated: every frame the models see is an "arm at home, nothing held" scene, so a subtask is executed by
an oracle that teleports the block to its target pose (with placement noise), lets physics settle it, and renders.

World frame: origin at the sheet centre on the table top, x away from the robot (up in the image), y = image left, z up.
Block frame: x along the long side. Rendering needs ``MUJOCO_GL=egl`` on a headless machine.
"""

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from verify2act.twin.config import ASSETS_DIR, load_config  # noqa: E402

_C = "(red|green|blue|yellow)"
_BIN = re.compile(rf"^pick and place {_C} block into the bin$")
_ON = re.compile(rf"^pick and place {_C} block on {_C} block$")
_SIDE = re.compile(rf"^pick and place {_C} block to the (left|right) of {_C} block$")

SHEET_THICKNESS = 0.0004
_FACES = ("px", "nx", "py", "ny", "pz", "nz")                     # MuJoCo cube order: right left up down front back
_FACE_ATTRS = ("fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


class InvalidSubtask(ValueError):
    """The subtask cannot be executed in the current scene (block absent or covered, or no free spot)."""


def parse_subtask(text: str) -> Tuple[str, str, Optional[str]]:
    """Subtask string -> (kind, block, reference). kind is ``bin`` | ``on`` | ``left`` | ``right``."""
    m = _BIN.match(text)
    if m:
        return "bin", m.group(1), None
    m = _ON.match(text)
    if m:
        return "on", m.group(1), m.group(2)
    m = _SIDE.match(text)
    if m:
        return m.group(2), m.group(1), m.group(3)
    raise InvalidSubtask(f"not in the subtask vocabulary: {text!r}")


def yaw_quat(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _rect_corners(x: float, y: float, yaw_deg: float, hl: float, hw: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def rects_overlap(a: np.ndarray, b: np.ndarray, gap: float = 0.0) -> bool:
    """Separating-axis test for two convex quads (4x2 corners), with ``gap`` of required clearance."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]]) / (np.linalg.norm(edge) + 1e-12)
            pa, pb = a @ axis, b @ axis
            if pa.max() + gap <= pb.min() or pb.max() + gap <= pa.min():
                return False
    return True


def camera_frame(pos: np.ndarray, lookat: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """3x3 matrix whose columns are the camera's x (image right), y (image up) and z (backwards) axes in the world."""
    f = lookat - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        r, u = c * r + s * u, -s * r + c * u
    return np.stack([r, u, -f], axis=1)


class DofbotTwin:
    """MuJoCo scene with 4 rectangular blocks on a sheet; subtasks from the robot vocabulary; arm-camera rendering."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, assets_dir: Optional[str] = None, render: bool = True):
        self.cfg = cfg or load_config()
        self.assets_dir = Path(assets_dir) if assets_dir else ASSETS_DIR
        self.colors: List[str] = list(self.cfg["blocks"]["colors"])
        self.size = np.array(self.cfg["blocks"]["size"], dtype=float)        # L, W, H
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._qadr = {c: self.model.jnt_qposadr[self.model.joint(f"{c}_joint").id] for c in self.colors}
        self._vadr = {c: self.model.jnt_dofadr[self.model.joint(f"{c}_joint").id] for c in self.colors}
        self._body = {c: self.model.body(c).id for c in self.colors}
        self._geom = {c: self.model.geom(f"{c}_geom").id for c in self.colors}
        self._mat = {c: self.model.mat(f"mat_{c}").id for c in self.colors}
        self._mat_rgba0 = {c: self.model.mat_rgba[self._mat[c]].copy() for c in self.colors}
        self._cam = self.model.camera("home").id
        self.present: Dict[str, bool] = {c: True for c in self.colors}
        cam = self.cfg["camera"]
        self.width, self.height = int(cam["width"]), int(cam["height"])
        self.base_camera = self._camera_params(cam)
        self.episode_camera = dict(self.base_camera)
        self.renderer = mujoco.Renderer(self.model, self.height, self.width) if render else None

    # ── model ──────────────────────────────────────────────────────────────

    def _texture_xml(self) -> str:
        """Asset XML and material names; real-image textures when ``textures.py`` has built them, else flat colours."""
        a = []
        wood, sheet = self.assets_dir / "wood.png", self.assets_dir / "sheet.png"
        if wood.exists():
            a.append(f'<texture name="tex_table" type="2d" file="{wood}"/>')
            a.append('<material name="mat_table" texture="tex_table" texrepeat="6 6" texuniform="true" specular="0.05"/>')
        else:
            a.append(f'<material name="mat_table" rgba="{_fmt(self.cfg["table"]["rgba"])}" specular="0.05"/>')
        if sheet.exists():
            a.append(f'<texture name="tex_sheet" type="2d" file="{sheet}"/>')
            a.append('<material name="mat_sheet" texture="tex_sheet" specular="0.02"/>')
        else:
            a.append(f'<material name="mat_sheet" rgba="{_fmt(self.cfg["sheet"]["rgba"])}" specular="0.02"/>')
        for c in self.colors:
            faces = [self.assets_dir / f"{c}_{f}.png" for f in _FACES]
            single = self.assets_dir / f"{c}.png"
            if all(p.exists() for p in faces):
                files = " ".join(f'{attr}="{p}"' for attr, p in zip(_FACE_ATTRS, faces))
                a.append(f'<texture name="tex_{c}" type="cube" {files}/>')
                a.append(f'<material name="mat_{c}" texture="tex_{c}" specular="0.08" shininess="0.2"/>')
            elif single.exists():
                a.append(f'<texture name="tex_{c}" type="cube" file="{single}"/>')
                a.append(f'<material name="mat_{c}" texture="tex_{c}" specular="0.08" shininess="0.2"/>')
            else:
                rgba = _fmt(self.cfg["blocks"]["rgba"][c])
                a.append(f'<material name="mat_{c}" rgba="{rgba}" specular="0.08" shininess="0.2"/>')
        return "\n    ".join(a)

    def _build_xml(self) -> str:
        cfg = self.cfg
        hl, hw, hh = self.size / 2
        sx, sy = cfg["sheet"]["size"]
        ox, oy = cfg["sheet"]["offset"]
        w, h = cfg["camera"]["width"], cfg["camera"]["height"]
        mass, fr = cfg["blocks"]["mass"], cfg["blocks"]["friction"]
        assets = self._texture_xml()
        bodies = []
        for i, c in enumerate(self.colors):
            px, py, pz = self._park_pos(i)
            bodies.append(
                f'<body name="{c}" pos="{px} {py} {pz}">\n'
                f'      <freejoint name="{c}_joint"/>\n'
                f'      <geom name="{c}_geom" type="box" size="{hl} {hw} {hh}" material="mat_{c}" mass="{mass}"'
                f' friction="{fr} 0.01 0.001" condim="4"/>\n'
                f'    </body>')
        body_xml = "\n    ".join(bodies)
        return f"""<mujoco model="dofbot_twin">
  <option timestep="{cfg['physics']['timestep']}" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.15 0.15 0.15" specular="0 0 0"/>
    <map znear="0.005" zfar="20"/>
  </visual>
  <asset>
    {assets}
  </asset>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="0 0 2" dir="0.2 0.1 -1" diffuse="0.6 0.6 0.6" specular="0.05 0.05 0.05"/>
    <geom name="table" type="plane" size="3 3 0.01" material="mat_table" friction="1 0.01 0.001"/>
    <geom name="sheet" type="box" size="{sx / 2} {sy / 2} {SHEET_THICKNESS / 2}" pos="{ox} {oy} {SHEET_THICKNESS / 2}"
          material="mat_sheet" friction="1 0.01 0.001"/>
    <camera name="home" pos="0 0 1" fovy="45"/>
    {body_xml}
  </worldbody>
</mujoco>"""

    def _park_pos(self, i: int) -> Tuple[float, float, float]:
        """Out of view (behind the camera): where blocks "in the bin" wait."""
        return -1.5, -0.45 + 0.3 * i, float(self.size[2] / 2 + 0.001)

    # ── state ──────────────────────────────────────────────────────────────

    def pose(self, c: str) -> Tuple[np.ndarray, float]:
        """(xyz, yaw_deg) of a block's centre."""
        q = self.data.qpos[self._qadr[c]:self._qadr[c] + 7]
        return q[:3].copy(), quat_yaw(q[3:7])

    def _set_pose(self, c: str, xyz, yaw_deg: float) -> None:
        a = self._qadr[c]
        self.data.qpos[a:a + 3] = xyz
        self.data.qpos[a + 3:a + 7] = yaw_quat(yaw_deg)
        self.data.qvel[self._vadr[c]:self._vadr[c] + 6] = 0.0

    def get_state(self) -> Dict[str, Any]:
        out = {}
        for c in self.colors:
            xyz, yaw = self.pose(c)
            out[c] = {"present": self.present[c], "pos": [round(float(v), 5) for v in xyz], "yaw": round(yaw, 3)}
        return out

    def set_state(self, state: Dict[str, Any]) -> None:
        for i, c in enumerate(self.colors):
            s = state[c]
            self.present[c] = bool(s["present"])
            self._set_pose(c, s["pos"] if self.present[c] else self._park_pos(i), s["yaw"])
        mujoco.mj_forward(self.model, self.data)

    def settle(self, steps: Optional[int] = None) -> None:
        for _ in range(steps or self.cfg["physics"]["settle_steps"]):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ── geometry & predicates ──────────────────────────────────────────────

    def footprint(self, c: str) -> np.ndarray:
        xyz, yaw = self.pose(c)
        return _rect_corners(xyz[0], xyz[1], yaw, self.size[0] / 2, self.size[1] / 2)

    def is_tipped(self, c: str, tol_deg: float = 12.0) -> bool:
        zaxis = self.data.xmat[self._body[c]].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(min(1.0, abs(zaxis[2])))) > tol_deg

    def on(self, c: str, b: str) -> bool:
        """``c`` rests on ``b``: above it by about one block height, centre inside ``b``'s footprint."""
        if c == b or not (self.present[c] and self.present[b]):
            return False
        (pc, _), (pb, yb) = self.pose(c), self.pose(b)
        dz = pc[2] - pb[2]
        if not (0.6 * self.size[2] < dz < 1.4 * self.size[2]):
            return False
        cs, sn = math.cos(math.radians(yb)), math.sin(math.radians(yb))
        d = pc[:2] - pb[:2]
        lx, ly = cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1]
        return abs(lx) <= self.size[0] / 2 and abs(ly) <= self.size[1] / 2

    def below(self, c: str) -> Optional[str]:
        return next((b for b in self.colors if self.on(c, b)), None)

    def clear(self, c: str) -> bool:
        return self.present[c] and not any(self.on(o, c) for o in self.colors if o != c)

    def on_table(self, c: str) -> bool:
        return self.present[c] and self.pose(c)[0][2] < SHEET_THICKNESS + 0.75 * self.size[2]

    def side_of(self, c: str, b: str, side: str) -> bool:
        """``c`` is ``side`` (left = +y = image left) of ``b``: 0.6–3.5 block widths away along y, level, ahead/behind
        by less than a block length (EVAL_TASKS.md)."""
        if c == b or not (self.present[c] and self.present[b]):
            return False
        (pc, _), (pb, _) = self.pose(c), self.pose(b)
        dy = (pc[1] - pb[1]) * (1 if side == "left" else -1)
        w = self.size[1]
        return (0.6 * w <= dy <= 3.5 * w and abs(pc[0] - pb[0]) < self.size[0]
                and abs(pc[2] - pb[2]) < self.size[2])

    def project(self, pts: np.ndarray, cam: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """World points (N,3) -> pixel (N,2) with the given (default: this episode's) camera. NaN behind the camera."""
        cam = cam or self.episode_camera
        R = camera_frame(np.asarray(cam["pos"]), np.asarray(cam["lookat"]), cam.get("roll", 0.0))
        pc = (np.asarray(pts) - np.asarray(cam["pos"])) @ R
        f = (self.height / 2) / math.tan(math.radians(cam["fovy"]) / 2)
        depth = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.width / 2 + f * pc[:, 0] / depth
            v = self.height / 2 - f * pc[:, 1] / depth
        uv = np.stack([u, v], axis=1)
        uv[depth <= 0] = np.nan
        return uv

    def block_corners(self, xyz, yaw_deg: float) -> np.ndarray:
        fp = _rect_corners(xyz[0], xyz[1], yaw_deg, self.size[0] / 2, self.size[1] / 2)
        hh = self.size[2] / 2
        return np.concatenate([np.c_[fp, np.full(4, xyz[2] - hh)], np.c_[fp, np.full(4, xyz[2] + hh)]])

    def in_view(self, xyz, yaw_deg: float, margin: Optional[float] = None) -> bool:
        m = self.cfg["placement"]["image_margin_px"] if margin is None else margin
        uv = self.project(self.block_corners(xyz, yaw_deg))
        if np.isnan(uv).any():
            return False
        return bool((uv[:, 0] >= m).all() and (uv[:, 0] <= self.width - m).all()
                    and (uv[:, 1] >= m).all() and (uv[:, 1] <= self.height - m).all())

    def _free_spot(self, c: str, x: float, y: float, yaw: float) -> bool:
        """Footprint of ``c`` at (x, y, yaw) on the table: no overlap with other table blocks, fully in view."""
        fp = _rect_corners(x, y, yaw, self.size[0] / 2, self.size[1] / 2)
        gap = self.cfg["placement"]["min_gap"]
        for o in self.colors:
            if o != c and self.present[o] and rects_overlap(fp, self.footprint(o), gap):
                return False
        return self.in_view((x, y, SHEET_THICKNESS + self.size[2] / 2), yaw)

    # ── episode setup ──────────────────────────────────────────────────────

    def reset(self, rng: np.random.Generator, blocks: Optional[List[str]] = None, max_tries: int = 200) -> None:
        """Random non-overlapping layout of ``blocks`` (default: all) on the sheet, all in view; the rest in the bin."""
        blocks = list(self.colors if blocks is None else blocks)
        pl = self.cfg["placement"]
        sx, sy = self.cfg["sheet"]["size"]
        ox, oy = self.cfg["sheet"]["offset"]
        hl, hw = self.size[0] / 2, self.size[1] / 2
        for _attempt in range(20):
            for i, c in enumerate(self.colors):
                self.present[c] = False
                self._set_pose(c, self._park_pos(i), 0.0)
            mujoco.mj_forward(self.model, self.data)
            ok = True
            for c in rng.permutation(blocks):
                for _ in range(max_tries):
                    yaw = self._flip(float(np.clip(rng.normal(0, pl["yaw_std"]), -pl["yaw_max"], pl["yaw_max"])), rng)
                    fp0 = _rect_corners(0, 0, yaw, hl, hw)
                    ex, ey = np.abs(fp0).max(axis=0) + pl["region_margin"]
                    if ex * 2 >= sx or ey * 2 >= sy:
                        continue
                    x = rng.uniform(ox - sx / 2 + ex, ox + sx / 2 - ex)
                    y = rng.uniform(oy - sy / 2 + ey, oy + sy / 2 - ey)
                    if self._free_spot(c, x, y, yaw):
                        self.present[c] = True
                        self._set_pose(c, (x, y, SHEET_THICKNESS + self.size[2] / 2 + 0.0005), yaw)
                        mujoco.mj_forward(self.model, self.data)
                        break
                else:
                    ok = False
                    break
            if ok:
                self.settle()
                return
        raise RuntimeError(f"could not place {blocks} on the sheet; check the sheet size and camera")

    # ── subtasks ───────────────────────────────────────────────────────────

    def check(self, subtask: str) -> Tuple[str, str, Optional[str]]:
        """Parse and validate; raise ``InvalidSubtask`` when the scene does not allow it."""
        kind, c, b = parse_subtask(subtask)
        if b == c:
            raise InvalidSubtask("block and reference block are the same")
        if not self.present[c]:
            raise InvalidSubtask(f"{c} block is not on the table")
        if not self.clear(c):
            raise InvalidSubtask(f"something is on the {c} block")
        if b is not None:
            if not self.present[b]:
                raise InvalidSubtask(f"{b} block is not on the table")
            if kind == "on" and not self.clear(b):
                raise InvalidSubtask(f"something is on the {b} block")
        return kind, c, b

    def apply(self, subtask: str, rng: np.random.Generator, max_tries: int = 60) -> None:
        """Execute one subtask (teleport + settle). Raises ``InvalidSubtask``; the scene is unchanged in that case."""
        kind, c, b = self.check(subtask)
        pl = self.cfg["placement"]
        fail = rng.random() < pl["p_fail"]
        snapshot = self.get_state()
        if kind == "bin":
            self.present[c] = False
            self._set_pose(c, self._park_pos(self.colors.index(c)), 0.0)
            mujoco.mj_forward(self.model, self.data)
            return
        pb, yb = self.pose(b)
        if kind == "on":
            d = rng.normal(0, pl["stack_xy_std"], 2)
            if fail:
                d += _rand_dir(rng) * rng.uniform(*pl["fail_offset"])
            yaw = self._flip(yb + rng.normal(0, pl["stack_yaw_std"]), rng)
            self._set_pose(c, (pb[0] + d[0], pb[1] + d[1], pb[2] + self.size[2] + 0.001), yaw)
            self.settle()
            if not fail and (not self.on(c, b) or self.is_tipped(c)):
                self.set_state(snapshot)
                raise InvalidSubtask(f"{c} did not settle on {b}")
            return
        spot = self.find_side_spot(c, b, kind, rng, max_tries, fail)
        if spot is not None:
            x, y, yaw = spot
            self._set_pose(c, (x, y, SHEET_THICKNESS + self.size[2] / 2 + 0.0005), yaw)
            self.settle()
            if not self.is_tipped(c):
                return
        self.set_state(snapshot)
        raise InvalidSubtask(f"no free spot {kind} of the {b} block for the {c} block")

    def find_side_spot(self, c: str, b: str, side: str, rng: np.random.Generator, max_tries: int = 60,
                       fail: bool = False) -> Optional[Tuple[float, float, float]]:
        """A free, in-view table pose (x, y, yaw) for ``c`` on ``side`` of ``b``, or None."""
        pl = self.cfg["placement"]
        pb, yb = self.pose(b)
        sign = 1.0 if side == "left" else -1.0
        for _ in range(max_tries):
            dy = rng.uniform(*pl["side_distance"])
            dx = rng.normal(0, pl["side_x_std"])
            if fail:
                dy -= rng.uniform(*pl["fail_offset"])
            axis = (yb + 90.0) % 180.0 - 90.0                      # the reference's long axis, mod 180
            yaw = self._flip(float(np.clip(axis + rng.normal(0, pl["side_yaw_std"]), -pl["yaw_max"], pl["yaw_max"])),
                             rng)
            x, y = pb[0] + dx, pb[1] + sign * dy
            # The moving block leaves its old spot first (it may be the one next to the reference).
            was = self.present[c]
            self.present[c] = False
            free = self._free_spot(c, x, y, yaw)
            self.present[c] = was
            if free:
                return x, y, yaw
        return None

    def _flip(self, yaw: float, rng: np.random.Generator) -> float:
        """The block is symmetric but its tag is on one end (-x): turn it so the tag end faces the camera (the robot)
        with probability ``placement.p_tag_toward_camera``, else away."""
        yaw = (yaw + 90.0) % 180.0 - 90.0                         # tag end toward the camera
        return yaw if rng.random() < self.cfg["placement"].get("p_tag_toward_camera", 0.5) else yaw + 180.0

    def valid_subtasks(self) -> List[str]:
        """Every subtask ``check`` accepts in the current scene (left/right may still fail for lack of space)."""
        out = []
        for c in self.colors:
            if not self.clear(c):
                continue
            out.append(f"pick and place {c} block into the bin")
            for b in self.colors:
                if b == c or not self.present[b]:
                    continue
                if self.clear(b):
                    out.append(f"pick and place {c} block on {b} block")
                out.append(f"pick and place {c} block to the left of {b} block")
                out.append(f"pick and place {c} block to the right of {b} block")
        return out

    # ── rendering ──────────────────────────────────────────────────────────

    def _camera_params(self, cam: Dict[str, Any]) -> Dict[str, Any]:
        fovy = float(cam["fovy"])
        intr = cam.get("intrinsics")
        if intr:
            fovy = math.degrees(2 * math.atan(self.height / 2 / float(intr["fy"])))
        return {"pos": np.array(cam["pos"], float), "lookat": np.array(cam["lookat"], float),
                "fovy": fovy, "roll": float(cam.get("roll", 0.0))}

    @staticmethod
    def _jitter(cam: Dict[str, Any], j: Dict[str, float], rng: np.random.Generator) -> Dict[str, Any]:
        return {"pos": cam["pos"] + rng.normal(0, j["pos"], 3), "lookat": cam["lookat"] + rng.normal(0, j["lookat"], 3),
                "fovy": cam["fovy"] + rng.normal(0, j["fovy"]) if j["fovy"] else cam["fovy"],
                "roll": cam["roll"] + rng.normal(0, j["roll"])}

    def randomize_episode(self, rng: np.random.Generator) -> None:
        """Per-episode camera, lighting and block tints. Call before ``reset``: layouts are checked against the
        episode camera's view."""
        self.sample_camera(rng)
        self.randomize_appearance(rng)

    def sample_camera(self, rng: np.random.Generator) -> None:
        """One of the home poses fitted to real frames (``camera.poses``; the sheet is not always in the same place
        relative to the robot), or the base pose, plus the per-episode jitter."""
        cam = self.cfg["camera"]
        poses = cam.get("poses") or []
        base = self.base_camera
        if poses:
            p = poses[int(rng.integers(len(poses)))]
            base = {"pos": np.array(p["pos"], float), "lookat": np.array(p["lookat"], float),
                    "fovy": self.base_camera["fovy"], "roll": float(p.get("roll", 0.0))}
        self.episode_camera = self._jitter(base, cam["jitter_episode"], rng)

    def randomize_appearance(self, rng: np.random.Generator) -> None:
        lr = self.cfg["render"]["lights"]
        el, az = math.radians(rng.uniform(*lr["dir_elevation"])), math.radians(rng.uniform(*lr["dir_azimuth"]))
        light = self.model.light("sun").id
        self.model.light_dir[light] = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)]
        self.model.light_diffuse[light] = rng.uniform(*lr["diffuse"]) * rng.uniform(0.95, 1.05, 3)
        self.model.light_castshadow[light] = bool(lr["shadows"])
        self.model.vis.headlight.ambient[:] = rng.uniform(*lr["ambient"]) * rng.uniform(0.97, 1.03, 3)
        for c in self.colors:
            tint = self._mat_rgba0[c].copy()
            tint[:3] = np.clip(tint[:3] * rng.uniform(0.93, 1.07, 3), 0, 1)
            self.model.mat_rgba[self._mat[c]] = tint

    def render(self, rng: Optional[np.random.Generator] = None, cam: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the arm camera; per-frame jitter on top of the episode camera when ``rng`` is given."""
        if self.renderer is None:
            raise RuntimeError("DofbotTwin(render=False)")
        cam = cam or self.episode_camera
        if rng is not None:
            cam = self._jitter(cam, self.cfg["camera"]["jitter_frame"], rng)
        R = camera_frame(np.asarray(cam["pos"]), np.asarray(cam["lookat"]), cam["roll"])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        self.model.cam_pos[self._cam] = cam["pos"]
        self.model.cam_quat[self._cam] = q
        self.model.cam_fovy[self._cam] = cam["fovy"]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="home")
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _fmt(v) -> str:
    return " ".join(str(x) for x in v)


def _rand_dir(rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0, 2 * math.pi)
    return np.array([math.cos(a), math.sin(a)])
