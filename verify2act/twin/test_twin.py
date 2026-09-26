"""Offline tests for the DOFBOT twin: python -m pytest verify2act/twin/test_twin.py -q"""

import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from verify2act.robot.prompts import is_valid_subtask  # noqa: E402
from verify2act.twin import generate  # noqa: E402
from verify2act.twin.calibrate import board_points, home_pose  # noqa: E402
from verify2act.twin.scene import (SHEET_THICKNESS, DofbotTwin, InvalidSubtask, camera_frame,  # noqa: E402
                                   parse_subtask, rects_overlap)
from verify2act.twin.tasks import (EVAL_TASKS, bin_goal, eval_goal, run_goal, sample_goal, side_goal,  # noqa: E402
                                   stack_goal)


@pytest.fixture(scope="module")
def twin():
    t = DofbotTwin()
    yield t
    t.close()


def _place(t, poses):
    for i, c in enumerate(t.colors):
        t.present[c] = c in poses
        x, y, yaw = poses.get(c, (*t._park_pos(i)[:2], 0.0))
        z = SHEET_THICKNESS + t.size[2] / 2 if c in poses else t._park_pos(i)[2]
        t._set_pose(c, (x, y, z), yaw)
    t.settle(50)


def test_parse_subtask():
    assert parse_subtask("pick and place red block into the bin") == ("bin", "red", None)
    assert parse_subtask("pick and place red block on blue block") == ("on", "red", "blue")
    assert parse_subtask("pick and place red block to the left of blue block") == ("left", "red", "blue")
    assert parse_subtask("pick and place red block to the right of blue block") == ("right", "red", "blue")
    with pytest.raises(InvalidSubtask):
        parse_subtask("pick red block")


def test_rects_overlap():
    a = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    assert rects_overlap(a, a + [1.5, 0])
    assert not rects_overlap(a, a + [2.5, 0])
    assert rects_overlap(a, a + [2.5, 0], gap=0.6)


def test_blocks_are_rectangular(twin):
    L, W, H = twin.size
    assert L > W, "long side along the block's x"


def test_reset_layout_in_view_and_apart(twin):
    rng = np.random.default_rng(0)
    for _ in range(5):
        twin.reset(rng)
        on = [c for c in twin.colors if twin.present[c]]
        assert len(on) == 4
        for i, a in enumerate(on):
            xyz, yaw = twin.pose(a)
            assert twin.in_view(xyz, yaw, margin=0)
            assert twin.on_table(a) and not twin.is_tipped(a)
            for b in on[i + 1:]:
                assert not rects_overlap(twin.footprint(a), twin.footprint(b))


def test_subtasks_and_predicates(twin):
    rng = np.random.default_rng(1)
    _place(twin, {"red": (-0.03, 0.05, 0), "blue": (-0.03, -0.05, 0), "yellow": (0.07, 0.0, 0), "green": (0.07, 0.07, 0)})
    twin.apply("pick and place red block on blue block", rng)
    assert twin.on("red", "blue") and not twin.is_tipped("red")
    assert not twin.clear("blue") and twin.below("red") == "blue"
    with pytest.raises(InvalidSubtask):
        twin.check("pick and place blue block into the bin")       # red sits on it
    with pytest.raises(InvalidSubtask):
        twin.check("pick and place green block on blue block")     # blue is covered
    twin.apply("pick and place green block to the right of yellow block", rng)
    assert twin.side_of("green", "yellow", "right") and not twin.side_of("green", "yellow", "left")
    twin.apply("pick and place yellow block into the bin", rng)
    assert not twin.present["yellow"]
    xyz, yaw = twin.pose("yellow")
    assert not twin.in_view(xyz, yaw), "a block in the bin is out of view"
    with pytest.raises(InvalidSubtask):
        twin.check("pick and place yellow block into the bin")


def test_left_is_image_left(twin):
    """World +y must be image left: that is what the robot's 'left of' means in the camera frame."""
    uv = twin.project(np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]]))
    assert uv[0, 0] < uv[1, 0]
    uv = twin.project(np.array([[0.08, 0.0, 0.0], [-0.05, 0.0, 0.0]]))
    assert uv[0, 1] < uv[1, 1], "x away from the robot is up in the image"


def test_invalid_apply_leaves_scene_unchanged(twin):
    rng = np.random.default_rng(2)
    _place(twin, {"red": (0.0, 0.0, 0), "blue": (0.06, 0.0, 0)})
    before = twin.get_state()
    with pytest.raises(InvalidSubtask):
        twin.apply("pick and place green block on blue block", rng)
    assert twin.get_state() == before


@pytest.mark.parametrize("maker", [bin_goal, side_goal, stack_goal])
def test_goal_families_reach_their_goal(twin, maker):
    rng = np.random.default_rng(3)
    reached = 0
    for _ in range(12):
        twin.reset(rng)
        g = maker(twin, rng)
        if g is None or g.check(twin) or not g.plan(twin, rng):
            continue
        try:
            steps = run_goal(twin, g, rng)
        except InvalidSubtask:                                 # no room beside the reference: the generator skips it
            continue
        assert all(is_valid_subtask(s) for s in steps)
        assert g.check(twin)
        reached += 1
    assert reached >= 6


def test_bin_goal_respects_keep_blocks(twin):
    rng = np.random.default_rng(4)
    for _ in range(20):
        twin.reset(rng)
        g = bin_goal(twin, rng)
        if g is None or not g.plan(twin, rng):
            continue
        present = {c for c in twin.colors if twin.present[c]}
        run_goal(twin, g, rng)                                  # bin subtasks always have room
        kept = {c for c in twin.colors if twin.present[c]}
        assert kept <= present
        if "except" in g.text or "leave" in g.text or "keep" in g.text:
            assert kept, g.text


def test_sample_goal_any_family(twin):
    rng = np.random.default_rng(5)
    twin.reset(rng)
    g = sample_goal(twin, rng)
    assert g is not None and g.text and g.plan(twin, rng)


def test_render_shape(twin):
    rng = np.random.default_rng(6)
    twin.reset(rng)
    twin.randomize_episode(rng)
    img = twin.render(rng)
    assert img.shape == (twin.height, twin.width, 3) and img.dtype == np.uint8 and img.std() > 5


def test_calibration_roundtrip():
    """Synthetic board image from a known twin camera -> solvePnP -> home_pose recovers the camera."""
    pos, look, roll = np.array([-0.139, 0.0, 0.363]), np.array([0.03, 0.0, 0.0]), 1.5
    Rw = camera_frame(pos, look, roll)                       # columns: right, up, back
    R_cv = np.stack([Rw[:, 0], -Rw[:, 1], -Rw[:, 2]])       # world -> OpenCV camera
    yaw = math.radians(30)                                    # board rotated on the table: must not matter
    Rb = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R = R_cv @ Rb
    t = -R_cv @ pos
    K = np.array([[780.0, 0, 320], [0, 780.0, 240], [0, 0, 1]])
    obj = board_points((9, 6), 0.024)
    img, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, None)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None)
    assert ok
    cam = home_pose(cv2.Rodrigues(rvec)[0], tvec)
    assert np.allclose(cam["pos"], pos, atol=2e-3)
    assert np.allclose(cam["lookat"], look, atol=5e-3)
    assert abs(cam["roll"] - roll) < 0.1


def test_generate_writes_loadable_dataset(tmp_path, twin):
    rng = np.random.default_rng(7)
    rows = []
    idx = 0
    while idx < 4:
        ep = generate.make_episode(twin, rng)
        if ep is None:
            continue
        ep["success"], ep["lang_goal"] = True, ep["lang_goal"] or "Clear the table"
        rows.extend(generate.write_episode(tmp_path, f"ep_{idx:06d}", ep))
        idx += 1
    with open(tmp_path / "transitions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        assert is_valid_subtask(r["action_text"])
        assert (tmp_path / r["image_t"]).exists() and (tmp_path / r["image_t1"]).exists()
        assert (tmp_path / r["goal_image"]).exists()

    from verify2act.data_loader import build_dofbot_contrastive_datasets
    tr, va = build_dofbot_contrastive_datasets(str(tmp_path), val_frac=0.25, image_size=64)
    for ds in (tr, va):
        m0, m1 = ds._sample_mode0(), ds._sample_mode1()
        assert m0["anchor"].shape == (3, 64, 64) and m0["lang_goal"]
        assert m1["positive"].shape == (3, 64, 64)


@pytest.mark.parametrize("task", sorted(EVAL_TASKS))
def test_eval_tasks_verbatim_and_reachable(twin, task):
    rng = np.random.default_rng(8)
    for _ in range(15):
        twin.randomize_episode(rng)
        twin.reset(rng)                                          # eval scenes: all four blocks
        g = eval_goal(twin, rng, task)
        if g is None or g.check(twin):
            continue
        steps = run_goal(twin, g, rng)
        assert g.check(twin) and g.task == task and g.family == "eval"
        assert all(is_valid_subtask(s) for s in steps)
        if task.startswith("task1"):
            assert len(steps) == len(g.moves)                     # one bin subtask per named block
        else:
            assert len(steps) == 1
        return
    pytest.fail(f"{task} never reachable")


def test_eval_task_texts_match_eval_doc():
    texts = {k: f().text for k, f in EVAL_TASKS.items()}
    assert texts["task1a"] == "Put the blue block and the yellow block into the bin"
    assert texts["task1d"] == ("Put the red block, the green block and the blue block into the bin except the yellow "
                               "block")
    assert texts["task3a"] == "Stack the blue block on top of the yellow block"


def test_tag_toward_camera_rate(twin):
    """Tag end (-x) toward the camera (-x world) with probability p_tag_toward_camera."""
    rng = np.random.default_rng(9)
    toward = total = 0
    for _ in range(60):
        twin.reset(rng)
        for c in twin.colors:
            if twin.present[c]:
                _, yaw = twin.pose(c)
                toward += abs((yaw + 180) % 360 - 180) < 90
                total += 1
    p = twin.cfg["placement"]["p_tag_toward_camera"]
    assert abs(toward / total - p) < 0.08
