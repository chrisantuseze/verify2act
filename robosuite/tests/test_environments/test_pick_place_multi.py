import pytest
import numpy as np
from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config


def _make_env(env_name):
    controller_config = load_composite_controller_config(controller="BASIC")
    env = make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=100,
        ignore_done=True,
    )
    return env


@pytest.mark.parametrize("env_name,expected_count", [("PickPlaceMulti3", 3), ("PickPlaceMulti4", 4)])
def test_pick_place_object_count(env_name, expected_count):
    env = _make_env(env_name)
    obs = env.reset()
    # check object count matches
    assert len(env.objects) == expected_count

    # step a few times with zero actions to make sure no immediate runtime errors
    a = np.zeros(env.action_dim)
    for _ in range(5):
        obs, r, done, info = env.step(a)
    env.close()
