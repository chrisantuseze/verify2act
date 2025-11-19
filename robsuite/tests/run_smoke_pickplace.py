from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np


def smoke_run(env_name):
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
    obs = env.reset()
    print(f"Instantiated {env_name}, detected objects:", [k for k in obs.keys() if '_pos' in k and 'robot' not in k.lower()])
    a = np.zeros(env.action_dim)
    for i in range(10):
        obs, r, done, info = env.step(a)
    env.close()


if __name__ == '__main__':
    smoke_run('PickPlaceMulti3')
    smoke_run('PickPlaceMulti4')
