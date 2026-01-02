import os

# Set environment variable for macOS OpenGL rendering
# os.environ['MUJOCO_GL'] = 'osmesa'

import robosuite as suite
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for macOS compatibility
import matplotlib.pyplot as plt

from robosuite.controllers import load_composite_controller_config

# Create environment with camera observations
controller_config = load_composite_controller_config(controller="BASIC")
env = suite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    env_configuration="opposed",            # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)

print("Environment created successfully!")
print(f"Action dimension: {env.action_dim}")
print(f"Action space: {env.action_spec}")

# Reset environment
obs = env.reset()
print(f"\nObservation keys: {obs.keys()}")

# Extract image observations
agentview_img = obs["agentview_image"]
eye_in_hand_img = obs["robot0_eye_in_hand_image"]

print(f"\nAgentview image shape: {agentview_img.shape}")
print(f"Eye-in-hand image shape: {eye_in_hand_img.shape}")
print(f"Image dtype: {agentview_img.dtype}")
print(f"Image value range: [{agentview_img.min()}, {agentview_img.max()}]")

# Run a few steps and collect images
print("\nRunning 10 steps...")
images_agentview = []
images_eye_in_hand = []

for step in range(10):
    # Random action
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])
    obs, reward, done, info = env.step(action)
    
    images_agentview.append(obs["agentview_image"])
    images_eye_in_hand.append(obs["robot0_eye_in_hand_image"])
    
    if done:
        obs = env.reset()

print(f"Collected {len(images_agentview)} frames from each camera")

# Visualize the first frame from both cameras
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(images_agentview[0])
axes[0].set_title("Agentview Camera")
axes[0].axis('off')

axes[1].imshow(images_eye_in_hand[0])
axes[1].set_title("Eye-in-Hand Camera")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("robosuite_observations.png", dpi=150, bbox_inches='tight')
print("\nSaved visualization to 'robosuite_observations.png'")
plt.show()

# Clean up
env.close()
print("\nEnvironment closed.")