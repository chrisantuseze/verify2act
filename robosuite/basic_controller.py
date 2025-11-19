from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np

def run_heuristic_policy(env_name="PickPlaceMulti3"):
    """
    Runs a simple state-machine-based heuristic policy for PickPlaceMulti environments.
    Works with both PickPlaceMulti3 and PickPlaceMulti4.
    
    Args:
        env_name (str): Either "PickPlaceMulti3" or "PickPlaceMulti4"
    """

    # --- 1. Create the Environment ---
    # controller_config = C.load_controller_config(default_controller="IK_POSE")

    controller_config = load_composite_controller_config(controller="BASIC")

    env = make(
        env_name="PickPlaceMulti3",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,            # <-- THIS IS STILL NEEDED (for the can)
        control_freq=20,
        horizon=2000,
        ignore_done=True,
    )

    # --- 2. Initialize Policy State ---
    obs = env.reset()

    # Get object names from environment (either 3 or 4 objects)
    object_names = [name for name in obs.keys() if "_pos" in name and "robot" not in name.lower()]
    print(f"\nDetected {len(object_names)} objects to place: {object_names}")
    
    # Keep track of placed objects and current target
    objects_to_place = object_names.copy()
    current_object = objects_to_place[0]  # Start with first object
    
    # The bin positions and target placements are static properties of the environment
    bin1_pos = env.bin1_pos
    bin2_pos = env.bin2_pos
    target_placements = env.target_bin_placements  # Each object's target position in bin2
    
    # Create mapping from object name to target position
    object_to_target = {}
    for obj_name, target_pos in zip(object_names, target_placements):
        # Add a small offset to help with reachability
        adjusted_pos = target_pos.copy()
        adjusted_pos[2] = bin2_pos[2]  # Use bin2's z-height
        object_to_target[obj_name] = target_pos
        print(f"Target position for {obj_name}: {adjusted_pos}")

    # Get z-heights for our policy
    obj_z_offset = 0.02
    bin_z_offset = 0.05

    max_bin_height = max(env.bin1_pos[2], env.bin2_pos[2])
    safe_z_height = max_bin_height + 0.25
    print(f"Using safe Z height: {safe_z_height}")

    P_GAIN = 10.0  # Proportional gain

    # State machine
    stage = "move_to_object"
    grasp_counter = 0
    release_counter = 0

    print("\nStarting heuristic policy for multi-object pick and place...")
    print(f"Source bin at: {bin1_pos}")
    print(f"Target bin at: {bin2_pos}")
    print(f"Currently targeting: {current_object}")

    # --- 3. Run the Policy Loop ---
    while True:
        try:
            # Get current state from observations
            eef_pos = obs['robot0_eef_pos']
            obj_pos = obs[current_object]

            action = np.zeros(env.action_dim)
            
            # --- State Machine Logic (using static bin_pos) ---

            if stage == "move_to_object":
                target_pos = obj_pos + np.array([0, 0, safe_z_height - obj_pos[2]])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = -1
                
                if np.linalg.norm(error) < 0.01:
                    print(f"Stage: {stage} -> lower_to_object")
                    stage = "lower_to_object"

            elif stage == "lower_to_object":
                target_pos = obj_pos + np.array([0, 0, obj_z_offset])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = -1
                
                if np.linalg.norm(error) < 0.005:
                    print(f"Stage: {stage} -> grasp")
                    stage = "grasp"

            elif stage == "grasp":
                action[:3] = 0
                action[6] = 1
                
                grasp_counter += 1
                if grasp_counter > 50:
                    print(f"Stage: {stage} -> lift_object")
                    stage = "lift_object"
                    grasp_counter = 0

            elif stage == "lift_object":
                target_pos = [eef_pos[0], eef_pos[1], safe_z_height]
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = 1
                
                if np.linalg.norm(error) < 0.01:
                    print(f"Stage: {stage} -> move_to_bin")
                    stage = "move_to_bin"
                    
            elif stage == "move_to_bin":
                # Get target position for current object from our mapping
                target_placement = object_to_target[current_object]
                
                # Move above the target position
                target_pos = target_placement + np.array([0, 0, safe_z_height - target_placement[2]])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = 1

                if np.linalg.norm(error[:2]) < 0.01:
                    print(f"Stage: {stage} -> lower_to_bin")
                    stage = "lower_to_bin"
            
            elif stage == "lower_to_bin":
                # Get target position for current object
                target_placement = object_to_target[current_object]
                
                # Lower to the target position
                target_pos = target_placement + np.array([0, 0, bin_z_offset])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = 1

                # print(f"Target pos in lower_to_bin: {target_pos}, EEF pos: {eef_pos}, Error: {np.linalg.norm(error)}")
                if np.linalg.norm(error) < 0.025:
                    print(f"Stage: {stage} -> release")
                    stage = "release"
            
            elif stage == "release":
                action[:3] = 0
                action[6] = -1
                
                release_counter += 1
                if release_counter > 50:
                    print(f"Stage: {stage} -> retract")
                    stage = "retract"
                    release_counter = 0

            elif stage == "retract":
                # Set retract position once when entering this stage
                if 'retract_target' not in locals():
                    retract_target = eef_pos + np.array([0, 0, 0.1])
                    print(f"Setting retract target to: {retract_target}")
                
                error = retract_target - eef_pos
                action[:3] = error * P_GAIN
                action[6] = -1

                # print(f"Retract target: {retract_target}, EEF pos: {eef_pos}, Error: {np.linalg.norm(error)}")

                if np.linalg.norm(error) < 0.01:
                    # Clear the retract target for next time
                    if 'retract_target' in locals():
                        del retract_target
                    
                    # Remove the placed object from our list
                    if current_object in objects_to_place:
                        objects_to_place.remove(current_object)
                    
                    # If we have more objects to place, continue with the next one
                    if objects_to_place:
                        current_object = objects_to_place[0]
                        print(f"\n--- Moving to next object: {current_object} ---")
                        stage = "move_to_object"
                    else:
                        print("\n--- All objects placed! Resetting episode. ---")
                        stage = "done"

            elif stage == "done":
                obs = env.reset()
                # Reset our object tracking
                objects_to_place = object_names.copy()
                current_object = objects_to_place[0]
                stage = "move_to_object"
                print(f"\nReset complete. Starting with: {current_object}")
            
            # --- End of Policy ---
            
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                print("--- ENVIRONMENT REPORTED TASK SUCCESS! ---")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PickPlaceMulti3', 
                      choices=['PickPlaceMulti3', 'PickPlaceMulti4'],
                      help='Which environment to run (3 or 4 objects)')
    args = parser.parse_args()
    
    run_heuristic_policy(env_name=args.env)