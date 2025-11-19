from robosuite.environments.base import make
from robosuite.controllers import load_composite_controller_config
import numpy as np

def run_heuristic_policy(env_name="PickPlaceMulti3"):
    """
    Runs a simple state-machine-based heuristic policy for PickPlaceMulti environments.
    Handles both top-down and side grasping for different object types.
    Includes grasp verification and retry logic.
    
    Args:
        env_name (str): Either "PickPlaceMulti3" or "PickPlaceMulti4"
    """

    # --- 1. Create the Environment ---
    controller_config = load_composite_controller_config(controller="BASIC")

    env = make(
        env_name="PickPlaceMulti4",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
    )

    # --- 2. Initialize Policy State ---
    obs = env.reset()

    # Get object names from environment
    object_names = [name for name in obs.keys() if "_pos" in name and "robot" not in name.lower()]
    print(f"\nDetected {len(object_names)} objects to place: {object_names}")
    
    # Define which objects need side grasping (tall/thin objects like milk cartons)
    SIDE_GRASP_OBJECTS = ["milk_pos", "cereal_pos"]  # Add object names that need side grasping
    
    # Determine grasp type for each object
    object_grasp_type = {}
    for obj_name in object_names:
        # Check if object needs side grasp
        needs_side_grasp = any(keyword in obj_name.lower() for keyword in ["cereal", "bottle"])
        object_grasp_type[obj_name] = "side" if needs_side_grasp else "top"
        print(f"Object {obj_name}: {object_grasp_type[obj_name]} grasp")
    
    # Keep track of placed objects and current target
    objects_to_place = object_names.copy()
    current_object = objects_to_place[0]
    current_grasp_type = object_grasp_type[current_object]
    
    # The bin positions and target placements
    bin1_pos = env.bin1_pos
    bin2_pos = env.bin2_pos
    target_placements = env.target_bin_placements
    
    # Create mapping from object name to target position
    object_to_target = {}
    for obj_name, target_pos in zip(object_names, target_placements):
        adjusted_pos = target_pos.copy()
        adjusted_pos[2] = bin2_pos[2]
        object_to_target[obj_name] = target_pos
        print(f"Target position for {obj_name}: {adjusted_pos}")

    # Get z-heights for our policy
    obj_z_offset_top = 0.01 #0.02  # For top-down grasping
    obj_z_offset_side = 0 #0.05  # For side grasping (higher to grasp at center of tall objects)
    bin_z_offset = 0.05

    max_bin_height = max(env.bin1_pos[2], env.bin2_pos[2])
    safe_z_height = max_bin_height + 0.25
    print(f"Using safe Z height: {safe_z_height}")

    P_GAIN = 10.0  # Proportional gain for position
    R_GAIN = 5.0   # Proportional gain for orientation

    # State machine
    stage = "move_to_object"
    grasp_counter = 0
    release_counter = 0
    align_counter = 0
    
    # Grasp verification variables
    grasp_attempts = 0
    max_grasp_attempts = 3
    pre_grasp_obj_pos = None
    grasp_height_threshold = 0.05  # Object should lift at least 5cm to be considered grasped

    print("\nStarting heuristic policy for multi-object pick and place...")
    print(f"Source bin at: {bin1_pos}")
    print(f"Target bin at: {bin2_pos}")
    print(f"Currently targeting: {current_object} ({current_grasp_type} grasp)")

    # --- 3. Run the Policy Loop ---
    while True:
        try:
            # Get current state from observations
            eef_pos = obs['robot0_eef_pos']
            obj_pos = obs[current_object]

            action = np.zeros(env.action_dim)
            
            # --- State Machine Logic ---

            if stage == "move_to_object":
                # Move above object first
                target_pos = obj_pos + np.array([0, 0, safe_z_height - obj_pos[2]])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[6] = -1  # Open gripper

                if np.linalg.norm(error) < 0.01:
                    if current_grasp_type == "side":
                        print(f"Stage: {stage} -> align_for_side_grasp")
                        stage = "align_for_side_grasp"
                    else:
                        print(f"Stage: {stage} -> lower_to_object")
                        stage = "lower_to_object"

            elif stage == "align_for_side_grasp":
                # Move to side of object and rotate gripper to horizontal
                # Position slightly to the side of the object
                side_offset = np.array([0.08, 0, 0])  # Offset in x-direction
                target_pos = obj_pos + side_offset + np.array([0, 0, obj_z_offset_side])
                
                pos_error = target_pos - eef_pos
                action[:3] = pos_error * P_GAIN
                
                # Rotate gripper 90 degrees around y-axis to point horizontally
                if align_counter < 30:  # Apply rotation for first 30 steps
                    action[3:6] = np.array([0, 0.1, 0])  # Rotate around world y-axis
                else:
                    action[3:6] = 0  # Stop rotating
                
                action[6] = -1  # Open gripper
                
                align_counter += 1
                
                if np.linalg.norm(pos_error) < 0.01 and align_counter > 30:
                    print(f"Stage: {stage} -> approach_from_side")
                    stage = "approach_from_side"
                    align_counter = 0

            elif stage == "approach_from_side":
                # Move gripper toward object from the side
                target_pos = obj_pos + np.array([0, 0, obj_z_offset_side])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain orientation
                action[6] = -1  # Open gripper
                
                if np.linalg.norm(error) < 0.01:
                    # Record object position before grasping for verification
                    pre_grasp_obj_pos = obj_pos.copy()
                    print(f"Stage: {stage} -> grasp (attempt {grasp_attempts + 1}/{max_grasp_attempts})")
                    stage = "grasp"

            elif stage == "lower_to_object":
                # Standard top-down approach
                target_pos = obj_pos + np.array([0, 0, obj_z_offset_top])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Keep orientation stable
                action[6] = -1  # Open gripper
                
                if np.linalg.norm(error) < 0.005:
                    # Record object position before grasping for verification
                    pre_grasp_obj_pos = obj_pos.copy()
                    print(f"Stage: {stage} -> grasp (attempt {grasp_attempts + 1}/{max_grasp_attempts})")
                    stage = "grasp"

            elif stage == "grasp":
                action[:3] = 0
                action[3:6] = 0
                action[6] = 1  # Close gripper
                
                grasp_counter += 1
                if grasp_counter > 50:
                    print(f"Stage: {stage} -> verify_grasp")
                    stage = "verify_grasp"
                    grasp_counter = 0

            elif stage == "verify_grasp":
                # Lift slightly and check if object moved with gripper
                target_pos = [eef_pos[0], eef_pos[1], eef_pos[2] + 0.1]
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain orientation
                action[6] = 1  # Keep gripper closed
                
                # Check if we've lifted enough to verify
                if eef_pos[2] > pre_grasp_obj_pos[2] + 0.08:
                    # Check if object lifted with gripper
                    obj_height_change = obj_pos[2] - pre_grasp_obj_pos[2]
                    
                    if obj_height_change > grasp_height_threshold:
                        # Successful grasp!
                        print(f"✓ Grasp successful! Object lifted {obj_height_change:.3f}m")
                        grasp_attempts = 0  # Reset attempts counter
                        print(f"Stage: {stage} -> lift_object")
                        stage = "lift_object"
                    else:
                        # Failed grasp
                        grasp_attempts += 1
                        print(f"✗ Grasp failed! Object only lifted {obj_height_change:.3f}m")
                        
                        if grasp_attempts < max_grasp_attempts:
                            print(f"Retrying grasp (attempt {grasp_attempts + 1}/{max_grasp_attempts})...")
                            stage = "move_to_object"
                        else:
                            print(f"Max grasp attempts reached. Skipping {current_object}")
                            grasp_attempts = 0
                            stage = "skip_object"

            elif stage == "lift_object":
                target_pos = [eef_pos[0], eef_pos[1], safe_z_height]
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain current orientation (no reorienting for side grasp!)
                action[6] = 1  # Keep gripper closed
                
                if np.linalg.norm(error) < 0.01:
                    print(f"Stage: {stage} -> move_to_bin")
                    stage = "move_to_bin"
                    
            elif stage == "move_to_bin":
                target_placement = object_to_target[current_object]
                target_pos = target_placement + np.array([0, 0, safe_z_height - target_placement[2]])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain orientation (especially important for side-grasped objects)
                action[6] = 1  # Keep gripper closed

                if np.linalg.norm(error[:2]) < 0.01:
                    print(f"Stage: {stage} -> lower_to_bin")
                    stage = "lower_to_bin"
            
            elif stage == "lower_to_bin":
                target_placement = object_to_target[current_object]
                target_pos = target_placement + np.array([0, 0, bin_z_offset])
                error = target_pos - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain orientation
                action[6] = 1  # Keep gripper closed

                if np.linalg.norm(error) < 0.025:
                    print(f"Stage: {stage} -> release")
                    stage = "release"
            
            elif stage == "release":
                action[:3] = 0
                action[3:6] = 0
                action[6] = -1  # Open gripper
                
                release_counter += 1
                if release_counter > 50:
                    print(f"Stage: {stage} -> retract")
                    stage = "retract"
                    release_counter = 0

            elif stage == "retract":
                if 'retract_target' not in locals():
                    retract_target = eef_pos + np.array([0, 0, 0.1])
                
                error = retract_target - eef_pos
                action[:3] = error * P_GAIN
                action[3:6] = 0  # Maintain orientation
                action[6] = -1  # Open gripper

                if np.linalg.norm(error) < 0.01:
                    if 'retract_target' in locals():
                        del retract_target
                    
                    if current_object in objects_to_place:
                        objects_to_place.remove(current_object)
                    
                    if objects_to_place:
                        next_object = objects_to_place[0]
                        next_grasp_type = object_grasp_type[next_object]
                        
                        # Check if we need to reset orientation
                        if current_grasp_type == "side" and next_grasp_type == "top":
                            print(f"\n--- Next object needs top grasp, resetting gripper orientation ---")
                            current_object = next_object
                            current_grasp_type = next_grasp_type
                            stage = "reset_orientation"
                        else:
                            current_object = next_object
                            current_grasp_type = next_grasp_type
                            print(f"\n--- Moving to next object: {current_object} ({current_grasp_type} grasp) ---")
                            stage = "move_to_object"
                    else:
                        print("\n--- All objects placed! Resetting episode. ---")
                        stage = "done"

            elif stage == "reset_orientation":
                # Rotate gripper back to vertical orientation (undo the side grasp rotation)
                action[:3] = 0  # Stay in place
                
                if align_counter < 30:
                    action[3:6] = np.array([0, -0.1, 0])  # Rotate back around y-axis
                else:
                    action[3:6] = 0
                
                action[6] = -1  # Keep gripper open
                
                align_counter += 1
                
                if align_counter > 30:
                    align_counter = 0
                    print(f"Orientation reset complete. Moving to: {current_object} ({current_grasp_type} grasp)")
                    stage = "move_to_object"

            elif stage == "skip_object":
                # Skip this object and move to the next one
                if current_object in objects_to_place:
                    objects_to_place.remove(current_object)
                
                if objects_to_place:
                    current_object = objects_to_place[0]
                    next_grasp_type = object_grasp_type[current_object]

                    # Check if we need to reset orientation
                    if current_grasp_type == "side" and next_grasp_type == "top":
                        print(f"\n--- Next object needs top grasp, resetting gripper orientation ---")
                        current_grasp_type = next_grasp_type
                        stage = "reset_orientation"
                    else:
                        current_grasp_type = next_grasp_type
                        print(f"\n--- Moving to next object: {current_object} ({current_grasp_type} grasp) ---")
                        stage = "move_to_object"
                else:
                    print("\n--- No more objects to place. Resetting episode. ---")
                    stage = "done"

            elif stage == "done":
                obs = env.reset()
                objects_to_place = object_names.copy()
                current_object = objects_to_place[0]
                current_grasp_type = object_grasp_type[current_object]
                grasp_attempts = 0
                stage = "move_to_object"
                print(f"\nReset complete. Starting with: {current_object} ({current_grasp_type} grasp)")
            
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