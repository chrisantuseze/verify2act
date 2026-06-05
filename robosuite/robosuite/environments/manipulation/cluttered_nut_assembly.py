"""
Cluttered Nut Assembly environment - extends NutAssembly with obstacles and stacking.
Goal: Place all round nuts on pegs while clearing square nut obstacles that block access.
"""

import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import PegsArena
from robosuite.models.objects import RoundNutObject, SquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class ClutteredNutAssembly(ManipulationEnv):
    """
    Environment for cluttered nut assembly with obstacles.
    
    Task: Place 4 round nuts on pegs. Square nuts act as obstacles that may
    be stacked on round nuts and must be cleared before grasping.
    
    This environment demonstrates symbolic replanning where the robot must:
    1. Detect when a round nut is blocked by a square nut on top
    2. Insert remove_obstacle subtasks into the plan
    3. Handle cascading effects when clearing obstacles creates new obstructions
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        num_round_nuts=4,
        num_square_nuts=3,
        guarantee_overlap=True,
        nut_type_mode="random",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=5000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # Task settings
        self.num_round_nuts = num_round_nuts
        self.num_square_nuts = num_square_nuts
        self.guarantee_overlap = guarantee_overlap
        
        # Nut type mode: "round" or "square" selects target type; "random" picks target each episode
        assert nut_type_mode in {"roundnut", "squarenut", "random", "alternate"}, \
            f"Invalid nut_type_mode: {nut_type_mode}. Must be one of: roundnut, squarenut, random, alternate"
        self.nut_type_mode = nut_type_mode
        self.current_nut_type = nut_type_mode if nut_type_mode != "random" else "roundnut"  # Default for init

        # Placement constraints to keep nuts away from pegs at initialization
        # (prevents spawning over pegs / intersecting peg geometry)
        self.peg_clearance_xy = 0.1 #0.07  # meters, min XY distance from peg center
        self.min_nut_distance = 0.08 #0.06 - For inference  # meters, min XY distance between any two nuts (relaxed to allow near-placement)
        self.placement_max_x = 0.15  # Max x-coordinate (keep nuts on robot side of pegs at x=0.23)
        self.placement_y_range = 0.20  # Max absolute y-coordinate (keep nuts within reach)
        self.placement_max_attempts = 200 #50 - For inference
        
        print(f"NUT TYPE MODE: {self.nut_type_mode}, CURRENT NUT TYPE: {self.current_nut_type}")  # DEBUG
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function - only the target nut type for this episode counts towards success.
        All nuts are present, but only one type contributes to reward.
        """
        # Check which target nuts are on pegs
        self._check_success()
        reward = np.sum(self.round_nuts_on_pegs) if self.current_nut_type == "roundnut" else np.sum(self.square_nuts_on_pegs)

        # Add shaped rewards if enabled
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        
        if self.reward_scale is not None:
            target_count = self.num_round_nuts if self.current_nut_type == "roundnut" else self.num_square_nuts
            reward *= self.reward_scale / max(target_count, 1)
        
        return reward

    def staged_rewards(self):
        """
        Staged rewards for shaping: reaching, grasping, lifting, hovering.
        Only considers the current target nut type.
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # Filter to only target nuts not yet on pegs
        if self.current_nut_type == "roundnut":
            target_names = self.round_nut_names
            target_on_pegs = self.round_nuts_on_pegs
        else:
            target_names = self.square_nut_names
            target_on_pegs = self.square_nuts_on_pegs

        active_nuts = [nut_name for i, nut_name in enumerate(target_names) if not target_on_pegs[i]]

        # Reaching reward - distance to closest active round nut
        r_reach = 0.0
        if active_nuts:
            dists = [
                np.linalg.norm(self.sim.data.body_xpos[self.obj_body_id[nut]][:2] - 
                             self.robots[0]._hand_pos[:2])
                for nut in active_nuts
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # Grasping reward - check if grasping any target nut
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for nut in active_nuts for g in self.obj_geom_id[nut]]
        )) * grasp_mult

        # Lifting reward - target nut lifted above table
        r_lift = 0.0
        if r_grasp > 0.0:
            z_target = self.table_offset[2] + 0.15
            for nut in active_nuts:
                nut_z = self.sim.data.body_xpos[self.obj_body_id[nut]][2]
                z_dist = max(z_target - nut_z, 0.0)
                r_lift = max(r_lift, grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (lift_mult - grasp_mult))

        # Hovering reward - distance from nut to its target peg
        r_hover = 0.0
        if r_lift > 0.0:
            for i, nut in enumerate(active_nuts):
                peg_id = self.nut_type_to_peg[self.current_nut_type]
                peg_pos = self.sim.data.body_xpos[self.peg_body_ids[peg_id]]
                nut_pos = self.sim.data.body_xpos[self.obj_body_id[nut]]
                dist = np.linalg.norm(peg_pos - nut_pos)
                r_hover = max(r_hover, lift_mult + (1 - np.tanh(10.0 * dist)) * (hover_mult - lift_mult))

        return r_reach, r_grasp, r_lift, r_hover

    def _load_model(self):
        """
        Loads arena, objects, and sets up placement.
        """
        super()._load_model()

        # Adjust robot base
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create arena with pegs
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create round nuts (targets)
        self.round_nuts = OrderedDict()
        self.round_nut_names = []
        for i in range(self.num_round_nuts):
            name = f"RoundNut{i}"
            nut = RoundNutObject(name=name)
            self.round_nuts[name] = nut
            self.round_nut_names.append(name)

        # Create square nuts (obstacles)
        self.square_nuts = OrderedDict()
        self.square_nut_names = []
        for i in range(self.num_square_nuts):
            name = f"SquareNut{i}"
            nut = SquareNutObject(name=name)
            self.square_nuts[name] = nut
            self.square_nut_names.append(name)

        # All nuts for placement
        self.nuts = OrderedDict()
        self.nuts.update(self.round_nuts)
        self.nuts.update(self.square_nuts)

        # Map nut types to pegs (PegsArena has 2 pegs)
        self.nut_type_to_peg = {"squarenut": 0, "roundnut": 1}

        # Map round nuts to pegs (cycle through available pegs)
        self.round_nut_to_peg = {}
        num_pegs = 2  # PegsArena has 2 pegs
        for i, nut_name in enumerate(self.round_nut_names):
            self.round_nut_to_peg[nut_name] = i % num_pegs

        # Create placement sampler if not provided
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            
            # Use a single region for all nuts to allow flexible placement
            # Constrain placement to robot-accessible area (before pegs at x=0.23)
            # and within reasonable y-range to avoid nuts being too far left/right
            for nut_obj in self.nuts.values():
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{nut_obj.name}Sampler",
                        mujoco_objects=nut_obj,
                        x_range=[-0.15, 0.10],  # Keep nuts between robot and pegs (pegs at x=0.23)
                        y_range=[-0.18, 0.18],   # Constrain y-range to keep nuts within reach
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False, #True, @TODO: Chris revisit
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                        rng=self.rng,
                    )
                )
        
        # Don't call reset() — it clears sampler mujoco_objects

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.nuts.values()),
        )

    def _setup_references(self):
        """
        Sets up object and peg body IDs for runtime access.
        """
        super()._setup_references()

        # Cache body IDs
        self.obj_body_id = {}
        self.obj_geom_id = {}
        
        for nut_name in list(self.round_nut_names) + list(self.square_nut_names):
            # robosuite adds "_main" suffix to object body names
            self.obj_body_id[nut_name] = self.sim.model.body_name2id(f"{nut_name}_main")
            
            # Get geom IDs for this nut
            geom_ids = []
            for i in range(self.sim.model.ngeom):
                geom_name = self.sim.model.geom_id2name(i)
                if geom_name and nut_name.lower() in geom_name.lower():
                    geom_ids.append(geom_name)
            self.obj_geom_id[nut_name] = geom_ids

        # Cache peg body IDs (PegsArena has 2 pegs)
        self.peg_body_ids = {}
        for i in range(2):  # Only 2 pegs available
            self.peg_body_ids[i] = self.sim.model.body_name2id(f"peg{i+1}")

        # Track which nuts are on pegs
        self.round_nuts_on_pegs = np.zeros(self.num_round_nuts)
        self.square_nuts_on_pegs = np.zeros(self.num_square_nuts)

    def _setup_observables(self):
        """
        Sets up observables for all nuts.
        """
        observables = super()._setup_observables()

        # Observables for all nuts (both round and square)
        if self.use_object_obs:
            modality = "object"

            for nut_name in list(self.round_nut_names) + list(self.square_nut_names):
                @sensor(modality=modality)
                def nut_pos(obs_cache, nut_name=nut_name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[nut_name]])

                @sensor(modality=modality)
                def nut_quat(obs_cache, nut_name=nut_name):
                    return T.convert_quat(
                        np.array(self.sim.data.body_xquat[self.obj_body_id[nut_name]]), to="xyzw"
                    )

                observables[f"{nut_name}_pos"] = Observable(
                    name=f"{nut_name}_pos",
                    sensor=nut_pos,
                    sampling_rate=self.control_freq,
                )
                observables[f"{nut_name}_quat"] = Observable(
                    name=f"{nut_name}_quat",
                    sensor=nut_quat,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation with all nuts placed. Only one type is the target for this episode.
        """
        # Set target nut type for this episode based on mode
        if self.nut_type_mode == "random":
            self.current_nut_type = self.rng.choice(["roundnut", "squarenut"])
        elif self.nut_type_mode == "alternate":
            # Alternate between round and square each reset
            self.current_nut_type = "squarenut" if self.current_nut_type == "roundnut" else "roundnut"
        # else: nut_type_mode is "roundnut" or "squarenut", current_nut_type stays fixed
        
        print(f"Episode starting with target type: {self.current_nut_type}")  # DEBUG
        
        super()._reset_internal()

        # Sample from the placement initializer for all objects, rejecting
        # placements too close to pegs to avoid spawning over/in peg geometry.
        object_placements = None
        for _ in range(self.placement_max_attempts):
            candidate = self.placement_initializer.sample()
            if self._placements_valid(candidate):
                object_placements = candidate
                break

        if object_placements is None:
            # Fallback: accept the last candidate but warn
            object_placements = self.placement_initializer.sample()
            print("Warning: Could not find peg-safe placements after "
                  f"{self.placement_max_attempts} attempts; using last sample.")

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Snap square nuts to a random multiple of 90° so their sides are always
        # axis-aligned with the table/peg. A random yaw causes the gripper to grasp
        # at an arbitrary angle, making square-peg insertion unreliable.
        _axis_aligned_angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for nut_name in self.square_nut_names:
            current_qpos = self.sim.data.get_joint_qpos(f"{nut_name}_joint0")
            pos = current_qpos[:3]
            angle = self.rng.choice(_axis_aligned_angles)
            # MuJoCo free-joint quat layout: [w, x, y, z]; z-axis rotation = [cos(θ/2), 0, 0, sin(θ/2)]
            axis_aligned_quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
            self.sim.data.set_joint_qpos(f"{nut_name}_joint0", np.concatenate([pos, axis_aligned_quat]))

        # Reset success tracking
        self.round_nuts_on_pegs = np.zeros(self.num_round_nuts)
        self.square_nuts_on_pegs = np.zeros(self.num_square_nuts)

        # Guarantee at least one stacked pair per episode when requested.
        if self.guarantee_overlap and self.round_nut_names and self.square_nut_names:
            self._force_guaranteed_overlap()

    def _placements_valid(self, placements):
        """
        Check that all placement positions are sufficiently far from pegs
        and within reachable area (not too far in x or y axes).
        """
        try:
            peg_positions = [
                np.array(self.sim.data.body_xpos[self.peg_body_ids[i]]) for i in range(2)
            ]
        except Exception:
            # If peg positions aren't available, accept placements
            return True

        positions = []
        for obj_pos, _, _ in placements.values():
            obj_xy = np.array(obj_pos[:2])
            
            # Check 1: Not too close to any peg (radial distance)
            for peg_pos in peg_positions:
                if np.linalg.norm(obj_xy - peg_pos[:2]) < self.peg_clearance_xy:
                    return False
            
            # Check 2: X-axis constraint - keep nuts on robot side (x < max_x)
            # Pegs are at x=0.23, so keep nuts at x < 0.15 to maintain clearance
            if obj_pos[0] > self.placement_max_x:
                return False
            
            # Check 3: Y-axis constraint - keep nuts within reasonable reach
            # Prevent nuts from being too far left or right
            if abs(obj_pos[1]) > self.placement_y_range:
                return False

            positions.append(obj_xy)

        # Check 4: Minimum distance between any two nuts to prevent overlap/stacking
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < self.min_nut_distance:
                    return False

        return True

    def _force_guaranteed_overlap(self):
        """
        Guarantee at least one obstacle nut is stacked on a target nut at episode start.

        This is called unconditionally every reset so the task always requires
        obstacle-clearing, regardless of ``initial_stacking_prob``.

        Strategy:
          - Determine which nut type is the target and which is the obstacle.
          - Pick one target nut and one obstacle nut at random.
          - Directly set the obstacle nut's joint qpos to sit on top of the target.
          - Run a short physics warmup so the configuration is physically consistent
            before the rest of reset proceeds.
        """
        # Determine target/obstacle roles
        if self.current_nut_type == "roundnut":
            target_nuts = list(self.round_nut_names)
            obstacle_nuts = list(self.square_nut_names)
        else:
            target_nuts = list(self.square_nut_names)
            obstacle_nuts = list(self.round_nut_names)

        if not target_nuts or not obstacle_nuts:
            return  # Nothing to stack

        # Shuffle to randomise which pair is chosen each episode
        self.rng.shuffle(target_nuts)
        self.rng.shuffle(obstacle_nuts)

        target_nut = target_nuts[0]
        obstacle_nut = obstacle_nuts[0]

        print(f"[Stacking] Guaranteeing overlap: {obstacle_nut} on top of {target_nut}")
        self._stack_nut_on_nut(obstacle_nut, target_nut)

        # Brief physics warmup so MuJoCo registers the new configuration
        for _ in range(5):
            self.sim.step()

    # _apply_initial_stacking has been removed; use guarantee_overlap=True instead.

    def _stack_nut_on_nut(self, top_nut, bottom_nut):
        """
        Stack top_nut on bottom_nut by setting position.
        """
        bottom_pos = self.sim.data.body_xpos[self.obj_body_id[bottom_nut]].copy()
        bottom_quat = self.sim.data.body_xquat[self.obj_body_id[bottom_nut]].copy()
        
        # Place on top with small offset
        stack_height = 0.04
        top_pos = bottom_pos.copy()
        top_pos[2] += stack_height
        
        # Set position and orientation (joint name has _joint0 suffix, not _jnt0)
        self.sim.data.set_joint_qpos(
            f"{top_nut}_joint0",
            np.concatenate([top_pos, bottom_quat])
        )
        
        # Zero out velocities
        self.sim.data.set_joint_qvel(f"{top_nut}_joint0", np.zeros(6))

    def _is_nut_stacked(self, nut_name):
        """
        Check if nut is already stacked on another nut (z > table + threshold).
        """
        nut_z = self.sim.data.body_xpos[self.obj_body_id[nut_name]][2]
        return nut_z > self.table_offset[2] + 0.05
    
    def on_peg(self, obj_pos, peg_id):
        peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_ids[peg_id]])
        # XY: radial distance from peg axis. 0.07m gives enough room for the nut
        # body to be slightly off-center while still clearly on the peg shaft.
        # Z: nut center must be below the peg top height (origin + 0.1m). Allowing up to 
        # peg_pos[2] + 0.085m accommodates up to 4 stacked nuts (highest center at 0.87m)
        # while still correctly rejecting a nut resting on the tip of the peg (center at 0.96m).
        xy_dist = np.linalg.norm(obj_pos[:2] - peg_pos[:2])
        return xy_dist < 0.07 and obj_pos[2] < peg_pos[2] + 0.085

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action, including checking for success.
        
        Args:
            action (np.array): Action to execute within the environment
            
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info dict with success status
        """
        reward = self.reward(action)

        # Episode ends when no target nuts remain graspable on the table —
        # whether they were properly inserted, resting on the peg tip, or fell off.
        all_processed = self._all_target_nuts_processed()

        # Success requires every target nut to be properly inserted (strict on_peg check).
        success = self._check_success() if all_processed else False

        self.done = all_processed or ((self.timestep >= self.horizon) and not self.ignore_done)

        info = {"success": success, "all_target_nuts_processed": all_processed}

        return reward, self.done, info

    def _check_success(self):
        """
        Check if all target nuts (based on current_nut_type) are on their target peg.
        Returns True if all target nuts are successfully placed, which will end the episode.
        """
        if self.current_nut_type == "roundnut":
            target_names = self.round_nut_names
            target_on_pegs = self.round_nuts_on_pegs
        else:
            target_names = self.square_nut_names
            target_on_pegs = self.square_nuts_on_pegs

        peg_id = self.nut_type_to_peg[self.current_nut_type]

        # Check each target nut to see if it's on the correct peg
        for i, nut_name in enumerate(target_names):
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[nut_name]]
            
            # Check if nut is on the target peg (not just any peg)
            on_target_peg = self.on_peg(obj_pos, peg_id)
            
            # Check if gripper is far enough away (to ensure nut is released)
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            gripper_clear = r_reach < 0.6
            
            # Nut is successfully placed if on peg and gripper is clear
            target_on_pegs[i] = int(on_target_peg and gripper_clear)

        # Success if all target nuts are placed
        success = np.sum(target_on_pegs) == len(target_names)
        # print(f"SUCCESS CHECK: target_type={self.current_nut_type}, on_pegs={target_on_pegs}, success={success}")  # DEBUG
        return success
    
    def get_available_target_nuts(self):
        """
        Get list of target nuts that are still available to grasp (on table, not on peg).
        
        Returns:
            List of nut names that can still be grasped and placed.
        """
        if self.current_nut_type == "roundnut":
            target_names = self.round_nut_names
            target_on_pegs = self.round_nuts_on_pegs
        else:
            target_names = self.square_nut_names
            target_on_pegs = self.square_nuts_on_pegs
        
        # Threshold for considering a nut "off the table" (fallen or removed)
        table_z_min = self.table_offset[2] - 0.1  # 10cm below table surface
        
        available_nuts = []
        for i, nut_name in enumerate(target_names):
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[nut_name]]
            
            # Check if nut is still on the table (not fallen off)
            on_table = obj_pos[2] > table_z_min
            
            # Available if on table and not successfully placed on peg
            if on_table and not target_on_pegs[i]:
                available_nuts.append(nut_name)
        
        return available_nuts
    
    def _all_target_nuts_processed(self):
        """
        Check if all target nuts have been processed (either on peg or off table).
        This allows episode to end even if placement wasn't fully successful.
        
        Returns True if no target type nuts remain on the table to be grasped.
        """
        available_nuts = self.get_available_target_nuts()
        all_processed = len(available_nuts) == 0
        
        if self.current_nut_type == "roundnut":
            target_on_pegs = self.round_nuts_on_pegs
        else:
            target_on_pegs = self.square_nuts_on_pegs
        
        if all_processed and not np.all(target_on_pegs):
            print(f"INFO: All {self.current_nut_type} nuts processed but not all on peg. "
                  f"on_pegs={target_on_pegs}, ending episode.")
        
        return all_processed
    
    def visualize(self, vis_settings):
        """
        Visualization pass-through.
        """
        super().visualize(vis_settings=vis_settings)
