from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class StackMulti(ManipulationEnv):
    """
    Multi-cube stacking environment supporting 2-4 cubes.
    
    This environment allows stacking multiple cubes in sequence.
    Subclasses should set `n_cubes` to specify the number of cubes.
    """

    n_cubes = 3

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
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
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
        # Settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # Reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # Whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Object placement initializer
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

    def reward(self, action):
        """
        Reward function for the multi-cube stacking task.

        Sparse un-normalized reward:
            - Discrete reward of 2.0 if all required cubes are stacked correctly

        Un-normalized components if using reward shaping:
            - Reaching: encourages the arm to reach cubes
            - Grasping: rewards successful grasping
            - Lifting: rewards lifting cubes
            - Aligning: encourages proper alignment
            - Stacking: rewards successful stacking

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:
                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning  
                - (float): reward for stacking
        """
        # Reaching is successful when the gripper site is close to cubeA
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        dist = min(
            [
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - cubeA_pos)
                for arm in self.robots[0].arms
            ]
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # Grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        # Lifting is successful when cubeA is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # Stacking is successful when all required contacts are made
        r_stack = 0
        if self._check_success():
            r_stack = 2.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        # create a palette of distinct materials
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        bluewood = CustomMaterial(texture="WoodBlue", tex_name="bluewood", mat_name="bluewood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        darkwood = CustomMaterial(texture="WoodDark", tex_name="darkwood", mat_name="darkwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)

        # standard sizes/colors similar to original Stack
        cubes = []
        # cubeA: red
        self.cubeA = BoxObject(name="cubeA", size_min=[0.02, 0.02, 0.02], size_max=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], material=redwood)
        cubes.append(self.cubeA)
        # cubeB: green (slightly larger)
        self.cubeB = BoxObject(name="cubeB", size_min=[0.025, 0.025, 0.025], size_max=[0.025, 0.025, 0.025], rgba=[0, 1, 0, 1], material=greenwood)
        cubes.append(self.cubeB)

        if self.n_cubes >= 3:
            # cubeC: blue
            self.cubeC = BoxObject(name="cubeC", size_min=[0.018, 0.018, 0.018], size_max=[0.018, 0.018, 0.018], rgba=[0, 0, 1, 1], material=bluewood)
            cubes.append(self.cubeC)

        if self.n_cubes >= 4:
            # cubeD: yellow
            self.cubeD = BoxObject(name="cubeD", size_min=[0.02, 0.02, 0.02], size_max=[0.02, 0.02, 0.02], rgba=[1, 1, 0, 1], material=darkwood)
            cubes.append(self.cubeD)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        
        # Set additional cube body ids if they exist
        if hasattr(self, "cubeC"):
            self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        if hasattr(self, "cubeD"):
            self.cubeD_body_id = self.sim.model.body_name2id(self.cubeD.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        
        # Low-level object information
        if self.use_object_obs:
            # Define observables modality
            modality = "object"
            
            # Add cubeA observables
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])
            
            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return np.array(self.sim.data.body_xquat[self.cubeA_body_id])
            
            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeB_body_id])
                
            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return np.array(self.sim.data.body_xquat[self.cubeB_body_id])
            
            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"] if "cubeB_pos" in obs_cache and "cubeA_pos" in obs_cache else np.zeros(3)
                
            @sensor(modality=modality)  
            def gripper_to_cubeA(obs_cache):
                return obs_cache["cubeA_pos"] - obs_cache["robot0_eef_pos"] if "cubeA_pos" in obs_cache and "robot0_eef_pos" in obs_cache else np.zeros(3)
                
            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["robot0_eef_pos"] if "cubeB_pos" in obs_cache and "robot0_eef_pos" in obs_cache else np.zeros(3)
            
            # Register cubeA and cubeB observables
            observables["cubeA_pos"] = Observable(
                name="cubeA_pos",
                sensor=cubeA_pos,
                sampling_rate=self.control_freq,
            )
            observables["cubeA_quat"] = Observable(
                name="cubeA_quat",
                sensor=cubeA_quat,
                sampling_rate=self.control_freq,
            )
            observables["cubeB_pos"] = Observable(
                name="cubeB_pos",
                sensor=cubeB_pos,
                sampling_rate=self.control_freq,
            )
            observables["cubeB_quat"] = Observable(
                name="cubeB_quat",
                sensor=cubeB_quat,
                sampling_rate=self.control_freq,
            )
            observables["cubeA_to_cubeB"] = Observable(
                name="cubeA_to_cubeB",
                sensor=cubeA_to_cubeB,
                sampling_rate=self.control_freq,
            )
            observables["gripper_to_cubeA"] = Observable(
                name="gripper_to_cubeA", 
                sensor=gripper_to_cubeA,
                sampling_rate=self.control_freq,
            )
            observables["gripper_to_cubeB"] = Observable(
                name="gripper_to_cubeB",
                sensor=gripper_to_cubeB,
                sampling_rate=self.control_freq,
            )
            
            # Add cubeC observables if it exists
            if hasattr(self, "cubeC"):
                @sensor(modality=modality)
                def cubeC_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.cubeC_body_id])
                
                @sensor(modality=modality)  
                def cubeC_quat(obs_cache):
                    return np.array(self.sim.data.body_xquat[self.cubeC_body_id])
                
                @sensor(modality=modality)
                def gripper_to_cubeC(obs_cache):
                    return obs_cache["cubeC_pos"] - obs_cache["robot0_eef_pos"] if "cubeC_pos" in obs_cache else np.zeros(3)
                
                observables["cubeC_pos"] = Observable(
                    name="cubeC_pos",
                    sensor=cubeC_pos,
                    sampling_rate=self.control_freq,
                )
                observables["cubeC_quat"] = Observable(
                    name="cubeC_quat", 
                    sensor=cubeC_quat,
                    sampling_rate=self.control_freq,
                )
                observables["gripper_to_cubeC"] = Observable(
                    name="gripper_to_cubeC",
                    sensor=gripper_to_cubeC,
                    sampling_rate=self.control_freq,
                )
            
            # Add cubeD observables if it exists  
            if hasattr(self, "cubeD"):
                @sensor(modality=modality)
                def cubeD_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.cubeD_body_id])
                
                @sensor(modality=modality)
                def cubeD_quat(obs_cache):
                    return np.array(self.sim.data.body_xquat[self.cubeD_body_id])
                
                @sensor(modality=modality)
                def gripper_to_cubeD(obs_cache):
                    return obs_cache["cubeD_pos"] - obs_cache["robot0_eef_pos"] if "cubeD_pos" in obs_cache else np.zeros(3)
                
                observables["cubeD_pos"] = Observable(
                    name="cubeD_pos",
                    sensor=cubeD_pos, 
                    sampling_rate=self.control_freq,
                )
                observables["cubeD_quat"] = Observable(
                    name="cubeD_quat",
                    sensor=cubeD_quat,
                    sampling_rate=self.control_freq, 
                )
                observables["gripper_to_cubeD"] = Observable(
                    name="gripper_to_cubeD",
                    sensor=gripper_to_cubeD,
                    sampling_rate=self.control_freq,
                )
        
        return observables

    def _check_success(self):
        """
        Check if the stacking task is successful for multi-cube.
        For Stack3: cubeA on cubeB and cubeC on cubeA
        For Stack4: additionally cubeD on cubeC
        """
        success = self.check_contact(self.cubeA, self.cubeB)
        if self.n_cubes >= 3:
            success = success and self.check_contact(self.cubeC, self.cubeA)
        if self.n_cubes >= 4:
            success = success and self.check_contact(self.cubeD, self.cubeC)
        return success

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)


class Stack3(StackMulti):
    """Stack environment with 3 colored cubes (red, green, blue)."""

    n_cubes = 3

    def __init__(self, robots, **kwargs):
        super().__init__(robots=robots, **kwargs)


class Stack4(StackMulti):
    """Stack environment with 4 colored cubes (red, green, blue, yellow)."""

    n_cubes = 4

    def __init__(self, robots, **kwargs):
        super().__init__(robots=robots, **kwargs)
