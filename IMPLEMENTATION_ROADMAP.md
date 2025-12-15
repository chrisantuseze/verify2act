# Points2Plans + Robosuite Integration: Implementation Roadmap

## Quick Start Summary

To integrate Points2Plans as a planner for robosuite, you need to:

1. **Create a state converter**: Transform robosuite observations → Points2Plans format
2. **Integrate LLM module**: Wrap existing Points2Plans/LLM for task decomposition (REQUIRED)
3. **Build a planner interface**: Load dynamics model and run inference
4. **Implement action executor**: Convert high-level plans → robot commands
5. **Create planning loop**: Orchestrate the full pipeline (LLM → Planner → Executor)

The good news: Your data collection pipeline already handles the format conversion, and Points2Plans already includes the LLM module!

## Detailed Implementation Guide

---

## 1. State Converter Module

### File: `robosuite/planning/state_converter.py`

This module converts robosuite observations to Points2Plans tensor format in real-time.

**Reuse these existing components**:
- `PointCloudGenerator` - Already generates segmented point clouds
- `StateCapture` - Already extracts object poses and states
- `DataFormatter` - Already computes relational predicates

**Key difference from data collection**: 
- Data collection: Batch process entire episodes
- Real-time: Process single timestep on-demand

### Implementation Template:

```python
import numpy as np
import torch
from typing import Dict, List
from robosuite.utils.pointcloud_generator import PointCloudGenerator
from data_capture.state_capture import StateCapture
from data_capture.metadata_extractor import MetadataExtractor

class StateConverter:
    """Convert robosuite observations to Points2Plans format."""
    
    def __init__(self, env, num_points=128, voxel_size=0.005):
        self.env = env
        self.sim = env.sim
        self.num_points = num_points
        
        # Initialize helpers
        self.metadata_extractor = MetadataExtractor(self.sim)
        self.pcd_generator = PointCloudGenerator(
            voxel_size=voxel_size,
            bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.7, 1.5]]
        )
        
        # Extract object metadata (once)
        self.object_metadata = self.metadata_extractor.extract_all_objects()
        self.state_capture = StateCapture(env, self.object_metadata)
        
        self.num_objects = len(self.object_metadata)
        
    def convert(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert current observation to Points2Plans format.
        
        Returns dictionary with keys:
            - batch_voxel_list_single: [1, num_objects, num_points, 3]
            - batch_one_hot_encoding: [1, num_objects, num_types]
            - batch_6DOF_pose: [1, num_objects, 6]
            - batch_all_obj_pair_relation: [1, num_objects, num_objects, num_predicates]
            - batch_env_identity: [1, num_objects, 3]
            - batch_grasp_identity: [1, num_objects, 1]
            - batch_edge_attr: Edge indices for graph
            - batch_num_objects: int
        """
        # 1. Generate point clouds
        point_clouds = self._generate_point_clouds()
        
        # 2. Get object states
        object_states = self.state_capture.get_object_states()
        
        # 3. Compute relational predicates
        relations = self._compute_relations(object_states)
        
        # 4. Build one-hot encodings
        one_hot = self._build_one_hot_encodings()
        
        # 5. Get environment features
        env_identity = self._build_env_identity()
        grasp_identity = self._build_grasp_identity()
        
        # 6. Format as tensors
        state_dict = {
            'batch_voxel_list_single': torch.FloatTensor(point_clouds).unsqueeze(0),
            'batch_one_hot_encoding': torch.FloatTensor(one_hot).unsqueeze(0),
            'batch_6DOF_pose': self._format_poses(object_states),
            'batch_all_obj_pair_relation': torch.FloatTensor(relations).unsqueeze(0),
            'batch_env_identity': torch.FloatTensor(env_identity).unsqueeze(0),
            'batch_grasp_identity': torch.FloatTensor(grasp_identity).unsqueeze(0),
            'batch_edge_attr': self._build_edge_index(),
            'batch_num_objects': self.num_objects,
        }
        
        return state_dict
    
    def _generate_point_clouds(self) -> np.ndarray:
        """Generate point cloud for each object."""
        # Capture RGB-D from cameras
        rgb, depth = self.pcd_generator.capture_rgbd(self.sim, ["frontview"])
        
        # Generate full point cloud
        full_pcd = self.pcd_generator.rgbd_to_pointcloud(rgb[0], depth[0], "frontview")
        
        # Segment by object
        segmented = self.pcd_generator.segment_by_objects(
            full_pcd, self.sim, self.object_metadata
        )
        
        # Downsample to fixed size
        point_clouds = []
        for obj_name in sorted(self.object_metadata.keys()):
            pcd = segmented.get(obj_name, np.zeros((self.num_points, 3)))
            # TODO: Resample/pad to exactly num_points
            point_clouds.append(pcd[:self.num_points])
        
        return np.array(point_clouds)
    
    def _compute_relations(self, object_states: Dict) -> np.ndarray:
        """Compute pairwise relational predicates."""
        n = self.num_objects
        relations = np.zeros((n, n, 3))  # [On, Inside, Graspable]
        
        # TODO: Implement based on DataFormatter logic
        # - Check vertical alignment and contact for "On"
        # - Check containment for "Inside"
        # - Check feasibility for "Graspable"
        
        return relations
    
    def _build_one_hot_encodings(self) -> np.ndarray:
        """Build one-hot encoding for object types."""
        # Map object types to indices (e.g., 0=milk, 1=cereal, 2=bread, etc.)
        type_to_idx = {
            'milk': 0, 'cereal': 1, 'bread': 2, 'can': 3,
            'bin': 4, 'table': 5
        }
        
        encodings = []
        for obj_name in sorted(self.object_metadata.keys()):
            obj_type = self.object_metadata[obj_name]['type']
            idx = type_to_idx.get(obj_type, 0)
            one_hot = np.zeros(len(type_to_idx))
            one_hot[idx] = 1
            encodings.append(one_hot)
        
        return np.array(encodings)
    
    # ... implement other helper methods
```

---

## 2. LLM Task Planner Integration

### File: `robosuite/planning/llm_task_planner.py`

This wraps the existing Points2Plans LLM module (`Points2Plans/LLM/`) for task decomposition.

### Implementation Template:

```python
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add Points2Plans to path
sys.path.append(str(Path(__file__).parent.parent.parent / "Points2Plans"))

from LLM.fm_planning import models, utils
import ast

class LLMTaskPlanner:
    """Wrapper for Points2Plans LLM module for task decomposition."""
    
    def __init__(self, model_config_path: str, api_key: str = None):
        """
        Initialize LLM task planner.
        
        Args:
            model_config_path: Path to model config (e.g., LLM/configs/models/pretrained/generative/gpt4.yaml)
            api_key: OpenAI API key (or from environment)
        """
        # Load LLM model
        self.model_factory = models.PretrainedModelFactory(
            model_config_path,
            api_key=api_key,
            device="auto"
        )
        self.model = self.model_factory()
        
        print(f"Loaded LLM model from {model_config_path}")
    
    def generate_goals_and_plans(self, 
                                task_description: str,
                                objects: List[str],
                                initial_predicates: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generate goal predicates and high-level action plans from natural language.
        
        This uses Points2Plans' two-stage LLM approach:
        1. Goal Prediction: Task description → Goal predicates
        2. Task Planning: Task + Goals → Action sequence
        
        Args:
            task_description: Natural language task (e.g., "Put all objects in the bin")
            objects: List of object names in scene
            initial_predicates: Current scene predicates
        
        Returns:
            goals: List of goal predicate lists (e.g., [["On(milk, bin)", "On(cereal, bin)"]])
            plans: List of action sequence lists (e.g., [["Pick(milk, table)", "Place(milk, bin)", ...]])
        """
        # Create prompt for goal prediction
        goal_prompt = self._create_goal_prediction_prompt(
            task_description,
            objects,
            initial_predicates
        )
        
        # Query LLM for goals
        response = self.model.forward(goal_prompt)
        predicted_goals = ast.literal_eval(response["choices"][0]["message"]["content"])
        
        # Create prompt for task planning
        plan_prompt = self._create_task_planning_prompt(
            task_description,
            objects,
            initial_predicates,
            predicted_goals
        )
        
        # Query LLM for plans
        response = self.model.forward(plan_prompt)
        predicted_plans = ast.literal_eval(response["choices"][0]["message"]["content"])
        
        print(f"LLM Generated Goals: {predicted_goals}")
        print(f"LLM Generated Plans: {predicted_plans}")
        
        return predicted_goals, predicted_plans
    
    def _create_goal_prediction_prompt(self,
                                       task_description: str,
                                       objects: List[str],
                                       initial_predicates: List[str]) -> dict:
        """Create prompt for goal prediction."""
        system_msg = """You are a robotic task planning assistant. Given a task description and objects, 
        predict the goal state as a list of spatial predicates.
        
        Available predicates:
        - On(object, surface): object is on top of surface
        - Inside(object, container): object is inside container
        - Near(object1, object2): objects are close to each other
        
        Return ONLY a Python list of goal predicates."""
        
        user_msg = f"""Task: {task_description}
        Objects: {', '.join(objects)}
        Initial state: {', '.join(initial_predicates)}
        
        Predict the goal state:"""
        
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        }
    
    def _create_task_planning_prompt(self,
                                     task_description: str,
                                     objects: List[str],
                                     initial_predicates: List[str],
                                     goals: List[str]) -> dict:
        """Create prompt for task planning."""
        system_msg = """You are a robotic task planning assistant. Given a task, objects, and goals, 
        generate a sequence of actions to achieve the goals.
        
        Available actions:
        - Pick(object, location): Pick up object from location
        - Place(object, location): Place object at location
        - Push(object, direction): Push object in direction
        
        Return ONLY a Python list of action sequences."""
        
        user_msg = f"""Task: {task_description}
        Objects: {', '.join(objects)}
        Initial state: {', '.join(initial_predicates)}
        Goals: {', '.join(goals)}
        
        Generate action plan:"""
        
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        }
    
    def goals_to_predicates(self, goals: List[str], 
                           object_to_id: Dict[str, int],
                           num_objects: int) -> np.ndarray:
        """
        Convert goal strings to predicate tensor.
        
        Args:
            goals: List of goal strings (e.g., ["On(milk, bin)"])
            object_to_id: Mapping from object names to IDs
            num_objects: Total number of objects
        
        Returns:
            goal_predicates: [num_objects, num_objects, 3] tensor
        """
        goal_predicates = np.zeros((num_objects, num_objects, 3))
        
        for goal in goals:
            # Parse goal string (e.g., "On(milk, bin)")
            if "On(" in goal:
                predicate_idx = 0
            elif "Inside(" in goal:
                predicate_idx = 1
            elif "Graspable(" in goal:
                predicate_idx = 2
            else:
                continue
            
            # Extract object names
            content = goal.split('(')[1].split(')')[0]
            parts = [p.strip() for p in content.split(',')]
            
            if len(parts) == 2:
                obj1, obj2 = parts
                if obj1 in object_to_id and obj2 in object_to_id:
                    id1 = object_to_id[obj1]
                    id2 = object_to_id[obj2]
                    goal_predicates[id1, id2, predicate_idx] = 1.0
        
        return goal_predicates
```

---

## 3. Points2Plans Planner Interface

### File: `robosuite/planning/points2plans_planner.py`

This wraps the Points2Plans model for inference.

### Implementation Template:

```python
import torch
import numpy as np
import sys
from pathlib import Path

# Add Points2Plans to path
sys.path.append(str(Path(__file__).parent.parent.parent / "Points2Plans"))

from relational_dynamics.base_RD import RelationalDynamics
from relational_dynamics.config.base_config import BaseConfig

class Points2PlansPlanner:
    """Wrapper for Points2Plans relational dynamics model."""
    
    def __init__(self, checkpoint_path: str, config_args: dict):
        """
        Initialize planner with pretrained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_args: Configuration dictionary (from BaseConfig)
        """
        # Create config
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.config = BaseConfig(config_args, dtype=dtype)
        
        # Load model
        self.model = RelationalDynamics(self.config)
        self.model.load_checkpoint(checkpoint_path)
        self.model.set_model_device(self.config.get_device())
        
        # Set to eval mode
        for m in self.model.get_model_list():
            m.eval()
        
        print(f"Loaded planner from {checkpoint_path}")
    
    def plan_action(self, state: dict, goal_predicates: np.ndarray, 
                    num_samples: int = 50) -> dict:
        """
        Plan best action using sampling-based planning.
        
        Args:
            state: Current state in Points2Plans format
            goal_predicates: Target relational predicates [num_objects, num_objects, 3]
            num_samples: Number of candidate actions to sample
        
        Returns:
            action: {
                'skill': 0 or 1,  # 0=pick-place, 1=push
                'object_id': int,
                'target_id': int,  # For place actions
                'offset': [x, y]   # Relative offset
            }
        """
        # Extract current embeddings
        with torch.no_grad():
            current_latent = self._encode_state(state)
        
        # Sample candidate actions
        candidates = self._sample_candidate_actions(state, num_samples)
        
        # Evaluate each candidate
        best_action = None
        best_cost = float('inf')
        
        for action in candidates:
            # Predict next state
            predicted_state = self._predict_forward(current_latent, action, state)
            
            # Compute cost (distance to goal)
            cost = self._evaluate_cost(predicted_state, goal_predicates)
            
            if cost < best_cost:
                best_cost = cost
                best_action = action
        
        return best_action
    
    def _encode_state(self, state: dict) -> torch.Tensor:
        """Encode current state to latent representation."""
        device = self.config.get_device()
        
        # Move to device
        voxel_data = state['batch_voxel_list_single'].to(device)
        one_hot = state['batch_one_hot_encoding'].to(device)
        
        # Encode point clouds
        batch_size, num_obj, num_pts, _ = voxel_data.shape
        reshaped = voxel_data.reshape(batch_size * num_obj, num_pts, 3)
        
        img_emb = self.model.emb_model(reshaped)
        img_emb = img_emb.reshape(batch_size, num_obj, -1)
        
        # Add one-hot encoding
        latent_one_hot = self.model.classif_model.one_hot_encoding_embed(
            torch.argmax(one_hot, dim=2)
        )
        
        # Concatenate
        node_pose = torch.cat([img_emb, latent_one_hot], dim=-1)
        
        return node_pose
    
    def _predict_forward(self, current_latent: torch.Tensor, 
                        action: dict, state: dict) -> dict:
        """
        Predict next state given action.
        
        Based on base_RD.py training() method dynamics forward pass.
        """
        device = self.config.get_device()
        
        # Encode action
        action_tensor = self._encode_action(action, state)
        
        # Concatenate state + action
        graph_node_action = torch.cat([
            current_latent,
            action_tensor['discrete'],
            action_tensor['continuous']
        ], dim=1)
        
        # Run dynamics model (skill-dependent)
        if action['skill'] == 0:
            next_latent = self.model.classif_model.graph_dynamics_0(
                graph_node_action[0],
                src_key_padding_mask=None  # TODO: build mask
            )
        else:
            next_latent = self.model.classif_model.graph_dynamics_1(
                graph_node_action[0],
                src_key_padding_mask=None
            )
        
        # Decode to predicates and poses
        pred_latent = next_latent[:-2, :].unsqueeze(0)
        decoded = self.model.classif_model_decoder(
            pred_latent,
            state['batch_edge_attr']
        )
        
        return {
            'latent': pred_latent,
            'predicates': decoded['pred_sigmoid'],
            'poses': decoded['predicted_pose'],
            'env_identity': decoded['env_identity'],
            'grasp_identity': decoded['grasp_identity']
        }
    
    def _sample_candidate_actions(self, state: dict, num_samples: int) -> list:
        """Sample candidate actions to evaluate."""
        candidates = []
        num_objects = state['batch_num_objects']
        
        # Sample pick-place actions
        for _ in range(num_samples):
            obj_id = np.random.randint(0, num_objects - 2)  # Exclude bins
            target_id = np.random.randint(0, num_objects)
            
            # Sample offset around target
            offset = np.random.uniform(-0.05, 0.05, size=2)
            
            candidates.append({
                'skill': 0,  # Pick-place
                'object_id': obj_id,
                'target_id': target_id,
                'offset': offset
            })
        
        return candidates
    
    def _evaluate_cost(self, predicted_state: dict, 
                      goal_predicates: np.ndarray) -> float:
        """
        Compute cost as distance to goal predicates.
        
        Lower cost = closer to goal.
        """
        pred_relations = predicted_state['predicates'].cpu().numpy()[0]
        
        # Binary cross-entropy or L2 distance
        cost = np.mean((pred_relations - goal_predicates) ** 2)
        
        return cost
    
    def _encode_action(self, action: dict, state: dict) -> dict:
        """Encode action as tensor."""
        device = self.config.get_device()
        num_objects = state['batch_num_objects']
        
        # Discrete part (one-hot object selection)
        discrete = torch.zeros(1, 1, num_objects).to(device)
        discrete[0, 0, action['object_id']] = 1.0
        
        # Continuous part (offset)
        continuous = torch.FloatTensor(action['offset']).unsqueeze(0).unsqueeze(0).to(device)
        
        return {'discrete': discrete, 'continuous': continuous}
```

---

## 3. Action Executor

### File: `robosuite/planning/action_executor.py`

Converts high-level primitives to robot commands.

### Implementation Template:

```python
import numpy as np
from typing import Dict

class ActionExecutor:
    """Execute high-level action primitives in robosuite."""
    
    def __init__(self, env):
        self.env = env
        
        # Control parameters (adapt from your heuristic policy)
        self.p_gain = 10.0
        self.r_gain = 5.0
        self.grasp_duration = 50
        
    def execute(self, action: Dict, obs: Dict) -> bool:
        """
        Execute action primitive.
        
        Args:
            action: {
                'skill': 0 (pick-place) or 1 (push),
                'object_id': int,
                'target_id': int,
                'offset': [x, y]
            }
            obs: Current robosuite observation
        
        Returns:
            success: True if action completed successfully
        """
        if action['skill'] == 0:
            return self._execute_pick_place(action, obs)
        elif action['skill'] == 1:
            return self._execute_push(action, obs)
        else:
            raise ValueError(f"Unknown skill: {action['skill']}")
    
    def _execute_pick_place(self, action: Dict, obs: Dict) -> bool:
        """Execute pick and place primitive."""
        # Get object position
        obj_name = self._get_object_name(action['object_id'])
        obj_pos = obs[f"{obj_name}_pos"]
        
        # Phase 1: Move to object
        success = self._move_to_position(obj_pos + np.array([0, 0, 0.1]))
        if not success:
            return False
        
        # Phase 2: Grasp
        success = self._grasp_object(obj_pos)
        if not success:
            return False
        
        # Phase 3: Lift
        success = self._move_to_position(obj_pos + np.array([0, 0, 0.2]))
        if not success:
            return False
        
        # Phase 4: Move to target
        target_name = self._get_object_name(action['target_id'])
        target_pos = obs[f"{target_name}_pos"]
        place_pos = target_pos + np.array([action['offset'][0], 
                                           action['offset'][1], 
                                           0.15])
        
        success = self._move_to_position(place_pos)
        if not success:
            return False
        
        # Phase 5: Place
        success = self._release_object()
        
        return success
    
    def _move_to_position(self, target_pos: np.ndarray, 
                         max_steps: int = 100) -> bool:
        """Move end-effector to target position."""
        for _ in range(max_steps):
            # Get current EE position
            ee_pos = self.env.robots[0].get_eef_position()
            
            # Compute error
            pos_error = target_pos - ee_pos
            
            # Check convergence
            if np.linalg.norm(pos_error) < 0.01:
                return True
            
            # Proportional control
            action = np.zeros(self.env.action_dim)
            action[:3] = self.p_gain * pos_error
            action = np.clip(action, -1, 1)
            
            # Step environment
            self.env.step(action)
        
        return False  # Timeout
    
    def _grasp_object(self, obj_pos: np.ndarray) -> bool:
        """Close gripper to grasp object."""
        # Move down to object
        self._move_to_position(obj_pos + np.array([0, 0, 0.02]))
        
        # Close gripper
        action = np.zeros(self.env.action_dim)
        action[-1] = 1.0  # Close gripper
        
        for _ in range(self.grasp_duration):
            self.env.step(action)
        
        # Verify grasp (check if object lifted)
        # TODO: Implement grasp verification
        
        return True
    
    def _release_object(self) -> bool:
        """Open gripper to release object."""
        action = np.zeros(self.env.action_dim)
        action[-1] = -1.0  # Open gripper
        
        for _ in range(self.grasp_duration):
            self.env.step(action)
        
        return True
    
    def _get_object_name(self, obj_id: int) -> str:
        """Map object ID to name."""
        # TODO: Implement based on your object metadata
        return f"object_{obj_id}"
```

---

## 4. Planning Controller

### File: `robosuite/planning/planning_controller.py`

Orchestrates the full planning loop.

```python
from .state_converter import StateConverter
from .points2plans_planner import Points2PlansPlanner
from .action_executor import ActionExecutor

class PlanningController:
    """Main controller for Points2Plans-based planning."""
    
    def __init__(self, env, checkpoint_path: str, llm_config_path: str, api_key: str = None):
        self.env = env
        
        # Initialize components
        self.state_converter = StateConverter(env)
        
        # Initialize LLM for task decomposition (REQUIRED)
        self.llm_planner = LLMTaskPlanner(
            llm_config_path,
            api_key=api_key
        )
        
        # Initialize dynamics model for action planning
        self.planner = Points2PlansPlanner(
            checkpoint_path,
            config_args=self._get_default_config()
        )
        
        self.executor = ActionExecutor(env)
    
    def run_episode(self, goal_description: str, max_steps: int = 100) -> bool:
        """
        Run one episode with planning.
        
        Args:
            goal_description: Natural language goal (e.g., "Put all objects in bin")
            max_steps: Maximum planning steps
        
        Returns:
            success: True if goal achieved
        """
        obs = self.env.reset()
        
        # Step 1: Use LLM to generate goals and high-level plan
        objects = list(self.state_converter.object_metadata.keys())
        initial_predicates = self._get_current_predicates(obs)
        
        goals, plans = self.llm_planner.generate_goals_and_plans(
            goal_description,
            objects,
            initial_predicates
        )
        
        # Convert LLM goals to predicate tensor
        object_to_id = {name: i for i, name in enumerate(sorted(objects))}
        goal_predicates = self.llm_planner.goals_to_predicates(
            goals[0],  # Use first goal set
            object_to_id,
            len(objects)
        )
        
        for step in range(max_steps):
            print(f"\n=== Step {step} ===")
            
            # Convert observation to planner format
            state = self.state_converter.convert(obs)
            
            # Plan action
            action = self.planner.plan_action(state, goal_predicates)
            print(f"Planned action: {action}")
            
            # Execute action
            success = self.executor.execute(action, obs)
            
            if not success:
                print("Action execution failed, replanning...")
                continue
            
            # Check goal satisfaction
            current_predicates = state['batch_all_obj_pair_relation'][0].numpy()
            if self._check_goal(current_predicates, goal_predicates):
                print("Goal achieved!")
                return True
            
            # Update observation
            obs, _, _, _ = self.env.step(np.zeros(self.env.action_dim))
        
        print("Max steps reached")
        return False
    
    def _parse_goal(self, description: str) -> np.ndarray:
        """Parse natural language to goal predicates."""
        # Simple hardcoded version
        # TODO: Use LLM for more flexible parsing
        num_objects = self.state_converter.num_objects
        goal = np.zeros((num_objects, num_objects, 3))
        
        if "bin" in description.lower():
            # Set all objects "On" bin
            bin_id = num_objects - 1  # Assume last object is bin
            for obj_id in range(num_objects - 1):
                goal[obj_id, bin_id, 0] = 1.0  # On relation
        
        return goal
    
    def _check_goal(self, current: np.ndarray, goal: np.ndarray) -> bool:
        """Check if current state matches goal predicates."""
        # Allow some tolerance
        diff = np.abs(current - goal)
        return np.mean(diff) < 0.1
    
    def _get_default_config(self) -> dict:
        """Return default planner config."""
        # TODO: Load from YAML config file
        return {
            'max_objects': 10,
            'node_emb_size': 256,
            'n_layers': 4,
            'n_heads': 4,
            # ... other config from Points2Plans
        }
```

---

## 5. Demo Script

### File: `robosuite/run_points2plans_planner.py`

End-to-end example.

```python
from robosuite.environments.base import make
from robosuite.planning.planning_controller import PlanningController

def main():
    # Create environment
    env = make(
        "PickPlaceMulti3",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "agentview"],
        control_freq=20,
    )
    
    # Initialize planner
    controller = PlanningController(
        env,
        checkpoint_path="Points2Plans/ckpt/checkpoint/cp_1.pth"
    )
    
    # Run episode
    success = controller.run_episode(
        goal_description="Put all objects in the bin",
        max_steps=50
    )
    
    print(f"Episode result: {'Success' if success else 'Failure'}")
    env.close()

if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### 1. Unit Tests

Test each component in isolation:

```bash
# Test state converter
python -m pytest robosuite/planning/tests/test_state_converter.py

# Test planner inference
python -m pytest robosuite/planning/tests/test_planner.py

# Test action executor
python -m pytest robosuite/planning/tests/test_executor.py
```

### 2. Integration Test

Test full pipeline:

```bash
python robosuite/run_points2plans_planner.py --debug
```

### 3. Validation

Compare with offline data:
- Load episode from dataset
- Run planner with same initial state
- Compare predicted actions with ground truth

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Tensor shape mismatch | Check batch dimension (should be 1 for inference) |
| CUDA out of memory | Use smaller point clouds or CPU inference |
| Action execution fails | Tune control gains, add error recovery |
| Planning too slow | Reduce num_samples, use GPU, cache embeddings |
| Predicted actions infeasible | Add constraints in action sampling |

---

## Performance Optimization

1. **Cache embeddings**: Don't recompute point cloud embeddings every step
2. **Parallel sampling**: Evaluate candidate actions in parallel
3. **Batch inference**: Process multiple candidates at once
4. **Point cloud compression**: Use fewer points if real-time is critical
5. **GPU utilization**: Move all tensors to GPU, minimize CPU-GPU transfers

---

## Next Steps

1. Start with `StateConverter` - reuse your data collection code
2. Test planner loading and inference independently
3. Implement simple action executor (can start with random actions)
4. Connect components in planning controller
5. Test on simplest task first (single object pick-place)
6. Gradually add complexity (multi-object, failures, replanning)

---

## Questions?

Common questions:

**Q: Do I need to retrain the model?**  
A: No, use your existing checkpoint. It's trained on your robosuite data.

**Q: What about the LLM module?**  
A: Optional. Start with hardcoded goals, add LLM later for flexibility.

**Q: How do I handle execution failures?**  
A: Replan with updated state. The planner is naturally robust to failures.

**Q: Can I use this for different tasks?**  
A: Yes! Just change the goal predicates. The model is task-agnostic.

**Q: What's the expected performance?**  
A: Depends on model quality. With your trained model, should match offline success rates.
