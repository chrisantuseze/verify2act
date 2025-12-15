"""
LLM Task Planner for Points2Plans

Wraps the existing Points2Plans LLM module for task decomposition and goal generation.

The LLM module (Points2Plans/LLM/) is a REQUIRED core component that:
1. Takes natural language task descriptions
2. Generates structured goal predicates
3. Produces high-level action plans

This happens ONCE at episode start (or when task changes).
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import ast

# Add Points2Plans to path
points2plans_path = Path(__file__).parent.parent.parent / "Points2Plans"
sys.path.insert(0, str(points2plans_path))

from LLM.fm_planning import models, utils


class LLMTaskPlanner:
    """
    Wrapper for Points2Plans LLM module for task decomposition.
    
    Two-stage LLM approach:
    1. Goal Prediction: Natural language -> Goal predicates
    2. Task Planning: Task + Goals -> High-level action sequence
    
    Uses BehaviorPromptManager for YAML-based prompt configuration with few-shot examples.
    """
    
    def __init__(self, 
                 model_config_path: Optional[str] = None,
                 prompt_config_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 device: str = "auto",
                 use_examples: bool = True):
        """
        Initialize LLM task planner.
        
        Args:
            model_config_path: Path to LLM model config 
                             (e.g., Points2Plans/LLM/configs/models/pretrained/generative/gpt4.yaml)
            prompt_config_path: Path to prompt YAML config
                              (e.g., Points2Plans/LLM/configs/prompts/examples/example_1.yaml)
                              If None, uses internal fallback prompts
            api_key: OpenAI API key (or None to use environment variable)
            device: Device for model (usually "auto")
            use_examples: Whether to use few-shot examples from YAML config
        """
        # Default to GPT-4 config if not specified
        if model_config_path is None:
            model_config_path = str(
                points2plans_path / "LLM/configs/models/pretrained/generative/gpt4.yaml"
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Load LLM model
        self.model_factory = models.PretrainedModelFactory(
            model_config_path,
            api_key=api_key,
            device=device
        )
        self.model = self.model_factory()
        
        # Load prompt manager if YAML config provided
        self.use_yaml_prompts = prompt_config_path is not None
        self.use_examples = use_examples
        
        if self.use_yaml_prompts:
            self.prompt_manager = models.BehaviorPromptManager.from_yaml(prompt_config_path)
        else:
            self.prompt_manager = None
        
        print(f"LLM Task Planner initialized:")
        print(f"  Model config: {model_config_path}")
        print(f"  Prompt config: {prompt_config_path if prompt_config_path else 'Using fallback prompts'}")
        print(f"  API key: {'set' if api_key else 'NOT SET'}")
        print(f"  Use examples: {use_examples}")
    
    def generate_goals_and_plans(self,
                                 task_description: str,
                                 objects: List[str],
                                 initial_predicates: Optional[List[str]] = None) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generate goal predicates and high-level action plans from natural language.
        
        Uses Points2Plans' two-stage LLM approach:
        1. Goal Prediction: Task description -> Goal predicates
        2. Task Planning: Task + Goals -> Action sequence
        
        Args:
            task_description: Natural language task (e.g., "Put all objects in the bin")
            objects: List of object names in scene (e.g., ["milk", "cereal", "bread", "bin"])
            initial_predicates: Current scene predicates (e.g., ["On(milk, table)", "On(cereal, table)"])
        
        Returns:
            goals: List of goal predicate lists (e.g., [[\"On(milk, bin)\", \"On(cereal, bin)\"]])
            plans: List of action sequence lists (e.g., [[\"Pick(milk, table)\", \"Place(milk, bin)\", ...]])
        """
        if initial_predicates is None:
            initial_predicates = []
        
        # Update prompt manager with current task if using YAML prompts
        if self.use_yaml_prompts and self.prompt_manager is not None:
            self.prompt_manager.task_prompt.instruction = task_description
            self.prompt_manager.task_prompt.objects = objects
            self.prompt_manager.task_prompt.predicates = initial_predicates
        
        # Stage 1: Goal Prediction
        if self.use_yaml_prompts and self.prompt_manager is not None:
            goal_prompt = self.prompt_manager.generate_prompt(
                behavior="goal_prediction",
                use_examples=self.use_examples
            )
        else:
            goal_prompt = self._create_goal_prediction_prompt(
                task_description,
                objects,
                initial_predicates
            )
        
        try:
            response = self.model.forward(goal_prompt)
            predicted_goals = ast.literal_eval(response["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error in goal prediction: {e}")
            print(f"Response: {response}")
            # Fallback: simple heuristic
            predicted_goals = self._fallback_goal_prediction(task_description, objects)
        
        # Update prompt manager with predicted goals for task planning
        if self.use_yaml_prompts and self.prompt_manager is not None:
            self.prompt_manager.task_prompt.goals = predicted_goals
        
        # Stage 2: Task Planning
        if self.use_yaml_prompts and self.prompt_manager is not None:
            plan_prompt = self.prompt_manager.generate_prompt(
                behavior="task_planning",
                use_examples=self.use_examples
            )
        else:
            plan_prompt = self._create_task_planning_prompt(
                task_description,
                objects,
                initial_predicates,
                predicted_goals
            )
        
        try:
            response = self.model.forward(plan_prompt)
            predicted_plans = ast.literal_eval(response["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error in task planning: {e}")
            print(f"Response: {response}")
            # Fallback: simple heuristic
            predicted_plans = self._fallback_plan_generation(predicted_goals, objects)
        
        print(f"\nLLM Generated Goals: {predicted_goals}")
        print(f"LLM Generated Plans: {predicted_plans}")
        
        return predicted_goals, predicted_plans
    
    def _create_goal_prediction_prompt(self,
                                      task_description: str,
                                      objects: List[str],
                                      initial_predicates: List[str]) -> Dict:
        """Create prompt for goal prediction (Stage 1)."""
        system_msg = """You are a robotic task planning assistant. Given a task description and objects, predict the goal state as a list of spatial predicates.

Available predicates:
- On(object, surface): object is resting on top of surface
- Inside(object, container): object is inside container
- Near(object1, object2): objects are close to each other

Return ONLY a Python list of lists containing goal predicates. Example: [["On(milk, bin)", "On(cereal, bin)"]]"""
        
        user_msg = f"""Task: {task_description}
Objects: {', '.join(objects)}
Initial state: {', '.join(initial_predicates) if initial_predicates else 'Not specified'}

Predict the goal state as a list of lists of predicates:"""
        
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
                                    goals: List[str]) -> Dict:
        """Create prompt for task planning (Stage 2)."""
        system_msg = """You are a robotic task planning assistant. Given a task, objects, and goals, generate a sequence of actions to achieve the goals.

Available actions:
- Pick(object, location): Pick up object from location
- Place(object, location): Place object at location
- Push(object, direction): Push object in direction (if needed)

Return ONLY a Python list of lists containing action sequences. Example: [["Pick(milk, table)", "Place(milk, bin)", "Pick(cereal, table)", "Place(cereal, bin)"]]"""
        
        user_msg = f"""Task: {task_description}
Objects: {', '.join(objects)}
Initial state: {', '.join(initial_predicates) if initial_predicates else 'Not specified'}
Goals: {', '.join(goals[0]) if goals else 'None'}

Generate action plan as a list of lists:"""
        
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        }
    
    def _fallback_goal_prediction(self, task_description: str, objects: List[str]) -> List[List[str]]:
        """
        Fallback heuristic for goal prediction if LLM fails.
        
        Simple heuristics:
        - "put X in Y" -> On(X, Y)
        - "put all objects in Y" -> On(obj, Y) for all non-container objects
        """
        task_lower = task_description.lower()
        goals = []
        
        # Find target container (bin, cupboard, etc.)
        target = None
        for obj in objects:
            if any(keyword in obj.lower() for keyword in ['bin', 'cupboard', 'shelf', 'container']):
                target = obj
                break
        
        if target:
            # Put all non-container objects on/in target
            for obj in objects:
                if obj != target and not any(keyword in obj.lower() for keyword in ['bin', 'cupboard', 'shelf', 'table']):
                    if 'in' in task_lower or 'inside' in task_lower:
                        goals.append(f"Inside({obj}, {target})")
                    else:
                        goals.append(f"On({obj}, {target})")
        
        return [goals] if goals else [[]]
    
    def _fallback_plan_generation(self, goals: List[List[str]], objects: List[str]) -> List[List[str]]:
        """
        Fallback heuristic for plan generation if LLM fails.
        
        Simple: For each goal On(X, Y), generate Pick(X, current_loc), Place(X, Y)
        """
        plans = []
        
        if not goals or not goals[0]:
            return [[]]
        
        for goal in goals[0]:
            # Parse goal: "On(milk, bin)" -> ["milk", "bin"]
            if '(' in goal and ')' in goal:
                content = goal.split('(')[1].split(')')[0]
                parts = [p.strip() for p in content.split(',')]
                
                if len(parts) == 2:
                    obj, target = parts
                    # Assume objects start on table
                    plans.append(f"Pick({obj}, table)")
                    plans.append(f"Place({obj}, {target})")
        
        return [plans] if plans else [[]]
    
    def goals_to_predicates(self,
                           goals: List[str],
                           object_name_to_id: Dict[str, int],
                           num_objects: int) -> np.ndarray:
        """
        Convert goal strings to predicate tensor for dynamics model.
        
        Args:
            goals: List of goal strings (e.g., ["On(milk, bin)", "Inside(cereal, cupboard)"])
            object_name_to_id: Mapping from object names to integer IDs
            num_objects: Total number of objects in scene
        
        Returns:
            goal_predicates: [num_objects, num_objects, num_predicates] tensor
                           Predicates: [On, Inside, Graspable, ...]
        """
        # Initialize predicate tensor (On, Inside, Graspable, etc.)
        # Based on Points2Plans: typically 3-6 predicate types
        num_predicates = 3  # On, Inside, Graspable (extend as needed)
        goal_predicates = np.zeros((num_objects, num_objects, num_predicates))
        
        for goal in goals:
            # Parse goal string
            if '(' not in goal or ')' not in goal:
                continue
            
            # Extract predicate type and arguments
            predicate_type = goal.split('(')[0].strip()
            content = goal.split('(')[1].split(')')[0]
            parts = [p.strip() for p in content.split(',')]
            
            # Map predicate type to index
            predicate_idx = self._predicate_type_to_idx(predicate_type)
            if predicate_idx is None:
                continue
            
            # Map object names to IDs
            if len(parts) == 2:
                obj1_name, obj2_name = parts
                if obj1_name in object_name_to_id and obj2_name in object_name_to_id:
                    obj1_id = object_name_to_id[obj1_name]
                    obj2_id = object_name_to_id[obj2_name]
                    goal_predicates[obj1_id, obj2_id, predicate_idx] = 1.0
            elif len(parts) == 1:
                # Unary predicate (e.g., Graspable(milk))
                obj_name = parts[0]
                if obj_name in object_name_to_id:
                    obj_id = object_name_to_id[obj_name]
                    # For unary predicates, set diagonal or specific pattern
                    goal_predicates[obj_id, obj_id, predicate_idx] = 1.0
        
        return goal_predicates
    
    def _predicate_type_to_idx(self, predicate_type: str) -> Optional[int]:
        """Map predicate type string to index."""
        predicate_map = {
            'On': 0,
            'Above': 0,  # Alias for On
            'Inside': 1,
            'In': 1,  # Alias for Inside
            'Graspable': 2,
            'CanGrasp': 2,  # Alias
        }
        return predicate_map.get(predicate_type, None)
