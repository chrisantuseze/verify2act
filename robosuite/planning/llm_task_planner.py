"""
LLM Task Planner for Points2Plans Integration - Clean Version

This is a thin wrapper that uses Points2Plans' existing BehaviorPromptManager
and PretrainedModelFactory, avoiding code duplication.

Usage:
    llm_planner = LLMTaskPlanner()
    
    goals, plans = llm_planner.generate_goals_and_plans(
        task_description="Stack all cubes",
        objects=["cubeA", "cubeB", "cubeC", "table"],
        initial_predicates=["On(cubeA, table)", "On(cubeB, table)", "On(cubeC, table)"]
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import ast

# Add Points2Plans to path (needs to be at Points2Plans root for LLM.fm_planning imports)
points2plans_path = Path(__file__).parent.parent.parent / "Points2Plans"
sys.path.insert(0, str(points2plans_path))

from LLM.fm_planning import models


class LLMTaskPlanner:
    """
    Thin wrapper around Points2Plans' existing LLM infrastructure.
    
    Leverages:
    - PretrainedModelFactory for model initialization
    - BehaviorPromptManager for prompt construction
    """
    
    def __init__(
        self,
        model_config_path: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        use_examples: bool = True
    ):
        """
        Initialize LLM task planner using Points2Plans infrastructure.
        
        Args:
            model_config_path: Path to model config (default: gpt_4_cot.yaml)
            prompt_config_path: Path to prompt config (default: stack_task.yaml)
            api_key: OpenAI API key (default: from environment)
            use_examples: Whether to use few-shot examples (default: True)
        """
        # Default paths
        if model_config_path is None:
            model_config_path = str(
                points2plans_path / "LLM/configs/models/pretrained/generative/gpt_4_cot.yaml"
            )
        
        if prompt_config_path is None:
            # Use the robosuite planning directory's config
            planning_dir = Path(__file__).parent
            prompt_config_path = str(planning_dir / "configs/prompts/tasks/stack_task.yaml")
        
        self.use_examples = use_examples
        self.prompt_config_path = prompt_config_path
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. LLM calls will fail.")
        
        # Initialize model using PretrainedModelFactory
        print(f"Loading model from: {model_config_path}")
        model_factory = models.PretrainedModelFactory(
            model_config_path,
            api_key=api_key,
            device="auto"
        )
        self.model = model_factory()
        
        print(f"LLM Task Planner initialized successfully")
        print(f"  Prompt config: {self.prompt_config_path}")
    
    def generate_goals_and_plans(
        self,
        task_description: str,
        objects: List[str],
        initial_predicates: List[str],
        additional_context: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Generate goals and plans for a task using the LLM.
        
        Args:
            task_description: Natural language task description (e.g., "Stack all cubes")
            objects: List of object names in the scene
            initial_predicates: List of initial state predicates
            additional_context: Optional additional context for the LLM
        
        Returns:
            goals: List of goal predicates (e.g., ["On(cubeA, cubeB)", "On(cubeB, cubeC)"])
            plans: List of primitive actions (e.g., ["Pick(cubeA, table)", "Place(cubeA, cubeB)"])
        """
        # Load prompt manager from YAML config
        prompt_manager = models.BehaviorPromptManager.from_yaml(self.prompt_config_path)
        
        # Update the task prompt with actual task details
        prompt_manager.task_prompt.instruction = task_description
        prompt_manager.task_prompt.objects = objects
        prompt_manager.task_prompt.predicates = initial_predicates

        # Query LLM using the model's forward method
        print(f"\n=== Querying LLM for task: {task_description} ===")
        print(f"Objects: {objects}")
        print(f"Initial predicates: {initial_predicates}")

        # Predict goals.
        goal_prediction_prompt = prompt_manager.generate_prompt(
            behavior="goal_prediction",
            use_examples=self.use_examples,
        )
        print(f"\n[Goal Prediction] Sending prompt to LLM...")
        response = self.model.forward(goal_prediction_prompt)
        response_text = response["choices"][0]["message"]["content"]
        print(f"Goal prediction response: {response_text[:200]}...")  # Debug: show first 200 chars
        
        # Parse goals with robust cleaning
        predicted_goals = self._parse_llm_list_response(response_text, "Goals:")
        print(f"Predicted goals: {predicted_goals}")

        # Update prompts with predicted goals.
        prompt_manager.task_prompt.goals = predicted_goals

        # Predict plans.
        task_planning_prompt = prompt_manager.generate_prompt(
            behavior="task_planning",
            use_examples=self.use_examples,
        )
        print(f"\n[Task Planning] Sending prompt to LLM...")
        response = self.model.forward(task_planning_prompt)
        response_text = response["choices"][0]["message"]["content"]
        print(f"Task planning response: {response_text[:200]}...")  # Debug: show first 200 chars
        
        # Parse plans with robust cleaning
        predicted_plans = self._parse_llm_list_response(response_text, "Plans:")
        print(f"Predicted plans: {predicted_plans}")
        
        # Extract the first goal and plan (assuming single goal for now)
        goals = predicted_goals[0] if predicted_goals else []
        plans = predicted_plans[0] if predicted_plans else []
        
        print(f"\n=== LLM Generation Complete ===")
        print(f"Selected goal: {goals}")
        print(f"Selected plan: {plans}")
        
        return goals, plans
    
    def _parse_llm_list_response(self, response_text: str, prefix: str) -> list:
        """
        Parse LLM response that should contain a list.
        Handles various formats:
        - "Goals: [[...]]"
        - "Goals:\n[[...]]"
        - "Goals:\n- [...]"
        - Just "[[...]]"
        
        Args:
            response_text: Raw LLM response
            prefix: Expected prefix (e.g., "Goals:", "Plans:")
        
        Returns:
            Parsed list
        """
        # Extract content after prefix if present
        if prefix in response_text:
            content = response_text.split(prefix, 1)[1].strip()
        else:
            content = response_text.strip()
        
        # Remove markdown formatting (bullet points, etc.)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Remove markdown bullet points
            if line.startswith('- '):
                line = line[2:].strip()
            elif line.startswith('* '):
                line = line[2:].strip()
            # Skip empty lines
            if line:
                cleaned_lines.append(line)
        
        # Rejoin and try to parse
        cleaned_content = '\n'.join(cleaned_lines)
        
        # If it's already a proper list format, use it
        if cleaned_content.startswith('['):
            try:
                return ast.literal_eval(cleaned_content)
            except Exception as e:
                print(f"Warning: Failed to parse as literal: {e}")
        
        # Try to extract list from text
        # Look for patterns like [[...], [...]]
        import re
        list_pattern = r'\[\[.*?\]\]'
        match = re.search(list_pattern, cleaned_content, re.DOTALL)
        if match:
            try:
                return ast.literal_eval(match.group(0))
            except Exception as e:
                print(f"Warning: Failed to parse matched pattern: {e}")
        
        # Last resort: try to parse the whole cleaned content
        try:
            return ast.literal_eval(cleaned_content)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response_text[:500]}...")
            print(f"Cleaned content: {cleaned_content[:500]}...")
            
            # Check if LLM is refusing or explaining
            refusal_keywords = ["not possible", "cannot generate", "unable to", "no predicates", "don't include"]
            if any(keyword in response_text.lower() for keyword in refusal_keywords):
                print("\n⚠️  LLM refused to generate response - likely due to invalid initial predicates")
                print("Returning empty list as fallback")
                return []
            
            raise ValueError(f"Failed to parse LLM response after '{prefix}'")
    
    def goals_to_predicates(
        self,
        goals: List[str],
        object_name_to_id: Dict[str, int],
        num_objects: int,
        num_predicates: int = 9
    ) -> np.ndarray:
        """
        Convert goal strings to predicate matrix.
        
        Args:
            goals: List of goal predicate strings (e.g., ["On(cubeA, cubeB)"])
            object_name_to_id: Mapping from object names to indices
            num_objects: Number of objects in the scene
            num_predicates: Number of predicate types (default: 9 for full predicate set)
        
        Returns:
            predicate_matrix: Binary matrix [num_objects, num_objects, num_predicates]
        """
        predicates = np.zeros((num_objects, num_objects, num_predicates))

        print(f"    goals: {goals}")
        
        for goal_str in goals:
            try:
                # Parse goal string (e.g., "On(cubeA, cubeB)" or "Stacked(cubeA, cubeB)")
                pred_type, content = goal_str.split('(')
                content = content.rstrip(')')
                obj1, obj2 = [x.strip() for x in content.split(',')]
                
                # Get object indices
                obj1_idx = object_name_to_id.get(obj1)
                obj2_idx = object_name_to_id.get(obj2)
                # print(f"  Parsing goal: {goal_str} -> pred_type: {pred_type}, obj1: {obj1} (idx {obj1_idx}), obj2: {obj2} (idx {obj2_idx})")
                
                if obj1_idx is None or obj2_idx is None:
                    print(f"Warning: Object not found in mapping for goal '{goal_str}'")
                    print(f"  Available objects: {list(object_name_to_id.keys())}")
                    continue
                
                # Get predicate type index
                pred_idx = self._predicate_type_to_idx(pred_type.strip(), num_predicates)
                
                # Set the predicate (obj1 has relation to obj2)
                predicates[obj1_idx, obj2_idx, pred_idx] = 1.0
                
                # print(f"  Set goal predicate: {pred_type}({obj1}[{obj1_idx}], {obj2}[{obj2_idx}])")
                
            except Exception as e:
                print(f"Warning: Failed to parse goal '{goal_str}': {e}")
                continue
        
        # print(f"    Generated predicate matrix:\n{predicates}")
        return predicates
    
    def _predicate_type_to_idx(self, pred_type: str, num_predicates: int) -> int:
        """Map predicate type string to index."""
        pred_type_lower = pred_type.lower()
        
        # Map to 9-predicate system used by decoder
        # Order: On, Inside, Left, Right, Front, Behind, Near, Touching, Grasped
        predicate_map = {
            'on': 0,
            'stacked': 0,  # Stacked is equivalent to On for our purposes
            'inside': 1,
            'in': 1,  # Alias for Inside
            'left': 2,
            'leftof': 2,
            'right': 3,
            'rightof': 3,
            'front': 4,
            'infront': 4,
            'behind': 5,
            'near': 6,
            'close': 6,  # Alias for Near
            'touching': 7,
            'touch': 7,
            'grasped': 8,
            'holding': 8  # Alias for Grasped
        }
        
        if pred_type_lower in predicate_map:
            idx = predicate_map[pred_type_lower]
            # Ensure index is within bounds
            if idx < num_predicates:
                return idx
        
        print(f"Warning: Unknown predicate type '{pred_type}', defaulting to 0")
        return 0
    
    def plans_to_primitives(self, plans: List[str]) -> List[Dict]:
        """
        Convert plan strings to primitive action dictionaries.
        
        Args:
            plans: List of plan strings (e.g., ["Pick(cubeA, table)", "Place(cubeA, cubeB)"])
        
        Returns:
            primitives: List of primitive action dicts with 'type', 'object', 'target'
        """
        primitives = []
        
        for plan_str in plans:
            try:
                # Parse plan string
                action_type, content = plan_str.split('(')
                content = content.rstrip(')')
                parts = [x.strip() for x in content.split(',')]
                
                primitive = {
                    'type': action_type.strip(),
                    'object': parts[0] if len(parts) > 0 else None,
                    'target': parts[1] if len(parts) > 1 else None,
                    'raw': plan_str
                }
                
                primitives.append(primitive)
                
            except Exception as e:
                print(f"Warning: Failed to parse plan '{plan_str}': {e}")
                continue
        
        return primitives


def test_llm_planner():
    """Quick test of the LLM planner."""
    planner = LLMTaskPlanner()
    
    # Test task
    task = "Stack cubeA on cubeB"
    objects = ["cubeA", "cubeB", "table"]
    initial_state = ["On(cubeA, table)", "On(cubeB, table)"]
    
    goals, plans = planner.generate_goals_and_plans(
        task_description=task,
        objects=objects,
        initial_predicates=initial_state
    )
    
    print("\nGenerated Goals:")
    for g in goals:
        print(f"  {g}")
    
    print("\nGenerated Plans:")
    for p in plans:
        print(f"  {p}")
    
    return goals, plans


if __name__ == "__main__":
    test_llm_planner()
