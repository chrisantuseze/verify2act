import logging
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

from calvin_agent.utils.utils import get_last_checkpoint
from calvin_agent.evaluation.utils import get_default_model_and_env

logger = logging.getLogger(__name__)


class MCILLowLevelPolicy:
    """Wrapper for the pre-trained CALVIN baseline policy (MCIL/HULC)."""
    
    def __init__(self, train_folder: str, dataset_path: str, device: torch.device, extra_dataset_path: str = None):
        logger.info(f"Loading MCIL baseline policy from {train_folder}...")
        
        train_folder_path = Path(train_folder)
        checkpoint = get_last_checkpoint(train_folder_path)
        
        # Load the model using CALVIN's built-in loader
        self.model, self.env, self.data_module = get_default_model_and_env(
            train_folder=train_folder,
            dataset_path=dataset_path,
            checkpoint=checkpoint,
            device_id=device.index if device.index is not None else 0,
            extra_dataset_path=extra_dataset_path,
        )
        self.model.eval()
        logger.info("MCIL baseline policy loaded successfully.")
        
    def reset(self):
        self.model.reset()
        
    def step(self, obs: Dict[str, np.ndarray], text_instruction: str) -> np.ndarray:
        # Returns a 7-DoF continuous action
        action = self.model.step(obs, text_instruction)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action

    def propose_trajectory(self, obs: Dict[str, np.ndarray], text_instruction: str, steps: int = 10) -> List[np.ndarray]:
        """
        Propose a sequence of actions to achieve the instruction.
        Note: MCIL is autoregressive on observations. It's difficult to propose a true
        closed-loop trajectory without stepping the environment.
        """
        return [self.step(obs, text_instruction) for _ in range(steps)]


class MoDEPolicyWrapper:
    """Wrapper for the pre-trained CALVIN MoDE Diffusion Policy."""
    
    def __init__(self, checkpoint_path: str, dataset_path: str, device: torch.device, extra_dataset_path: str = None):
        logger.info(f"Loading MoDE Policy from {checkpoint_path}...")
        
        # Append MoDE repository to PYTHONPATH so we can import it
        mode_repo_path = str(Path(__file__).resolve().parents[2] / "third_party" / "MoDE_Diffusion_Policy")
        if mode_repo_path not in sys.path:
            sys.path.insert(0, mode_repo_path)
            
        from mode.evaluation.utils import get_default_mode_and_env
        
        # The MoDE loader expects the train_folder to be the parent and checkpoint to be the dir name.
        checkpoint_dir = Path(checkpoint_path)
        train_folder = str(checkpoint_dir.parent)
        checkpoint_name = checkpoint_dir.name
        
        # Determine device ID for MoDE
        device_id = device.index if device.index is not None else 0
        if device.type == "cpu":
            device_id = "cpu"
            
        model, env, data_module, lang_embeddings = get_default_mode_and_env(
            train_folder=train_folder,
            dataset_path=dataset_path,
            checkpoint=checkpoint_name,
            env=None,
            lang_embeddings=None,
            device_id=device_id,
        )
        model = model.to(device)
        model.eval()
        
        # Save env, data_module and dataset info
        self.model = model
        self.env = env
        self.data_module = data_module
        
        dataloader = data_module.val_dataloader()
        self.dataset = dataloader["lang"].dataset
        
        # Use MoDE's own wrapper to handle preprocessing and postprocessing
        from mode.evaluation.agent_proxy import CalvinAgentWrapper
        self.agent = CalvinAgentWrapper(
            self.model,
            self.dataset.observation_space,
            self.dataset.proprio_state,
            self.dataset.transforms
        )
        logger.info("MoDE Policy loaded successfully.")

    def reset(self):
        self.agent.reset()

    def step(self, obs: Dict[str, np.ndarray], text_instruction: str) -> np.ndarray:
        # MoDE step returns a 7-DoF continuous action
        action = self.agent.step(obs, text_instruction)
        return action

    def propose_trajectory(self, obs: Dict[str, np.ndarray], text_instruction: str, steps: int = 10) -> List[np.ndarray]:
        # Propose a sequence of actions by rolling out the policy
        from optree import tree_map
        try:
            obs_transformed = self.agent._transform_observation(obs)
            obs_transformed = tree_map(lambda x: x.to(self.model.device), obs_transformed)
            with torch.no_grad():
                output = self.model(obs_transformed, {"lang_text": text_instruction})
                # MoDE policy output is usually action tensor of shape (B, Horizon, Action_dim)
                if isinstance(output, torch.Tensor):
                    action_pred = output[0].cpu().numpy()
                elif isinstance(output, dict) and 'action' in output:
                    action_pred = output['action'][0].cpu().numpy()
                else:
                    action_pred = output[0].cpu().numpy()
                
                trajectory = []
                for a in action_pred[:steps]:
                    trajectory.append(self.agent._transform_action(torch.from_numpy(a).unsqueeze(0).to(self.model.device)))
                return trajectory
        except Exception as e:
            logger.error(f"MoDE propose_trajectory failed: {e}")
            # Fallback to step repetition
            return [self.step(obs, text_instruction) for _ in range(steps)]


class LowLevelPolicyFactory:
    @staticmethod
    def get_policy(policy_type: str, train_folder: str, dataset_path: str, device: torch.device, extra_dataset_path: str = None, **kwargs):
        if policy_type.lower() == "hulc":
            return MCILLowLevelPolicy(train_folder, dataset_path, device, extra_dataset_path)
        elif policy_type.lower() in ["diffusion", "mode"]:
            return MoDEPolicyWrapper(train_folder, dataset_path, device, extra_dataset_path)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
