"""
Policy Adapters for Data Collection

Provides a unified interface for different heuristic policies to work with BatchCollector.
Each adapter ensures the policy exposes:
- step() -> (action, done) 
- obs attribute for state tracking
"""

import numpy as np
from typing import Tuple, Dict, Any


class PolicyAdapter:
    """Base class for policy adapters."""
    
    def __init__(self, policy):
        self.policy = policy
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """Execute one step and return (action, done)."""
        raise NotImplementedError
    
    @property
    def obs(self):
        """Get current observations."""
        return self.policy.obs
    
    @obs.setter
    def obs(self, value):
        """Update observations."""
        self.policy.obs = value


class StackPolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicStackPolicy."""
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the stack policy.
        
        Returns:
            (action, done): Action array and completion flag
        """
        action, task_done = self.policy.step()
        
        # Check if done (all pairs stacked or stage is "done")
        done = (task_done or self.policy.stage == "done" or 
                self.policy.pair_idx >= len(self.policy.stacking_pairs))
        
        return action, done


class NutAssemblyPolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicNutAssemblyPolicy (cluttered nut assembly)."""
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the nut assembly policy.
        
        Returns:
            (action, done): Action array and completion flag
        """
        action, done = self.policy.step()
        return action, done


class PickPlacePolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicPickPlacePolicy."""
    
    def step(self) -> Tuple[np.ndarray, bool]:
        """
        Execute one step of the pick-place policy.
        
        Returns:
            (action, done): Action array and completion flag
        """
        action, done = self.policy.step()
        return action, done


# Factory functions for creating adapted policies

def create_stack_policy(env, data_collection_mode=True):
    """Create adapted stack policy."""
    from run_stack import HeuristicStackPolicy
    policy = HeuristicStackPolicy(env)
    return StackPolicyAdapter(policy)


def create_nut_assembly_policy(env, data_collection_mode=True):
    """Create adapted cluttered nut assembly policy."""
    from run_cluttered_nutassembly import HeuristicNutAssemblyPolicy
    policy = HeuristicNutAssemblyPolicy(env, data_collection_mode=data_collection_mode)
    return NutAssemblyPolicyAdapter(policy)


def create_pickplace_policy(env, data_collection_mode=True):
    """Create adapted pick-place policy."""
    from run_pickplace import HeuristicPickPlacePolicy
    policy = HeuristicPickPlacePolicy(env)
    return PickPlacePolicyAdapter(policy)


# Policy factory registry
POLICY_FACTORIES = {
    'stack': create_stack_policy,
    'nut_assembly': create_nut_assembly_policy,
    'pickplace': create_pickplace_policy,
}


def get_policy_factory(policy_name: str):
    """
    Get policy factory by name.
    
    Args:
        policy_name: Name of the policy ('stack', 'nut_assembly', 'pickplace')
        
    Returns:
        Policy factory function
    """
    if policy_name not in POLICY_FACTORIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICY_FACTORIES.keys())}")
    
    return POLICY_FACTORIES[policy_name]
