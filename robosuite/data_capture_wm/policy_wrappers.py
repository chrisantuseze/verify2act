"""
Policy Adapters for Data Collection

Provides a unified interface for different heuristic policies to work with
BatchCollector.  Each adapter exposes:

    step()            -> (action, done)
    get_action_info() -> ActionInfo(skill, object_name, cartesian_target)
    obs               -> current observation dict  (read/write)

The stage-to-skill mapping lives *here* so that batch_collect.py never needs
to inspect policy internals.  Adding a new task means adding one adapter
subclass with its own _STAGE_SKILL map — nothing else in the pipeline changes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────── shared dataclass ─────────────────────────── #

@dataclass
class ActionInfo:
    """Action metadata returned by every adapter."""
    skill: str                      # "pick" | "place" | "insert"
    object_name: str                # e.g. "round nut", "cubeA", "bread"
    cartesian_target: np.ndarray    # shape (3,), world-frame metres
    stage: str                      # raw policy stage name
    event_tag: Optional[str] = None # optional keyframe marker, e.g. "pick_start"


# ─────────────────────────── base class ───────────────────────────────── #

class PolicyAdapter:
    """Base class for policy adapters."""

    def __init__(self, policy):
        self.policy = policy
        self._prev_skill: str = "pick"     # safe default
        self._prev_stage: Optional[str] = None

    def _event_on_stage_entry(self, stage: str, stage_to_event: dict[str, str]) -> Optional[str]:
        event_tag = None
        if stage != self._prev_stage:
            event_tag = stage_to_event.get(stage)
        self._prev_stage = stage
        return event_tag

    def step(self) -> Tuple[np.ndarray, bool]:
        """Execute one policy step.  Returns (action, done)."""
        raise NotImplementedError

    def get_action_info(self) -> ActionInfo:
        """Return current (skill, object_name, cartesian_target).

        Must be called *after* step() so the policy's internal stage is
        up-to-date for the current timestep.
        """
        raise NotImplementedError

    # ── observation pass-through ──

    @property
    def obs(self):
        return self.policy.obs

    @obs.setter
    def obs(self, value):
        self.policy.obs = value


# ─────────────────────────── NutAssembly ──────────────────────────────── #

class NutAssemblyPolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicNutAssemblyPolicy (ClutteredNutAssembly)."""

    # Stage → skill mapping.
    # Transitional stages (release, retract, …) inherit the previous skill
    # so that the prompt stays consistent across the manipulation phase.
    _PICK_STAGES = frozenset({
        "move_to_nut", "lower_to_nut", "grasp", "verify_grasp", "lift_nut",
    })
    _INSERT_STAGES = frozenset({
        "move_to_peg", "align_over_peg", "lower_to_peg",
    })
    _PLACE_STAGES = frozenset({
        "move_to_table", "lower_to_table",
    })

    _EVENT_TAG_BY_STAGE = {
        "move_to_nut": "pick_start",
        "move_to_peg": "pick_end",
        "align_over_peg": "insert_start",
        "release": "insert_end",
        "move_to_table": "place_start",
        "lower_to_table": "place_end",
    }

    def step(self) -> Tuple[np.ndarray, bool]:
        action, done = self.policy.step()
        return action, done

    def get_action_info(self) -> ActionInfo:
        p = self.policy
        stage = getattr(p, "stage", "")

        # ── skill ──
        if stage in self._PICK_STAGES:
            skill = "pick"
        elif stage in self._INSERT_STAGES:
            skill = "insert"
        elif stage in self._PLACE_STAGES:
            skill = "place"
        else:
            # Transitional stages: release, retract, reset_orientation,
            # skip_nut — inherit whatever we were doing.
            skill = self._prev_skill
        self._prev_skill = skill

        # ── object name ──
        raw = getattr(p, "current_nut", "") or ""
        raw = raw.lower()
        if "round" in raw:
            object_name = "round nut"
        elif "square" in raw:
            object_name = "square nut"
        elif raw:
            object_name = raw.replace("_", " ")
        else:
            object_name = "nut"

        # ── cartesian target (nut world position) ──
        cartesian_target = np.zeros(3)
        nut_attr = getattr(p, "current_nut", None)
        if nut_attr and p.obs is not None:
            for key in (f"{nut_attr}_pos",
                        f"{nut_attr.capitalize()}_pos",
                        f"{nut_attr.lower()}_pos"):
                if key in p.obs:
                    cartesian_target = np.asarray(p.obs[key][:3], dtype=np.float64)
                    break
            else:
                for key in ("robot0_eef_pos", "eef_pos"):
                    if key in p.obs:
                        cartesian_target = np.asarray(p.obs[key][:3], dtype=np.float64)
                        break

        return ActionInfo(
            skill=skill,
            object_name=object_name,
            cartesian_target=cartesian_target,
            stage=stage,
            event_tag=self._event_on_stage_entry(stage, self._EVENT_TAG_BY_STAGE),
        )


# ─────────────────────────── Stack ────────────────────────────────────── #

class StackPolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicStackPolicy."""

    _PICK_STAGES = frozenset({
        "move_to_cube", "lower_to_cube", "grasp", "lift_cube",
        "move_horizontal_to_next",
    })
    _PLACE_STAGES = frozenset({
        "move_above_target", "lower_to_target",
    })

    _EVENT_TAG_BY_STAGE = {
        "move_to_cube": "pick_start",
        "move_above_target": "pick_end|place_start",
        "retract": "place_end",
    }

    def step(self) -> Tuple[np.ndarray, bool]:
        action, task_done = self.policy.step()
        done = (task_done or self.policy.stage == "done"
                or self.policy.pair_idx >= len(self.policy.stacking_pairs))
        return action, done

    def get_action_info(self) -> ActionInfo:
        p = self.policy
        stage = getattr(p, "stage", "")

        # ── skill ──
        if stage in self._PICK_STAGES:
            skill = "pick"
        elif stage in self._PLACE_STAGES:
            skill = "place"
        else:
            skill = self._prev_skill
        self._prev_skill = skill

        # ── object name  (source cube of current pair) ──
        idx = min(p.pair_idx, len(p.stacking_pairs) - 1)
        source_name, _ = p.stacking_pairs[idx]
        object_name = source_name          # e.g. "cubeA"

        # ── cartesian target (source cube position) ──
        cartesian_target = np.zeros(3)
        key = f"{source_name}_pos"
        if p.obs is not None and key in p.obs:
            cartesian_target = np.asarray(p.obs[key][:3], dtype=np.float64)

        return ActionInfo(
            skill=skill,
            object_name=object_name,
            cartesian_target=cartesian_target,
            stage=stage,
            event_tag=self._event_on_stage_entry(stage, self._EVENT_TAG_BY_STAGE),
        )


# ─────────────────────────── PickPlace ────────────────────────────────── #

class PickPlacePolicyAdapter(PolicyAdapter):
    """Adapter for HeuristicPickPlacePolicy."""

    _PICK_STAGES = frozenset({
        "move_to_object", "align_for_side_grasp", "approach_from_side",
        "lower_to_object", "grasp", "verify_grasp", "lift_object",
    })
    _PLACE_STAGES = frozenset({
        "move_to_bin", "lower_to_bin",
    })

    def step(self) -> Tuple[np.ndarray, bool]:
        action, done = self.policy.step()
        return action, done

    def get_action_info(self) -> ActionInfo:
        p = self.policy
        stage = getattr(p, "stage", "")

        # ── skill ──
        if stage in self._PICK_STAGES:
            skill = "pick"
        elif stage in self._PLACE_STAGES:
            skill = "place"
        else:
            skill = self._prev_skill
        self._prev_skill = skill

        # ── object name ──
        object_name = getattr(p, "current_object", "object") or "object"

        # ── cartesian target ──
        cartesian_target = np.zeros(3)
        if p.obs is not None and object_name in p.obs:
            cartesian_target = np.asarray(p.obs[object_name][:3], dtype=np.float64)

        return ActionInfo(
            skill=skill,
            object_name=object_name,
            cartesian_target=cartesian_target,
            stage=stage,
            event_tag=None,
        )


# ─────────────────────────── factories ────────────────────────────────── #

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
    """Get policy factory by name."""
    if policy_name not in POLICY_FACTORIES:
        raise ValueError(
            f"Unknown policy: {policy_name}. "
            f"Available: {list(POLICY_FACTORIES.keys())}"
        )
    return POLICY_FACTORIES[policy_name]
