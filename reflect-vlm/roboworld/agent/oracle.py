import numpy as np
from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle

class OracleAgent(object):

    def __init__(self, oracle: AssemblyOracle):
        self.oracle = oracle
        self.action_primitives = ["pick up", "put down", "insert", "reorient", "done"]

    def act(self, image=None, goal_image=None, inp=None):
        self.oracle.update_state_from_env()
        oracle_action = self.oracle.get_action()
        return self.convert_action(oracle_action)
    
    def get_all_feasible_actions(self, image=None, goal_image=None, inp=None):
        self.oracle.update_state_from_env()
        all_feasible_actions = self.oracle.get_all_feasible_actions()
        return [self.convert_action(a) for a in all_feasible_actions]

    def convert_action(self, act_desc):
        if act_desc == "done":
            return act_desc
        act, obj = None, None
        for a in self.action_primitives:
            if a in act_desc:
                act = a
        obj = act_desc.split()[-2]
        return " ".join([act, obj])
