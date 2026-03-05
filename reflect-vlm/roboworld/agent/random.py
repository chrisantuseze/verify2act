import numpy as np

class RandomAgent(object):

    def __init__(self, seed=0):
        self.action_primitives = ["pick up", "put down", "insert", "reorient", "done"]
        np.random.seed(seed)

    def act(self, candidate_objects, image=None, goal_image=None, inp=None):
        act = np.random.choice(self.action_primitives)
        obj = np.random.choice(candidate_objects)
        return " ".join([act, obj])
