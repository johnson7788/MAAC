from utils.core import *
import random


class SimpleEnv:
    def __init__(self):
        self.desired_actions: Dict[AgentKey, int] = {}

    @property
    def config(self):
        return Config(reward_scale=100,
                      pol_hidden_dim=128,
                      critic_hidden_dim=128)

    @property
    def agent_type_topologies(self):
        return [(5, 5), (5, 4)]

    def reset(self):
        pass

    def next(self, actions: Dict[AgentKey, AgentAction]) -> Dict[AgentKey, AgentResponseFrame]:
        results: Dict[AgentKey, AgentResponseFrame] = {}
        new_desired_actions: Dict[AgentKey, int] = {}
        for i in range(3):
            rand_action = random.randint(0, 4)
            key = AgentKey(0, '0-' + str(i))
            results[key] = AgentResponseFrame(0, False, [1 if j == rand_action else 0 for j in range(5)])
            new_desired_actions[key] = rand_action

        for i in range(2):
            rand_action = random.randint(0, 3)
            key = AgentKey(1, '0-' + str(i))
            results[key] = AgentResponseFrame(0, False, [1 if j == rand_action else 0 for j in range(5)])
            new_desired_actions[key] = rand_action

        if actions is not None:
            for k, v in actions.items():
                results[k].reward = 1 if v.get_action_index() == self.desired_actions[k] else 0

        self.desired_actions = new_desired_actions

        return results
