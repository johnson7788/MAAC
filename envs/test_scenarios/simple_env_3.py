from utils.core import *
import random


def randval():
    return random.randrange(0., 1.)


class SimpleEnv:
    @property
    def config(self):
        return Config(reward_scale=100,
                      pol_hidden_dim=128,
                      critic_hidden_dim=128)

    @property
    def agent_type_topologies(self):
        return [(3, 3), (3, 2)]

    def reset(self):
        pass

    def next(self, actions: Dict[AgentKey, AgentAction]) -> Dict[AgentKey, AgentResponseFrame]:
        results: Dict[AgentKey, AgentResponseFrame] = {}
        for i in range(3):
            results[AgentKey(0, '0-' + str(i))] = AgentResponseFrame(0, False, [randval() for _ in range(3)])

        for i in range(2):
            results[AgentKey(1, '0-' + str(i))] = AgentResponseFrame(0, False, [randval() for _ in range(3)])

        if actions is not None:
            print("Actions", actions)
            for k, v in actions.items():
                if k.type == 0:
                    results[k].reward = 1 if v.get_action_index() == 1 else 0
                elif k.type == 1:
                    results[k].reward = 1 if v.get_action_index() == 0 else 0

        return results
