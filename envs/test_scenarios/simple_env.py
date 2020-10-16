from utils.core import *

class SimpleEnv:
    @property
    def config(self):
        return Config(reward_scale=100,
                      pol_hidden_dim=128,
                      critic_hidden_dim=128)

    @property
    def agent_type_topologies(self):
        return [(3, 2), (3, 2)]

    def reset(self):
        pass

    def next(self, actions: Dict[AgentKey, AgentAction]) -> Dict[AgentKey, AgentResponseFrame]:
        if actions is not None:
            print("Actions", [action.get_action_index() for action in actions.values()])
        results: Dict[AgentKey, AgentResponseFrame] = {}
        for i in range(2):
            key = AgentKey(i, '0-1')
            if actions is None:
                results[key] = AgentResponseFrame(0, False, [0, 0, 0])
            else:
                results[key] = AgentResponseFrame(1 if actions[key].get_action_index() == 1 else 0, False, [0, 0, 0])
        return results
