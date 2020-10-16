from utils.core import *
import random
from train import run
import unittest
import random
from typing import List, Callable, Dict, Tuple
from envs.base_env import *


class SimpleEnv(BaseEnv):
    """
    Just env 7 but won't self destruct on conversion.
    """
    @property
    def config(self):
        return Config(model_name="SimpleEnv8",
                      reward_scale=100,
                      pol_hidden_dim=8,
                      critic_hidden_dim=8,
                      episode_length=10,
                      games_per_update=4)

    @property
    def agent_type_topologies(self) -> List[Tuple[int, int]]:
        return [(3, 20), (3, 2)] # none, n, w, s, e

    def simulate(self, model: Callable[[Dict[AgentKey, AgentObservation]], Dict[AgentKey, AgentAction]], buffer: ReplayBuffer) -> float:
        """
        Here the objective is to have every agent of type 0 convert itself into agent 1 by selecting action 1,
        destroying itself in the process.
        """

        agents: Dict[AgentKey, int] = {}
        for i in range(20):
            agents[AgentKey(0, i + 1)] = 0
        agents[AgentKey(1, 0)] = 1

        sim_buffer: SimulationBuffer = SimulationBuffer()

        for i in range(10):
            observations: Dict[AgentKey, AgentObservation] = \
                {k: AgentObservation([0, 0, 0]) for k, v in agents.items()}

            actions: Dict[AgentKey, AgentAction] = model(observations)

            rewards: Dict[AgentKey, float] = {}

            dones: Dict[AgentKey, bool] = {}

            for k, v in actions.items():
                if k.type == 0 and v.get_action_index() == 1 and agents[k] == 0:
                    rewards[k] = 1
                    agents[k] = 1
                else:
                    rewards[k] = 0
                dones[k] = False

            sim_buffer.push(observations, actions, rewards, dones)

        sim_buffer.push_to_replay_buffer(buffer)

        vals = list(agents.values())
        return vals.count(1) - vals.count(0)


class SimpleEnvTest(unittest.TestCase):
    def test_env(self):
        run(SimpleEnv())
