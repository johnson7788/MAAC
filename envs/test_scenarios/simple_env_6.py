from utils.core import *
import random
from train import run
import unittest
import random
from typing import List
from envs.base_env import BaseEnv


class SimpleEnv(BaseEnv):
    """
    Test a random number of agents reaching some location.
    """
    def __init__(self):
        self.spawn_frequency = 10
        self.board_size = 10
        self.reset()

    def board_val(self):
        return random.randint(-self.board_size, self.board_size)

    def rand_point(self) -> List[int]:
        return [self.board_val() for _ in range(2)]

    def displacement(self):
        return [self.target_loc[i] - self.item_loc[i] for i in range(2)]

    def dist(self):
        displacement = self.displacement()
        val = 0
        for dis in displacement:
            val += abs(dis)
        return val

    @property
    def config(self):
        return Config(reward_scale=2,
                      pol_hidden_dim=8,
                      critic_hidden_dim=8)

    @property
    def agent_type_topologies(self):
        return [(2, 5), (2, 5)] # none, n, w, s, e

    def reset(self):
        self.item_loc = self.rand_point()
        self.target_loc = self.rand_point()
        self.turn_num = 0

    def next(self, actions: Dict[AgentKey, AgentAction]) -> Dict[AgentKey, AgentResponseFrame]:
        key = AgentKey(0, '0-1')

        if actions is not None:
            action_index = actions[key].get_action_index()
            if action_index == 0:
                self.item_loc[0] += 1
            elif action_index == 1:
                self.item_loc[0] -= 1
            elif action_index == 2:
                self.item_loc[1] += 1
            elif action_index == 3:
                self.item_loc[1] -= 1

        results: Dict[AgentKey, AgentResponseFrame] = {}
        results[key] = AgentResponseFrame(-self.dist(), False, self.displacement())
        results[AgentKey(1, '0-1')] = AgentResponseFrame(0, False, [0, 0])

        self.turn_num += 1

        return results

    def global_reward(self) -> float:
        return self.dist()


class SimpleEnvTest(unittest.TestCase):
    def test_env(self):
        run(SimpleEnv())
