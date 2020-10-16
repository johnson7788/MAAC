from utils.core import *
import random
from train import run
import unittest
import random
from typing import List


class SimpleEnv:
    """
    Test a random number of agents reaching some location.
    """
    def __init__(self):
        self.spawn_frequency = 10
        self.board_size = 10
        self.item_locs: Dict[int, List[int]] = {}
        self.target_loc = [0, 0]
        self.reset()

    def board_val(self):
        return random.randint(-self.board_size, self.board_size)

    def rand_point(self) -> List[int]:
        return [self.board_val() for _ in range(2)]

    def displacement_to(self, from_loc: List[int]):
        return [self.target_loc[i] - from_loc[i] for i in range(2)]

    def dist_to(self, from_loc: List[int]):
        displacement = self.displacement_to(from_loc)
        dist = 0
        for dis in displacement:
            dist += abs(dis)
        return dist

    def at_target(self, loc):
        for i in range(2):
            if loc[i] != self.target_loc[i]:
                return False
        return True

    @property
    def config(self):
        return Config(reward_scale=10,
                      pol_hidden_dim=8,
                      critic_hidden_dim=8)

    @property
    def agent_type_topologies(self):
        return [(2, 5)] # none, n, w, s, e

    def reset(self):
        if len(self.item_locs) > 0:
            distances = [self.dist_to(item_loc) for item_loc in self.item_locs.values()]
            print("Avg dist", sum(distances) / len(distances))
        self.item_locs: Dict[AgentKey, List[int]] = {}
        self.target_loc = self.rand_point()
        self.agent_num = 0
        self.turn_num = 0

    def next(self, actions: Dict[AgentKey, AgentAction]) -> Dict[AgentKey, AgentResponseFrame]:
        if actions is not None:
            for k, v in actions.items():
                loc = self.item_locs[k]
                action_index = v.get_action_index()
                if action_index == 0:
                    loc[0] += 1
                elif action_index == 1:
                    loc[0] -= 1
                elif action_index == 2:
                    loc[1] += 1
                elif action_index == 3:
                    loc[1] -= 1

        if self.turn_num % self.spawn_frequency == 0:
            self.item_locs[AgentKey(0, self.agent_num)] = self.rand_point()
            self.agent_num += 1

        results: Dict[AgentKey, AgentResponseFrame] = {}
        for k, v in self.item_locs.items():
            results[k] = AgentResponseFrame(-self.dist_to(v), False, self.displacement_to(v))

        self.turn_num += 1

        return results


class SimpleEnvTest(unittest.TestCase):
    def test_env(self):
        run(SimpleEnv())
