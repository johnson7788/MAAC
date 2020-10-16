import numpy as np
from typing import List, Tuple, Dict
from utils.iterablehelper import zip_equal
import random
from utils.core import *


class RewardStats:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel threading
    """
    def __init__(self, max_steps: int):
        """
        Inputs:
            agent_type_topologies (List[Tuple[int, int]): ex. [(5, 2), (5, 3)]

        Data is stored by data[agent_type] = List[FinalizedFrame]
        where tuple is observation, action, reward, done, next observation
        """
        self.max_steps = max_steps

        # actually wait why pad it when we can just use a dict?  then we can sample directly and immediately preprocess
        self.data: List[Dict[AgentKey, AgentReplayFrame]] = []

    def push(self, new_data: Dict[AgentKey, AgentReplayFrame]):
        self.data.append(new_data)
        if len(self.data) > self.max_steps:
            self.data = self.data[1:]

    def sample(self, N: int, norm_rews=True) -> List[Dict[AgentKey, AgentReplayFrame]]:
        # print("Sampling from buffer length: ", len(self.data))
        # print("len of buffer is", len(self.data))
        sample = random.sample(self.data, min(N, len(self.data)))
        if norm_rews:
            sample_norm = []
            key_superset = set(key for sample_row in sample for key in sample_row.keys())
            reward_stats_dict = {k: self.get_reward_stats(k) for k in key_superset}
            for data_row in sample:
                data_row_new = {}
                for k, v in data_row.items():
                    stats = reward_stats_dict[k]
                    data_row_new[k] = AgentReplayFrame(v.obs, v.action, (v.reward - stats.mean) / (stats.std + 1.), v.done, v.next_obs)
                sample_norm.append(data_row_new)
            return sample_norm
        else:
            return sample

    def get_reward_stats(self, key: AgentKey) -> RewardStats:
        reward_vals = np.array([row[key].reward for row in self.data if key in row])
        return RewardStats(reward_vals.mean(), reward_vals.std())

    def length(self):
        return len(self.data)

    def get_average_rewards(self, N) -> Dict[AgentKey, float]:
        start = max(0, len(self.data) - N)
        key_superset = set(key for row in self.data[start:] for key in row.keys())
        stats: Dict[AgentKey, float] = \
            {key: np.array([row[key].reward for row in self.data[start:] if key in row]).mean()
             for key in key_superset}
        return stats
