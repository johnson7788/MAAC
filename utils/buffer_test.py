import unittest
from utils.buffer import ReplayBuffer
import numpy as np
from utils.core import *


class TestBuffer(unittest.TestCase):
    def test_buffer(self):
        data = \
            {AgentKey(0, '0-1'): AgentReplayFrame([2, 1, 2, 2, 3], [0, 1, 0], 3, False, [3, 1, 1, 2, 3]),
             AgentKey(0, '0-2'): AgentReplayFrame([1, 1, 3, 2, 1], [0, 1, 0], 4, False, [2, 1, 1, 2, 2]),
             AgentKey(1, '0-1'): AgentReplayFrame([2, 0, 3, 1, 2], [0, 1], 5, False, [3, 0, 1, 3, 4])}

        max_steps = 4
        buffer = ReplayBuffer(max_steps)
        for i in range(5):
            buffer.push(data)
            self.assertEqual(buffer.length(), min(i + 1, max_steps))

        sample: List[Dict[AgentKey, AgentReplayFrame]] = buffer.sample(2, norm_rews=False)
        for s in sample:
            for k, v in s.items():
                self.assertEqual(v.reward, data[k].reward)

        sample: List[Dict[AgentKey, AgentReplayFrame]] = buffer.sample(2, norm_rews=True)
        for s in sample:
            for k, v in s.items():
                self.assertEqual(v.reward, 0)

        avg_rewards = buffer.get_average_rewards(3)
        for k, v in avg_rewards.items():
            self.assertEqual(v, data[k].reward)
