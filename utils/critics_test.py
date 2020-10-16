import unittest
import numpy as np
from torch import Tensor
from typing import List
from utils.critics import AttentionCritic
from utils.core import *
from utils.buffer import AgentReplayFrame


class TestHaliteCritic(unittest.TestCase):
    def testAttentionCritic(self):
        critic = AttentionCritic([(5, 3), (5, 2)], attend_heads=4)
        sample_frames = \
            [{AgentKey(0, '0-1'): AgentReplayFrame([2, 1, 2, 2, 3], [0, 1, 0], 3, False, [3, 1, 1, 2, 3]),
              AgentKey(0, '0-2'): AgentReplayFrame([1, 1, 3, 2, 1], [0, 1, 0], 5, False, [2, 1, 1, 2, 2]),
              AgentKey(0, '0-3'): AgentReplayFrame([2, 0, 3, 0, 2], [1, 0, 0], 1, False, [3, 0, 1, 3, 4]),
              AgentKey(1, '0-1'): AgentReplayFrame([2, 0, 3, 1, 2], [0, 1], 3, False, [3, 0, 1, 3, 4])},
             {AgentKey(0, '0-1'): AgentReplayFrame([2, 1, 2, 2, 3], [0, 1, 0], 3, False, [3, 1, 1, 2, 3]),
              AgentKey(0, '0-2'): AgentReplayFrame([1, 1, 3, 2, 1], [0, 1, 0], 5, False, [2, 1, 1, 2, 2]),
              AgentKey(0, '0-3'): AgentReplayFrame([2, 0, 3, 0, 2], [1, 0, 0], 0, True, [3, 0, 1, 3, 4]),
              AgentKey(1, '0-1'): AgentReplayFrame([2, 0, 3, 1, 2], [0, 1], 3, False, [3, 0, 1, 3, 4])},
             {AgentKey(0, '0-1'): AgentReplayFrame([2, 1, 2, 2, 3], [0, 1, 0], 3, False, [3, 1, 1, 2, 3]),
              AgentKey(0, '0-2'): AgentReplayFrame([1, 1, 3, 2, 1], [0, 1, 0], 5, False, [2, 1, 1, 2, 2]),
              AgentKey(1, '0-1'): AgentReplayFrame([2, 0, 3, 1, 2], [0, 1], 3, False, [3, 0, 1, 3, 4])}]

        sample_frames: Dict[AgentKey, BatchedAgentReplayFrame] = preprocess_to_batch(sample_frames)

        results: Dict[AgentKey, List[float]] = critic.forward(sample_frames)

        print(results)

        for k in sample_frames.keys():
            self.assertTrue(k in results)

        # self.assertEqual(len(results), len(tups))
        # for i in range(len(tups)):
        #     self.assertEqual(len(results[i]), len(tups[i]))
