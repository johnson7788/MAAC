import unittest
import torch
from algorithms.attention_sac import AttentionSAC
from envs.halite.halite_env import HaliteTrainHelper
from utils.core import *
import random
import traceback


def rval() -> float:
    return random.randrange(-1., 1.)


class HaliteAttentionSACTest(unittest.TestCase):
    def test_model(self):
        torch.autograd.set_detect_anomaly(True)
        self.algo = AttentionSAC([(5, 3), (5, 2)],
                                 tau=0.01,
                                 pi_lr=0.01,
                                 q_lr=0.01,
                                 gamma=0.95,
                                 pol_hidden_dim=128,
                                 critic_hidden_dim=128,
                                 attend_heads=4,
                                 reward_scale=10.)

        self.algo.prep_rollouts(device='cpu')

        sample: Dict[AgentKey, AgentObservation] = \
            {AgentKey(0, '0-1'): AgentObservation([1, 2, 3, 2, 3]),
             AgentKey(0, '0-2'): AgentObservation([2, 4, 3, 2, 4]),
             AgentKey(0, '0-3'): AgentObservation([2, 4, 3, 2, 4]),
             AgentKey(1, '0-1'): AgentObservation([1, 1, 3, 1, 4]),
             AgentKey(1, '0-2'): AgentObservation([1, 1, 3, 1, 4])}

        results = self.algo.step(sample, explore=True)

        self.assertEqual(len(results[AgentKey(0, '0-1')].action), 3)
        self.assertEqual(len(results[AgentKey(1, '0-1')].action), 2)

        for key in sample:
            self.assertTrue(key in results)

        for i in range(20):
            self.algo.step(sample)

        self.algo.prep_training(device='cpu')

        # Generate random training sample
        train_sample: List[Dict[AgentKey, AgentReplayFrame]] = \
            [{AgentKey(0, '0-1'): AgentReplayFrame([rval() for i in range(5)], [0, 1, 0], 5, False, [rval() for i in range(5)]),
              AgentKey(0, '0-2'): AgentReplayFrame([rval() for i in range(5)], [1, 0, 0], 5, False, [rval() for i in range(5)]),
              AgentKey(0, '0-3'): AgentReplayFrame([rval() for i in range(5)], [0, 1, 0], 5, False, [rval() for i in range(5)]),
              AgentKey(1, '0-1'): AgentReplayFrame([rval() for i in range(5)], [0, 1], 5, False, [rval() for i in range(5)]),
              AgentKey(1, '0-2'): AgentReplayFrame([rval() for i in range(5)], [0, 1], 5, False, [rval() for i in range(5)])}
             for _ in range(3)]
        train_sample: Dict[AgentKey, BatchedAgentReplayFrame] = preprocess_to_batch(train_sample)
        self.algo.update_critic(train_sample, logger=None)
        self.algo.update_policies(train_sample, logger=None)
        self.algo.update_all_targets()
