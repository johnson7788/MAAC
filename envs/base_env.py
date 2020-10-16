from utils.core import *
import random
import unittest
import random
from typing import List, Tuple, Callable, Dict
from abc import ABC
from utils.buffer import ReplayBuffer


class SimulationBuffer:
    def __init__(self):
        self.frames: List[Dict[AgentKey, AgentReplayFrameBuilder]] = []

    def push(self,
             observations: Dict[AgentKey, AgentObservation],
             actions: Dict[AgentKey, AgentAction],
             rewards: Dict[AgentKey, float],
             dones: Dict[AgentKey, bool]):
        replay_frames: Dict[AgentKey, AgentReplayFrameBuilder] = {}
        for k in observations.keys():
            replay_frames[k] = AgentReplayFrameBuilder(
                observations[k].obs,
                actions[k].action,
                rewards[k],
                dones[k],
                None)
        self.frames.append(replay_frames)

    def push_to_replay_buffer(self, buffer: ReplayBuffer):
        for i in range(len(self.frames)):
            frame = self.frames[i]
            next_frame = self.frames[i + 1] if i < len(self.frames) - 1 else None
            for k in frame.keys():
                if next_frame is None:
                    frame[k].next_obs = [0] * len(frame[k].obs)
                elif k not in next_frame:
                    assert frame[k].done
                    frame[k].next_obs = [0] * len(frame[k].obs)
                else:
                    frame[k].next_obs = next_frame[k].obs
            buffer.push({k: v.build() for k, v in frame.items()})



class BaseEnv(ABC):
    """
    Override this base class in order to construct new scenarios for which you'd like to apply the MAAC algorithm.
    """

    @property
    def config(self) -> Config:
        pass

    @property
    def agent_type_topologies(self) -> List[Tuple[int, int]]:
        pass

    def simulate(self, model: Callable[[Dict[AgentKey, AgentObservation]], Dict[AgentKey, AgentAction]], buffer: ReplayBuffer) -> float:
        """
        Simulate the completion of the game environment, returning a ranking at the end indicating how effective
        the specified agent was at selecting actions.
        """
        pass
