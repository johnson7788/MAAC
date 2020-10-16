from typing import List
from torch import Tensor
from torch.autograd import Variable
from typing import List, Dict


class Config:
    """
    num_rollout_threads (int):
        The number of threads that will be used to run copies of the game environment, each of which
        will push data to the replay buffer for training purposes.
    gamma (float):
        Discount factor in (0, 1) range, 0 being only looking at immediate returns, .99 being looking
        more at long-term horizon (>1 might result in infinite horizon error though)
    tau (float):
        Target update rate
    pi_lr (float):
        Learning rate for policy.
    reward_scale (float):
        Scaling for reward (has effect of optimal policy entropy).
    hidden_dim (int):
        Number of hidden dimensions for networks.
    buffer_length (int):
        The maximum length of the replay buffer (at which point it behaves like a stack, removing
        the oldest element for the newest element).
    episode_length (int):
        How many turns into each individual game the environment should simulate.
    """
    def __init__(self,
                 model_name: str = "halite",
                 n_rollout_threads: int = 1,
                 buffer_length: int = 1900,
                 n_episodes: int = 50000,
                 episode_length: int = 200,
                 games_per_update: int = 200,
                 num_updates: int = 4,
                 batch_size: int = 300,
                 save_interval: int = 900,
                 pol_hidden_dim: int = 128,
                 critic_hidden_dim: int = 128,
                 attend_heads: int = 4,
                 pi_lr: float = .001,
                 q_lr: float = .001,
                 tau: float = .001,
                 gamma: float = 0.99,
                 reward_scale: int = 10,
                 use_gpu: bool = True):
        self.model_name = model_name
        self.n_rollout_threads = n_rollout_threads
        self.buffer_length = buffer_length
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.games_per_update = games_per_update
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.pol_hidden_dim = pol_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.attend_heads = attend_heads
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.tau = tau
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.use_gpu = use_gpu


class AgentKey:
    def __init__(self, type: int, id: int):
        self.type = type
        self.id = id

    def __eq__(self, other):
        return (self.type, self.id) == (other.type, other.id)

    def __hash__(self):
        return hash((self.type, self.id))

    def __str__(self):
        return "AgentKey(" + str(self.type) + ", " + str(self.id) + ")"

    def __repr__(self):
        return str(self)


def not_none_or_empty(vals: List[float]):
    return vals is not None and len(vals) > 0


class AgentObservation:
    def __init__(self, obs: List[float]):
        assert not_none_or_empty(obs)
        self.obs = obs

    def __str__(self):
        return "AgentObservation([" + str(self.obs) + "])"

    def __repr__(self):
        return str(self)


class AgentAction:
    def __init__(self, action: List[float]):
        assert not_none_or_empty(action)
        self.action = action

    def get_action_index(self):
        return self.action.index(max(self.action))

    def __str__(self):
        return "AgentAction([" + str(self.action) + "])"

    def __repr__(self):
        return str(self)


class AgentReplayFrame:
    def __init__(self,
                 obs: List[float],
                 action: List[float],
                 reward: float,
                 done: bool,
                 next_obs: List[float]):
        assert not_none_or_empty(obs)
        assert not_none_or_empty(action)
        assert not_none_or_empty(next_obs)
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs


class AgentReplayFrameBuilder:
    def __init__(self,
                 obs: List[float],
                 action: List[float],
                 reward: float,
                 done: bool,
                 next_obs: List[float]):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs

    def build(self) -> AgentReplayFrame:
        return AgentReplayFrame(self.obs,
                                self.action,
                                self.reward,
                                self.done,
                                self.next_obs)

    def valid(self):
        return not_none_or_empty(self.obs) and \
               not_none_or_empty(self.action) and \
               not_none_or_empty(self.next_obs)


def valid_variable(torch_var: Variable):
    for val in torch_var.size():
        if val == 0:
            return False
    return True


class BatchedAgentReplayFrame:
    def __init__(self, obs: Variable, acs: Variable, rews: Variable, dones: Variable, next_obs: Variable):
        assert valid_variable(obs)
        assert valid_variable(acs)
        assert valid_variable(rews)
        assert valid_variable(dones)
        assert valid_variable(next_obs)
        self.obs = obs
        self.acs = acs
        self.rews = rews
        self.dones = dones
        self.next_obs = next_obs


class BatchedAgentObservationAction:
    def __init__(self, obs: Variable, acs: Variable):
        assert valid_variable(obs)
        assert valid_variable(acs)
        self.obs = obs
        self.acs = acs


def append_if_not_empty(obs: List[List[float]], val: List[float], empty_val: List[float]):
    if val is not None:
        obs.append(val)
    else:
        obs.append(empty_val)


def preprocess_to_batch(
        inps: List[Dict[AgentKey, AgentReplayFrame]],
        to_gpu=False) -> Dict[AgentKey, BatchedAgentReplayFrame]:
    """
    For efficient training we restructure the training data into matrices
    """
    key_superset = set(key for inp in inps for key in inp.keys())
    results: Dict[AgentKey, List[BatchedAgentReplayFrame]] = {}
    for key in key_superset:
        obs = []
        obs_size = 0
        acs = []
        acs_size = 0
        rews = []
        dones = []
        next_obs = []
        next_obs_size = 0
        for inp in inps:
            if key in inp:
                frame = inp[key]
                obs_size = len(frame.obs)
                acs_size = len(frame.action)
                next_obs_size = len(frame.next_obs)
                break
        for inp in inps:
            if key not in inp:
                obs.append([0] * obs_size)
                acs.append([0] * acs_size)
                rews.append([0])
                dones.append([1])
                next_obs.append([0] * next_obs_size)
            else:
                frame = inp[key]
                append_if_not_empty(obs, frame.obs, [0] * obs_size)
                append_if_not_empty(acs, frame.action, [0] * acs_size)
                rews.append([frame.reward])
                dones.append([1 if frame.done else 0])
                append_if_not_empty(next_obs, frame.next_obs, [0] * next_obs_size)

        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        results[key] = BatchedAgentReplayFrame(
            cast(obs),
            cast(acs),
            cast(rews),
            cast(dones),
            cast(next_obs))

    return results
