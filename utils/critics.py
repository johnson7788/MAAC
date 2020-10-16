import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List, Tuple, Dict
from utils.core import *
from utils.buffer import AgentReplayFrame



"""
I think I need to separate these into an ID-based matrix layout, with agent type as supplementary
data.  The issue now is that when I sample something from the replay buffer, I expect the following:

agent_1 obs frame 0 | agent_1 action frame 0
agent_1 obs frame 3 | agent_1 action frame 3

agent_2 obs frame 0 | agent_2 action frame 0
agent_2 obs frame 3 | agent_2 action frame 3

whereas I'm actually returning something randomized

agent_1 obs frame 0 | agent_1 action frame 0
agent_1 obs frame 3 | agent_1 action frame 0

agent_2 obs frame 1 | agent_2 obs frame 1
agent_2 obs frame 3 | agent_2 obs frame 3

Essentially the frames don't match up since I'm not storing them by agent.  

I'll essentially need to do a padded matrix for when new agents are added and destroyed, and add to 
it dynamically.  

I think I still just need agent_types total trained networks though
"""


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, agent_types: List[Tuple[int, int]], hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            agent_types (ex. [(3, 2), (5, 3)]): Size of state
            and action spaces per agent type
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.attend_heads = attend_heads
        self.n_agent_types = len(agent_types)

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        # Construct a network for each agent type
        for sdim, adim in agent_types:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        # No reference to len(agents) anywhere here, that's a really good sign
        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self, nagents):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / nagents)

    def forward(self, inps: Dict[AgentKey, BatchedAgentObservationAction],
                return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0) -> Dict[AgentKey, List[float]]:
        """
        Inputs:
            inps (List[Dict[AgentKey, AgentReplayFrame]): batch of observations
                                for agent types/indices over some number of rounds
                                (these rounds may have occurred in different threads)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """

        inp_keys = [k for k in inps.keys()]

        # Actions
        actions = [v.acs for v in inps.values()]

        # extract state-action encoding for each agent type
        sa_encodings = [self.critic_encoders[k.type](torch.cat((v.obs, v.acs), dim=1)) for k, v in inps.items()]

        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[k.type](v.obs) for k, v in inps.items()]

        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for enc in s_encodings] for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(inps))]
        all_attend_logits = [[] for _ in range(len(inps))]
        all_attend_probs = [[] for _ in range(len(inps))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, selector in zip(range(len(inps)), curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != i]
                values = [v for j, v in enumerate(curr_head_values) if j != i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate Q per agent
        all_rets: Dict[AgentKey, List[float]] = {}
        for i in range(len(inps)):
            # head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
            #                    .mean()) for probs in all_attend_probs[agent_type_index]]
            agent_rets = []
            inp_key = inp_keys[i]
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[inp_key.type](critic_in)
            int_acs = actions[i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            # if logger is not None:
            #     logger.add_scalars('agent%i/attention' % agent_type_index,
            #                        dict(('head%i_entropy' % h_i, ent) for h_i, ent
            #                             in enumerate(head_entropies)),
            #                        niter)
            all_rets[inp_key] = agent_rets
        return all_rets
