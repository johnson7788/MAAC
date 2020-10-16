import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from utils.iterablehelper import zip_equal
from typing import List, Tuple
from utils.core import *
import numpy as np

MSELoss = torch.nn.MSELoss()


class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task

    Had to change a couple things in this class to make agent networks work by
    type as opposed to a constant number of agents
    """

    def __init__(self, algo_config: List[Tuple[int, int]],
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            algo_config (List[Tuple[int, int]]): Agent types which will exist in this environment
                Ex. [(20, 8), (20, 2)]
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """

        print(algo_config)
        # Dictionary which maps agent type to its topology
        self.agents = [AttentionAgent(sdim, adim, lr=pi_lr, hidden_dim=pol_hidden_dim) for sdim, adim in algo_config]
        self.critic = AttentionCritic(algo_config, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        self.target_critic = AttentionCritic(algo_config, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

        self.init_dict = {'gamma': gamma, 'tau': tau,
                          'pi_lr': pi_lr, 'q_lr': q_lr,
                          'reward_scale': reward_scale,
                          'pol_hidden_dim': pol_hidden_dim,
                          'critic_hidden_dim': critic_hidden_dim,
                          'attend_heads': attend_heads,
                          'algo_config': algo_config}

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations: Dict[AgentKey, AgentObservation], explore=False) -> Dict[AgentKey, AgentAction]:
        return {k: AgentAction(self.agents[k.type].step(Variable(torch.Tensor(np.array([v.obs])), requires_grad=False), explore=explore).tolist()[0])
                for k, v in observations.items()}

    def update_critic(self,
                      finalized_frames: Dict[AgentKey, BatchedAgentReplayFrame],
                      soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """

        # Q loss
        next_acs: Dict[AgentKey, Tensor] = {}
        next_log_pis: Dict[AgentKey, float] = {}
        for k, v in finalized_frames.items():
            pi = self.target_policies[k.type]
            ob = v.next_obs
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs[k] = curr_next_ac
            next_log_pis[k] = curr_next_log_pi
        trgt_critic_in = {k: BatchedAgentObservationAction(v.next_obs, next_acs[k]) for k, v in finalized_frames.items()}
        critic_in = {k: BatchedAgentObservationAction(v.obs, v.acs) for k, v in finalized_frames.items()}
        next_qs = self.target_critic(trgt_critic_in) # calls "forward", also TODO this doesn't need to be computed for frames in which done = True
        curr_qs = self.critic(critic_in, regularize=True, logger=logger, niter=self.niter)
        q_loss = 0
        for k, v in finalized_frames.items():
            (nq,) = next_qs[k]
            log_pi = next_log_pis[k]
            (pq, regs) = curr_qs[k]

            target_q = (v.rews +
                        self.gamma * nq *
                        (1 - v.dones))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        num_agents = len(finalized_frames.items())
        self.critic.scale_shared_grads(num_agents)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * num_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self,
                        finalized_frames: Dict[AgentKey, BatchedAgentReplayFrame],
                        soft=True, logger=None, **kwargs):
        samp_acs = {}
        all_probs = {}
        all_log_pis = {}
        all_pol_regs = {}

        for k, v in finalized_frames.items():
            pi = self.policies[k.type]
            ob = v.obs
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            if logger is not None:
                logger.add_scalar('agent%s/policy_entropy' % k.id, ent,
                                  self.niter)
            samp_acs[k] = curr_ac
            all_probs[k] = probs
            all_log_pis[k] = log_pi
            all_pol_regs[k] = pol_regs

        critic_in = {k: BatchedAgentObservationAction(v.obs, samp_acs[k]) for k, v in finalized_frames.items()}
        critic_rets = self.critic(critic_in, return_all_q=True)

        for k, val in finalized_frames.items():
            probs = all_probs[k]
            log_pi = all_log_pis[k]
            pol_regs = all_pol_regs[k]
            (q, all_q) = critic_rets[k]

            curr_agent = self.agents[k.type]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            pol_loss.backward()
            enable_gradients(self.critic)

        for curr_agent in self.agents:
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            # if logger is not None:
            #     logger.add_scalar('agent%s/losses/pol_loss' % k.id,
            #                       pol_loss, self.niter)
            #     logger.add_scalar('agent%s/grad_norms/pi' % k.id,
            #                       grad_norm, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip_equal(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
