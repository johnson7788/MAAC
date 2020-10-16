import torch
import os
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from algorithms.attention_sac import AttentionSAC
from utils.core import *
import time
from envs.base_env import BaseEnv
from utils.savehelper import run_setup


def run(halite_env: BaseEnv, load_latest: bool=False):
    config = halite_env.config

    model_path, run_num, run_dir, log_dir = run_setup(config.model_name, get_latest_model=load_latest)

    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    # Build MAAC model
    if model_path is None:
        model = AttentionSAC(halite_env.agent_type_topologies,
                             tau=config.tau,
                             pi_lr=config.pi_lr,
                             q_lr=config.q_lr,
                             gamma=config.gamma,
                             pol_hidden_dim=config.pol_hidden_dim,
                             critic_hidden_dim=config.critic_hidden_dim,
                             attend_heads=config.attend_heads,
                             reward_scale=config.reward_scale)
    else:
        model = AttentionSAC.init_from_save(model_path, load_critic=True)

    # Build replay buffer
    replay_buffer = ReplayBuffer(config.buffer_length)

    prev_time = time.perf_counter()

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        curr_time = time.perf_counter()
        print("Episodes %i-%i of %i (%is)" % (ep_i + 1,
                                              ep_i + 1 + config.n_rollout_threads,
                                              config.n_episodes,
                                              (curr_time - prev_time)))
        model.prep_rollouts(device='cpu')

        game_reward = halite_env.simulate(lambda o: model.step(o, explore=True), replay_buffer)

        t += config.n_rollout_threads
        if (replay_buffer.length() >= config.batch_size and
            (t % config.games_per_update) < config.n_rollout_threads):
            print("Training")
            if config.use_gpu:
                model.prep_training(device='gpu')
            else:
                model.prep_training(device='cpu')
            for u_i in range(config.num_updates):
                sample: List[Dict[AgentKey, AgentReplayFrame]] = replay_buffer.sample(config.batch_size)
                # print("Original sample size", len(sample))
                # print("Preprocessing to batch structure")
                sample: Dict[AgentKey, BatchedAgentReplayFrame] = preprocess_to_batch(sample, to_gpu=config.use_gpu)
                # print("Filtered sample size", len(sample))
                # if len(sample) < 5:
                #     print("Sample size keys:", sample.keys())
                # print("Updating model critic")
                model.update_critic(sample, logger=logger)
                # print("Updating model policies")
                model.update_policies(sample, logger=logger)
                model.update_all_targets()
            model.prep_rollouts(device='cpu')

        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for k, v in ep_rews.items():
            logger.add_scalar('agent%s/mean_episode_rewards' % str(k), v, ep_i)

        logger.add_scalar("global_env_rewards", game_reward, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            print("Saving")
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')
            print("run_dir", run_dir)

        prev_time = curr_time

    model.save(run_dir / 'model.pt')
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
