from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from tqdm import tqdm

import core.pytorch_util as ptu
import core.utils
from core.logger import Logger

from dqn_agent import DQNAgent
from core.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs
)


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        print(self.params)
        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params['env_name'])
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            print("if 'env_wrappers' in self.params:")
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"),video_callable=video_schedule, force=True)
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        
        self.env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
                
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0] # #if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n
        print("ob_dim = {}, ac_dim = {}".format(self.env.observation_space, ac_dim))
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy):
        
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = self.params['scalar_log_freq'] if isinstance(self.agent, DQNAgent) else 1

        for itr in tqdm(range(n_iter)):
            if itr % print_period == 0:
                print("\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                envsteps_this_batch = 1
            
            self.total_envsteps += envsteps_this_batch

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # log/save
            if itr % self.params['scalar_log_freq'] == 0:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_dqn_logging(all_logs)

                #if self.params['save_params']:
                #    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))
        
        print("\n\n********** Training finished ************")
        all_logs = self.train_agent()
        self.perform_dqn_logging(all_logs)

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])
            #if len(ob_batch) != 0:
                #print("rl_trainer/train_agent ", ob_batch[0].shape, next_ob_batch[0].shape)
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs


    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        logs["Train_EpisodeSoFar"] = self.agent.num_episodes
        print("Timestep %d" % (self.agent.t,))
        print("Num Episodes %d" % (self.agent.num_episodes,))
        if self.agent.num_episodes > 0:
            print("Grounded rate(%) = {0:.2f}".format(self.agent.num_grounded * 100 /self.agent.num_episodes))
            print("Success rate(%) = {0:.2f}".format(self.agent.num_at_site * 100 /self.agent.num_episodes))

        logs["Num_Episode_Grounded"] = self.agent.num_grounded
        logs["Num_Episode_Grounded_at_site"] = self.agent.num_at_site

        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('\t{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()


def video_schedule(episode_id):
    '''
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0
    '''
    return episode_id % 100 == 0