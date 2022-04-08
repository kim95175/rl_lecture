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


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
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
        if ('Pointmass' in self.params['env_name']):
            import matplotlib
            matplotlib.use('Agg')
            self.env.set_logdir(self.params['logdir'] + '/expl_')
            #self.eval_env.set_logdir(self.params['logdir'] + '/eval_')

        if 'env_wrappers' in self.params:
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir']), force=True)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        
        self.env.seed(seed)
    
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

        print_period = self.params['scalar_log_freq']
        for itr in tqdm(range(n_iter)):
            if itr % print_period == 0:
                print("\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            #if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
            self.agent.step_env()
            envsteps_this_batch = 1
            
            self.total_envsteps += envsteps_this_batch

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            if itr % print_period == 0:
                self.dump_density_graphs(itr)


            # log/save
            if itr % self.params['scalar_log_freq'] == 0:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_dqn_logging(all_logs)

        
        print("\n\n********** Training finished ************")
        all_logs = self.train_agent()
        self.perform_dqn_logging(all_logs)
        self.dump_density_graphs(itr)

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])
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
            print("Success rate(%) = {0:.2f}".format(self.agent.num_at_site * 100 /self.agent.num_episodes))

        logs["Num_Episode_reach_the_goal"] = self.agent.num_at_site

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
        print('Done logging...\n\n')


    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return
        
        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')
        
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)
        if self.agent.exploration_model is not None:
            plt.clf()            
            density = self.agent.exploration_model.forward_np(obs)
            density = density.reshape(ii.shape)
            plt.imshow(density[::-1])
            plt.colorbar()
            plt.title('RND Value')
            self.fig.savefig(filepath('rnd_value'))#, bbox_inches='tight')
   
        plt.clf()
        exploration_values = self.agent.dqn.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title('predicted Q value')
        self.fig.savefig(filepath('predicted_q_value')) #, bbox_inches='tight')
