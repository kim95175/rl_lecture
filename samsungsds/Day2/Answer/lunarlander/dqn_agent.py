import numpy as np
from core.dqn_utils import ReplayBuffer, PiecewiseSchedule
from dqn import DQN

class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.dqn = DQN(agent_params, self.optimizer_spec)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = ReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander, n_step=agent_params['n_step'], gamma=agent_params['gamma'])
        self.replay_buffer.init_replay_buffer(self.last_obs)
        self.t = 0
        self.num_param_updates = 0
        self.num_episodes = 0

        self.num_grounded = 0
        self.num_at_site = 0

        self.n_step = agent_params['n_step']
        self.gamma = agent_params['gamma']


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        
        obs = self.last_obs
        #print(self.replay_buffer_idx)
        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        if np.random.random() < eps or self.t < self.learning_starts :
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            
            #action = np.random.randint(self.num_actions)
            action = self.env.action_space.sample()
        else:
            # TODO query the policy with enc_last_obs to select action
            action = self.dqn.get_action(obs)
        
        # TODO take a step in the environment using the action from the policy
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        next_obs, reward, done, info = self.env.step(action)
        self.last_obs=next_obs

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `write` function
        self.replay_buffer.write(obs, action, reward, next_obs, done)
        
        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

            self.num_episodes += 1
            if info['at_site']:
                self.num_at_site += 1
            if info['grounded']:
                self.num_grounded += 1

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, obs, action, reward, next_obs, done):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log = self.dqn.update(
                obs, action, next_obs, reward, done
            )
            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.dqn.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
    
    