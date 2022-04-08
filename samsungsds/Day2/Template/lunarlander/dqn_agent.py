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
        obs = TODO
        
        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
        perform_randon_action = TODO
        if perform_random_action:
            action = TODO
        else:
            # select action
            # HINT : see your dwn's 'get_action' function
            action = TODO
        
        # TODO take a step in the environment using the action from the policy
        # HINT: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)       
        TODO
        
        self.last_obs=next_obs

        # TODO store the result of taking this action into the replay buffer
        # HINT: see your replay buffer's `write` function
        TODO
        

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

             # TODO fill in the call to the update function using the appropriate tensors
            log = self.dqn.update(
                TODO
            )

            # TODO update the target network periodically 
            # HINT: dqn.py already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                TODO

            self.num_param_updates += 1

        self.t += 1
        return log
    
    