"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import random
from collections import deque, namedtuple

import gym
import numpy as np
from torch import nn
import torch.optim as optim

from gym.envs.registration import register

import torch

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)


def register_custom_envs():
    from gym.envs.registration import registry
    if 'LunarLander-v3' not in registry.env_specs:
        register(
            id='LunarLander-v3',
            entry_point='envs.lunar_lander:LunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )
    
    if 'PointmassEasy-v0' not in registry.env_specs:
        register(
            id='PointmassEasy-v0',
            entry_point='envs.pointmass:Pointmass',
            kwargs={'difficulty': 0}
        )
    if 'PointmassMedium-v0' not in registry.env_specs:
        register(
            id='PointmassMedium-v0',
            entry_point='envs.pointmass:Pointmass',
            kwargs={'difficulty': 1}
        )
    if 'PointmassHard-v0' not in registry.env_specs:
        register(
            id='PointmassHard-v0',
            entry_point='envs.pointmass:Pointmass',
            kwargs={'difficulty': 2}
        )
    if 'PointmassVeryHard-v0' not in registry.env_specs:
        register(
            id='PointmassVeryHard-v0',
            entry_point='envs.pointmass:Pointmass',
            kwargs={'difficulty': 3}
        )
    if 'PointmassSpiral-v0' not in registry.env_specs:
        register(
            id='PointmassSpiral-v0',
            entry_point='envs.pointmass:Pointmass',
            kwargs={'difficulty': 4}
        )


def get_env_kwargs(env_name):
    if env_name == 'LunarLander-v3':
        def lunar_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': lander_optimizer(),
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 200000, #500000,
            'env_wrappers': lunar_empty_wrapper
        }
        kwargs['exploration_schedule'] = lander_exploration_schedule(kwargs['num_timesteps'])

    elif 'Pointmass' in env_name:
        def pointmass_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': pointmass_optimizer(),
            #'q_func': create_q_network,
            'replay_buffer_size': int(1e5),
            'gamma': 0.95,
            'learning_starts': 2000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 300,
            'grad_norm_clipping': 10,
            'lander': False,
            'num_timesteps': 50000,
            'env_wrappers': pointmass_empty_wrapper
        }
        kwargs['exploration_schedule'] = exploration_schedule(kwargs['num_timesteps'])
        
    else:
        raise NotImplementedError

    return kwargs


class Ipdb(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        import ipdb; ipdb.set_trace()
        return x

def lander_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3,  # keep init learning rate
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def pointmass_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=lambda epoch: 1e-3,  # keep init learning rate
    )

def exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len, lander=False, n_step=1, gamma = 1):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        #self.lander = lander
        self.float_ops = True
        self.size = size
        self.frame_history_len = frame_history_len

        self.idx      = 0
        self.num_in_buffer = 0
        
        self.obs      = None
        self.action   = None
        self.reward   = None
        self.next_obs = None
        self.done     = None
        
        self.gamma = gamma

        self.n_step = n_step
        if n_step > 1:
            self.n_states = deque(maxlen=n_step+1)
            self.n_actions = deque(maxlen=n_step+1)
            self.n_rewards = deque(maxlen=n_step+1)

    def init_replay_buffer(self, frame):
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.float_ops else np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.next_obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.float_ops else np.uint8)
            self.done     = np.empty([self.size],                     dtype=np.bool)
    
    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return (batch_size + 1)*self.n_step <= self.num_in_buffer

    def _encode_sample(self, idxes):
        #obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        obs_batch      = np.concatenate([self.obs[idx][None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch      = np.concatenate([self.next_obs[idx][None] for idx in idxes], 0)
        #next_obs_batch = np.concatenate([self._encode_next_observation(idx)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - (1 + self.n_step)), batch_size)
        #print("buffer.sample idxes = {}".format(idxes))
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        print("[rb]obs.shape = ", self.obs.shape)
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def _encode_next_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.next_obs.shape) == 2:
            return self.next_obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.next_obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.next_obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.next_obs.shape[1], self.next_obs.shape[2]
            return self.next_obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.lander else np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.next_obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.lander else np.uint8)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        
        #print("Store_frame idx = ", self.idx)
        self.obs[self.idx] = frame
        
        ret = self.idx
        self.idx = (self.idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        
        return ret

    def write(self, obs, action, reward, next_obs, done):
        if self.n_step == 1:
            self.store(obs, action, reward, next_obs, done)
        else:
            self.n_states.append(obs)
            self.n_actions.append(action)
            self.n_rewards.append(reward)

            if done:
                n_step_reward = 0
                while len(self.n_states) > 1:
                    n_step_obs = self.n_states.popleft()
                    n_step_action = self.n_actions.popleft()
                    n_step_next_obs = self.n_states[-1]
                    for i in range(len(self.n_rewards)):
                        n_step_reward += (self.gamma ** i ) * self.n_rewards.popleft()

                    self.store(n_step_obs, n_step_action, n_step_reward, n_step_next_obs, done)

            elif len(self.n_states) == self.n_step + 1:
                n_step_reward = 0
                for i in range(len(self.n_rewards)):
                     n_step_reward += (self.gamma ** i ) * self.n_rewards.popleft()
                self.store(self.n_states[0], self.n_actions[0], n_step_reward, self.n_states[-1], done)

    def store(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx]   = done
        self.next_obs[self.idx] = next_obs

        self.idx = (self.idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
            


