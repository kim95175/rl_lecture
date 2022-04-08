import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

import core.pytorch_util as ptu

def create_lander_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )

class DQN(object):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        self.q_net = create_lander_q_network(self.ob_dim, self.ac_dim)
        self.q_net_target = create_lander_q_network(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

    def update(self, obs, action, next_obs, reward, done):
        """
            Update the parameters of the critic.
            arguments:
                obs: shape: (batch_size, ob_dim)
                next_obs: shape: (batch_size, ob_dim). The observation after taking one step forward
                reward: length: batch_size. Each element in reward is a scalar containing
                    the reward for each timestep
                done: length: batch_size. Each element in done is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action).to(torch.long)
        next_obs = ptu.from_numpy(next_obs)
        reward = ptu.from_numpy(reward)
        done = ptu.from_numpy(done)

        q_pred = self.q_net(obs)
        q_value = q_pred.gather(1, action.unsqueeze(1)).squeeze(1)
        

        # TODO compute the Q-values from the target network 
        q_pred_next_target = self.q_net_target(next_obs)

        if self.double_q:
            # You must fill this part for Q2 of the Q-learning portion of the homework.
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
            next_actions = self.q_net(next_obs).argmax(dim=1)
            q_value_next_target = q_pred_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_value_next_target, _ = q_pred_next_target.max(dim=1)

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        q_target = reward + self.gamma*q_value_next_target*(1-done)
        q_target = q_target.detach()

        assert q_value.shape == q_target.shape
        loss = self.loss(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
    
    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        q_values = self.qa_values(observation)
        action = q_values.argmax(-1)

        return action[0]
