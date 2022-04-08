import torch.optim as optim
from torch import nn
import torch
import numpy as np

import core.pytorch_util as ptu

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, size, init_method):
        super(MLP, self).__init__()
        # Layer 2 - 400 - 400 - 5
        layers = []
        in_size = input_size
        for _ in range(n_layers):
            curr_layer = nn.Linear(in_size, size)
            if init_method is not None:
                curr_layer.apply(init_method)
            layers.append(curr_layer)
            layers.append(nn.Tanh())
            in_size = size
        
        last_layer = nn.Linear(in_size, output_size)
        if init_method is not None:
            last_layer.apply(init_method)
        layers.append(last_layer)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x


class RNDModel(nn.Module):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size'] # default : 5
        self.n_layers = hparams['rnd_n_layers'] # default : 2
        self.size = hparams['rnd_size'] # default : 400
        self.optimizer_spec = optimizer_spec

        # TODO: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT) There are two weight init methods defined above
        self.f = MLP(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        self.f_hat = MLP(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)

        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        #self.optimizer = torch.optim.Adam(self.f_hat.parameters(), lr=1)
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, next_obs):
        # TODO: Get the prediction error for next_obs
        # HINT: Remember to detach the output of self.f!
        f_out = self.f(next_obs).detach()
        f_hat_out = self.f_hat(next_obs)
        error = torch.norm(f_out - f_hat_out, dim=1)  # mean error over ob_dim for each item in the batch
        return error

    def forward_np(self, next_obs):
        next_obs = ptu.from_numpy(next_obs)
        error = self(next_obs)
        return ptu.to_numpy(error)

    def update(self, next_obs):
        # TODO: Update f_hat using next_obs
        # Hint: Take the mean prediction error across the batch
        next_obs = ptu.from_numpy(next_obs)
        error = self.forward(next_obs)
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



