import numpy as np
import torch.nn as nn


class RandomPolicy(nn.Module):
    def __init__(self, params):
        super(RandomPolicy, self).__init__()
        self.num_envs = params.env_params.num_envs
        self.continuous_action = params.continuous_action
        self.action_dim = params.action_dim
        if self.continuous_action:
            action_low, action_high = params.action_spec
            self.action_mean = (action_low + action_high) / 2
            self.action_scale = (action_high - action_low) / 2

    def act_randomly(self):
        if self.continuous_action:
            return self.action_mean + self.action_scale * np.random.uniform(-1, 1, (self.num_envs, self.action_dim))
        else:
            return np.random.randint(self.action_dim, size=self.num_envs)

    def act(self, obs):
        return self.act_randomly()

    def save(self, path):
        pass
