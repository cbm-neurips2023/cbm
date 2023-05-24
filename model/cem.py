import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class ActionDistribution:
    def __init__(self, params):
        self.action_dim = action_dim = params.action_dim
        self.continuous_action = params.continuous_action
        self.num_envs = num_envs = params.env_params.num_envs

        cem_params = params.policy_params.cem_params
        self.num_top_candidates = num_top_candidates = cem_params.num_top_candidates

        num_horizon_steps = cem_params.num_horizon_steps
        std_scale = cem_params.std_scale
        device = params.device

        if self.continuous_action:
            mu = torch.zeros(num_envs, num_horizon_steps, action_dim, dtype=torch.float32, device=device)
            std = torch.ones(num_envs, num_horizon_steps, action_dim, dtype=torch.float32, device=device) * std_scale
            self.init_dist = Normal(mu, std)

            action_low, action_high = params.action_spec
            self.action_low_device = torch.tensor(action_low, dtype=torch.float32, device=device)
            self.action_high_device = torch.tensor(action_high, dtype=torch.float32, device=device)
        else:
            probs = torch.ones(num_envs, num_horizon_steps, action_dim, dtype=torch.float32, device=device)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.init_dist = Categorical(probs=probs)

        self.dist = self.init_dist

        self.selection_cache = torch.arange(num_envs, device=device).repeat(num_top_candidates, 1)

    def reset(self):
        self.dist = self.init_dist

    def sample(self, shape):
        """
        :param shape: int or tuple
        :return: actions
            if self.continuous_action: (*shape, num_envs, num_horizon_steps, action_dim)
            else: (*shape, num_envs, num_horizon_steps, 1)
        """
        if isinstance(shape, int):
            shape = (shape,)
        actions = self.dist.sample(shape)
        if self.continuous_action:
            actions = torch.clip(actions, self.action_low_device, self.action_high_device)
        else:
            actions = actions.unsqueeze(dim=-1)
        return actions

    def update(self, actions, rewards):
        """
        :param actions:
            if self.continuous_action: (num_candidates, num_envs, num_horizon_steps, action_dim) 
            else: (num_candidates, num_envs, num_horizon_steps, 1)
        :param rewards: (num_candidates, num_envs, num_horizon_steps, 1)
        :return:
        """
        sum_rewards = rewards.sum(dim=(2, 3))                           # (num_candidates, num_envs)

        # (num_top_candidates, num_envs)
        top_candidate_idxes = torch.argsort(-sum_rewards, dim=0)[:self.num_top_candidates]
        # (num_top_candidates, num_envs, num_horizon_steps, action_dim)
        top_actions = actions[top_candidate_idxes, self.selection_cache, :, :]

        if self.continuous_action:
            mu = top_actions.mean(dim=0)                                # (num_envs, num_horizon_steps, action_dim)
            std = torch.std(top_actions - mu, dim=0, unbiased=False)    # (num_envs, num_horizon_steps, action_dim)
            std = torch.clip(std, min=1e-6)
            self.dist = Normal(mu, std)
        else:
            top_actions = top_actions.squeeze(dim=-1)                   # (num_top_candidates, num_envs, num_horizon_steps)
            top_actions = F.one_hot(top_actions, self.action_dim)       # (num_top_candidates, num_envs, num_horizon_steps, action_dim)
            probs = top_actions.sum(dim=0)                              # (num_envs, num_horizon_steps, action_dim)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.dist = Categorical(probs=probs)

    def get_action(self):
        if self.continuous_action:
            action = self.dist.mean[:, 0]
            action = torch.clip(action, self.action_low_device, self.action_high_device)
        else:
            action = self.dist.probs[:, 0].argmax(dim=-1)
        return to_numpy(action)


class CEM(nn.Module):
    def __init__(self, encoder, inference, reward_predictor, params):
        super(CEM, self).__init__()

        self.encoder = encoder
        self.inference = inference
        self.reward_predictor = reward_predictor

        self.params = params
        self.device = device = params.device
        self.num_envs = num_envs = params.env_params.num_envs
        self.continuous_action = params.continuous_action
        self.action_dim = params.action_dim

        self.cem_params = cem_params = params.policy_params.cem_params

        self.action_dist = ActionDistribution(params)
        if self.continuous_action:
            self.action_low, self.action_high = params.action_spec
            self.action_mean = (self.action_low + self.action_high) / 2
            self.action_scale = (self.action_high - self.action_low) / 2

        self.num_horizon_steps = cem_params.num_horizon_steps
        self.num_iters = cem_params.num_iters
        self.num_candidates = cem_params.num_candidates

    def setup_annealing(self, step):
        pass

    def update(self, step):
        pass

    def act_randomly(self):
        if self.continuous_action:
            return self.action_mean + self.action_scale * np.random.uniform(-1, 1, (self.num_envs, self.action_dim))
        else:
            return np.random.randint(self.action_dim, size=self.num_envs)

    def act(self, obs, deterministic=False):
        """
        :param obs: (num_envs, obs_spec)
        """
        if not deterministic and not self.continuous_action:
            if np.random.rand() < self.cem_params.action_noise_eps:
                return self.act_randomly()

        self.inference.eval()
        self.action_dist.reset()

        action = self.cem(obs)

        if not deterministic and self.continuous_action:
            action_noise = self.cem_params.action_noise
            action_noise = np.random.normal(scale=action_noise, size=self.action_dim)
            action = np.clip(action + action_noise, self.action_low, self.action_high)

        if self.continuous_action:
            eef_pos = obs["robot0_eef_pos"]
            global_low, global_high = self.params.policy_params.hippo_params.skill_params.global_xyz_bound
            global_low, global_high = np.array(global_low), np.array(global_high)
            controller_scale = 0.05
            action[:, :3] = np.clip(action[:, :3],
                                    (global_low - eef_pos) / controller_scale,
                                    (global_high - eef_pos) / controller_scale)

            action = np.clip(action, self.action_low, self.action_high)

        return action

    @staticmethod
    def repeat_tensor(tensor, shape):
        """
        :param tensor: 2-dimensional state/goal tensor or None (do nothing if it's None)
        :param shape: repeat shape
        :return:
        """
        if tensor is None:
            return None

        if isinstance(shape, int):
            shape = (shape,)

        assert tensor.ndim in [1, 2]

        shape += (-1,) * tensor.ndim
        return tensor.expand(*shape)

    def cem(self, obs):
        """
        cross-entropy method
        :param obs: (num_envs, obs_spec)
        :return: action: (num_envs, action_dim)
        """
        num_candidates = self.num_candidates
        inference = self.inference
        reward_predictor = self.reward_predictor

        with torch.no_grad():
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}

            feature = self.encoder(obs)                                     # (num_envs, feature_dim)
            feature = self.repeat_tensor(feature, num_candidates)           # (num_candidates, num_envs, feature_dim)

            # assumed the goal is fixed in the episode
            goal_feature = reward_predictor.extract_goal_feature(obs)       # (num_envs, goal_dim)
            # (num_candidates, num_horizon_steps, num_envs, goal_dim)
            goal_feature = self.repeat_tensor(goal_feature, (num_candidates, self.num_horizon_steps))
            # (num_candidates, num_envs, num_horizon_steps, goal_dim)
            goal_feature = goal_feature.permute(0, 2, 1, 3)

            for i in range(self.num_iters):
                # (num_candidates, num_envs, num_horizon_steps, action_dim)
                actions = self.action_dist.sample(num_candidates)

                # (num_candidates, num_envs, num_horizon_steps, feature_dim)
                next_features = inference.predict_with_feature(feature, actions)
                pred_features = torch.cat([feature.unsqueeze(dim=2), next_features[:, :, :-1]], dim=2)
                pred_rewards = \
                    reward_predictor.pred_reward_with_feature(pred_features, actions, goal_feature, next_features)
                self.action_dist.update(actions, pred_rewards)

        return self.action_dist.get_action()                            # (action_dim,)
