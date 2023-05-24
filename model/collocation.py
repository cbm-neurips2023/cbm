import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class Collocation:
    def __init__(self, encoder, inference, reward_predictor, params):

        self.encoder = encoder
        self.inference = inference
        self.reward_predictor = reward_predictor

        self.params = params
        self.device = device = params.device
        self.collocation_params = collocation_params = params.policy_params.collocation_params

        inference_algo = params.training_params.inference_algo
        self.use_contrastive = "contrastive" in inference_algo
        if not self.use_contrastive:
            raise NotImplementedError

        if self.continuous_state:
            self.action_low, self.action_high = params.action_spec
            self.action_mean = (self.action_low + self.action_high) / 2
            self.action_scale = (self.action_high - self.action_low) / 2
            assert (self.action_mean == 0).all(), "implementation assumes the action space is certained at 0"
            self.action_low_tensor = torch.tensor(self.action_low, device=device)
            self.action_high_tensor = torch.tensor(self.action_high, device=device)
            self.action_scale_tensor = torch.tensor(self.action_scale, device=device)
        else:
            raise NotImplementedError

        self.action_dim = params.action_dim
        self.feature_dim = params.feature_dim

        self.num_horizon_steps = collocation_params.num_horizon_steps
        self.num_plans = collocation_params.num_plans
        self.num_opt_iters = collocation_params.num_opt_iters

        self.lam_opt_freq = collocation_params.lam_opt_freq

        self.lam_dyn_lr = collocation_params.lam_dyn_lr
        self.lam_action_lr = collocation_params.lam_action_lr

        self.lam_dyn_init = collocation_params.lam_dyn_init
        self.lam_action_init = collocation_params.lam_action_init

        self.dyn_thre = collocation_params.dyn_thre
        self.action_thre = collocation_params.action_thre

    def act_randomly(self):
        if self.continuous_state:
            return self.action_mean + self.action_scale * np.random.uniform(-1, 1, self.action_scale.shape)
        else:
            raise NotImplementedError

    def act(self, obs, deterministic=False):
        """
        :param obs: (obs_spec)
        """
        self.inference.eval()

        action = self.collocate(obs)

        if self.continuous_state:
            eef_pos = obs["robot0_eef_pos"]
            global_low, global_high = self.params.policy_params.cem_params.hippo_params.skill_params.global_xyz_bound
            controller_scale = 0.05
            action[:3] = np.clip(action[:3],
                                 (global_low - eef_pos) / controller_scale,
                                 (global_high - eef_pos) / controller_scale)

        action = np.clip(action, self.action_low, self.action_high)

        return action

    def collocate(self, obs):
        device = self.device

        inference = self.inference
        reward_predictor = self.reward_predictor

        action_dim = self.action_dim
        feature_dim = self.feature_dim
        num_plans = self.num_plans
        num_horizon_steps = self.num_horizon_steps

        action_thre = self.action_thre
        delta_state_thre = self.delta_state_thre

        obs = postprocess_obs(preprocess_obs(obs, self.params))
        obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
        feature = self.encoder(obs)                         # (bs, feature_dim)
        need_squeeze = False
        if feature.ndim == 1:
            need_squeeze = True
            feature = feature.unsqueeze(dim=0)
        feature = feature.expand(num_plans, 1, -1, -1)      # (num_plans, 1, bs, feature_dim)

        bs = feature.shape[1]

        # assumed the goal is fixed in the episode
        goal_feature = self.extract_goal_feature(obs)       # (bs, goal_dim)
        if goal_feature is not None:
            if goal_feature.ndim == 1:
                goal_feature = goal_feature.unsqueeze(dim=0)
            # (num_plans, num_horizon_steps, bs, goal_dim)
            goal_feature = self.repeat_tensor(goal_feature, (num_plans, num_horizon_steps))

        # variables to optimize
        plan = torch.randn(num_plans, num_horizon_steps, feature_dim + action_dim,
                           device=device, requires_grad=True)
        lambda_dyn = torch.ones(num_plans, num_horizon_steps, bs, feature_dim) * self.lam_dyn_init
        lambda_action = torch.ones(num_plans, num_horizon_steps, bs, action_dim) * self.lam_action_init

        for i in range(self.num_opt_iters):
            plan = self.opt_step(plan, feature, goal_feature, lambda_dyn, lambda_action)

            next_features, actions = torch.split(plan, [feature_dim, action_dim], dim=-1)
            # (num_plans, num_horizon_steps + 1, bs, feature_dim)
            features = torch.concat([feature, next_features[:, :-1]], dim=1)
            # (num_plans, num_horizon_steps, bs, feature_dim)
            pred_next_features = inference.forward_step(features, actions)
            # (num_plans, num_horizon_steps, bs, feature_dim)
            dyn_viol = (next_features - pred_next_features).pow(2)
            # (num_plans, num_horizon_steps, bs, action_dim)
            action_viol = (actions - self.action_scale_tensor).pow(2)

            if i % self.lam_opt_freq == 0:
                lambda_dyn += self.lam_dyn_lr * torch.log(dyn_viol / self.dyn_thre + 0.01)
                lambda_action += self.lam_action_lr * torch.log(dyn_viol / self.action_thre + 0.01)

        # select best plan
        # (num_plans, num_horizon_steps, bs, action_dim)
        actions = torch.clip(actions, self.action_low_tensor, self.action_high_tensor)
        # (num_plans, num_horizon_steps, bs, feature_dim)
        pred_next_features = inference.forward_with_feature(feature[:, 0], actions)
        # (num_plans, num_horizon_steps, bs, feature_dim)
        pred_features = torch.concat([feature, pred_next_features[:, :-1]], dim=1)
        # (num_plans, num_horizon_steps, bs, 1)
        rewards = reward_predictor.pred_reward_with_feature(pred_features, actions, goal_feature, pred_next_features)
        rewards = rewards.sum(dim=(1, 3))                                       # (num_plans, bs)
        best_plan_idx = rewards.argmax(dim=0)                                   # (bs,)
        best_plan = actions[best_plan_idx, :, torch.arange(bs, device=device)]  # (num_horizon_steps, bs, action_dim)

        return best_plan[0]                                                     # (bs, action_dim)
