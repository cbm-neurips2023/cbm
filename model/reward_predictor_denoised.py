import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.reward_predictor import RewardPredictorDense
from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class RewardPredictorDenoised(RewardPredictorDense):
    def __init__(self, encoder, denoised_mask, params):
        super(RewardPredictorDenoised, self).__init__(encoder, params, denoised_mask=denoised_mask)

    def init_model(self):
        params = self.params
        dense_params = self.reward_predictor_params.dense_params

        self.feature_dim = feature_dim = params.feature_dim
        self.action_dim = action_dim = params.action_dim

        self.goal_keys = self.encoder.goal_keys
        goal_dim = params.goal_dim

        if self.use_next_state:
            in_dim = feature_dim + goal_dim
        else:
            in_dim = feature_dim + goal_dim + action_dim

        self.x_fcs = self.get_mlp(in_dim, dense_params.fc_dims + [1], dense_params.activations + ["relu"])
        self.y_fcs = self.get_mlp(in_dim, dense_params.fc_dims + [1], dense_params.activations + ["relu"])

    def setup_annealing(self, step):
        self.denoised_mask.setup_annealing(step)

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature, mask=None, use_x=True):
        if mask is None:
            mask = self.get_mask(return_bool=True).float()

        if self.use_next_state:
            inputs = [next_feature * mask]
        else:
            inputs = [feature * mask, action]

        if goal_feature is not None:
            inputs.append(goal_feature)

        inputs = torch.cat(inputs, dim=-1)

        if use_x:
            pred_reward = self.x_fcs(inputs)
        else:
            pred_reward = self.y_fcs(inputs)
        return pred_reward

    def update(self, obs, action, next_obs, reward, eval=False):
        feature = self.encoder(obs)
        goal_feature = self.extract_goal_feature(obs)
        next_feature = self.encoder(next_obs)

        bs = action.shape[:-1]
        mask = self.denoised_mask(bs)
        x_mask, y_mask, _ = mask.unbind(dim=-1)

        x_pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature, x_mask, use_x=True)
        y_pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature, y_mask, use_x=False)
        pred_loss = torch.abs(x_pred_reward + y_pred_reward - reward).squeeze(dim=-1).mean()
        re_reg_loss = self.denoised_mask.get_prob()[..., 0].mean() * self.reward_predictor_params.denoised_params.x_reg_coef
        loss = pred_loss + re_reg_loss

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_detail = {"pred_loss": loss}

        return loss_detail

    def get_mask(self, return_bool=False):
        prob = self.denoised_mask.get_prob()
        if return_bool:
            return prob.argmax(dim=-1) == 0
        else:
            return prob[..., 0]

    def get_threshold(self):
        return 0.5
