import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.reward_predictor import RewardPredictorDense
from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class RewardPredictorTIA(RewardPredictorDense):
    def __init__(self, encoder, tia_mask, params):
        super(RewardPredictorTIA, self).__init__(encoder, params, tia_mask=tia_mask)
        self.re_optimizer = optim.Adam(list(self.re_fcs.parameters()) + list(tia_mask.parameters()),
                                       lr=self.reward_predictor_params.lr)
        self.ir_rew_optimizer = optim.Adam(self.ir_fcs.parameters(), lr=self.reward_predictor_params.lr)

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

        self.re_fcs = self.get_mlp(in_dim, dense_params.fc_dims + [1], dense_params.activations + ["relu"])
        self.ir_fcs = self.get_mlp(in_dim, dense_params.fc_dims + [1], dense_params.activations + ["relu"])

    def setup_annealing(self, step):
        self.tia_mask.setup_annealing(step)

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature, mask=None, use_relevant=True):
        if mask is None:
            mask = self.get_mask(return_bool=True).float()
            if not use_relevant:
                raise ValueError("irrelevant predictor should only be used for training with provided mask")

        if self.use_next_state:
            inputs = [next_feature * mask]
        else:
            inputs = [feature * mask, action]

        if goal_feature is not None:
            inputs.append(goal_feature)

        inputs = torch.cat(inputs, dim=-1)

        if use_relevant:
            pred_reward = self.re_fcs(inputs)
        else:
            pred_reward = self.ir_fcs(inputs)
        return pred_reward

    def update(self, obs, action, next_obs, reward, eval=False):
        feature = self.encoder(obs)
        goal_feature = self.extract_goal_feature(obs)
        next_feature = self.encoder(next_obs)

        bs = action.shape[:-1]
        mask = self.tia_mask(bs)
        re_mask, ir_mask = mask.unbind(dim=-1)
        if not eval:
            for _ in range(self.reward_predictor_params.tia_params.num_irrelevant_opt_steps):
                pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature,
                                                            ir_mask.detach(), use_relevant=False)
                self.ir_rew_optimizer.zero_grad()
                loss = torch.abs(pred_reward - reward).squeeze(dim=-1).mean()
                loss.backward()
                self.ir_rew_optimizer.step()

        re_pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature,
                                                       re_mask, use_relevant=True)
        ir_pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature,
                                                       ir_mask, use_relevant=False)

        re_pred_loss = torch.abs(re_pred_reward - reward).squeeze(dim=-1).mean()
        ir_pred_loss = torch.abs(ir_pred_reward - reward).squeeze(dim=-1).mean()
        re_reg_loss = self.tia_mask.get_prob()[..., 0].mean() * self.reward_predictor_params.tia_params.relevant_reg_coef
        loss = re_pred_loss - ir_pred_loss + re_reg_loss

        if not eval:
            self.ir_rew_optimizer.zero_grad()
            self.re_optimizer.zero_grad()
            loss.backward()
            self.re_optimizer.step()

        loss_detail = {"relevant_pred_loss": re_pred_loss,
                       "irrelevant_pred_loss": ir_pred_loss}

        return loss_detail

    def get_mask(self, return_bool=False):
        prob = self.tia_mask.get_prob()[..., 0]
        if return_bool:
            return prob > 0.5
        else:
            return prob

    def get_threshold(self):
        return 0.5

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "re_optimizer": self.re_optimizer.state_dict(),
                    "ir_rew_optimizer": self.ir_rew_optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("reward predictor loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.re_optimizer.load_state_dict(checkpoint["re_optimizer"])
            self.ir_rew_optimizer.load_state_dict(checkpoint["ir_rew_optimizer"])
