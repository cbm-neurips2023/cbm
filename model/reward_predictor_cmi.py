import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model.reward_predictor import RewardPredictorDense
from model.inference_utils import reset_layer, forward_network
from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class RewardPredictorCMI(RewardPredictorDense):
    def __init__(self, encoder, params):
        self.init_graph(params)
        super(RewardPredictorCMI, self).__init__(encoder, params)
        cmi_params = self.cmi_params
        self.eval_tau = cmi_params.eval_tau

        self.mask_opt_freq = cmi_params.mask_opt_freq
        self.full_opt_freq = cmi_params.full_opt_freq
        self.causal_opt_freq = cmi_params.causal_opt_freq

        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer
        self.mask_need_update = True

        self.update_num = 0

    def init_model(self):
        params = self.params
        self.cmi_params = cmi_params = self.reward_predictor_params.cmi_params

        self.feature_dim = feature_dim = params.feature_dim
        self.action_dim = action_dim = params.action_dim

        self.goal_keys = self.encoder.goal_keys
        goal_dim = params.goal_dim

        # state feature extractor
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()

        in_dim = 1
        for out_dim in cmi_params.feature_fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # goal (+ action) feature extractor
        if self.use_next_state:
            in_dim = goal_dim
        else:
            in_dim = goal_dim + action_dim

        self.ag_feature_fcs = self.get_mlp(in_dim, cmi_params.feature_fc_dims)

        self.rew_predictor = self.get_mlp(2 * cmi_params.feature_fc_dims[-1], cmi_params.rew_predictor_fc_dims + [1])

        training_masks = []
        for i in range(feature_dim):
            training_masks.append(self.get_eval_mask((1,), i))

        # 1st feature_dim: variable to predict, 2nd feature_dim: input variable to ignore
        training_masks = torch.cat(training_masks, dim=0)
        self.training_masks = training_masks.view(feature_dim, feature_dim, 1, 1)

    def reset_params(self):
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            assert w.ndim == b.ndim == 3
            for i in range(w.shape[0]):
                reset_layer(w[i], b[i])

    def init_graph(self, params):
        feature_dim = params.feature_dim
        device = params.device
        self.CMI_threshold = params.reward_predictor_params.cmi_params.CMI_threshold

        # used for masking diagonal elementss
        self.mask_CMI = torch.ones(feature_dim, device=device) * self.CMI_threshold
        self.prev_mask = self.mask = torch.ones(feature_dim, dtype=torch.bool, device=device)

    def pred_reward_with_sag_feature(self, state_feature, ag_feature):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param state_feature: (bs, sa_feature_dim) or (feature_dim, bs, sa_feature_dim)
        :param ag_feature: (bs, ag_feature_dim)
        :param rew_encoding: (bs, num_samples, encoding_dim)
        :return: energy: (bs, num_samples)
        """
        assert state_feature.ndim in [2, 3]
        is_mask_feature = state_feature.ndim == 3

        if is_mask_feature:
            ag_feature = ag_feature.unsqueeze(dim=0).expand(self.feature_dim, -1, -1)

        # (bs, sa_feature_dim + ag_feature_dim) or (feature_dim, bs, sa_feature_dim + ag_feature_dim)
        sag_feature = torch.cat([state_feature, ag_feature], dim=-1)
        # (bs, 1) or (feature_dim, bs, 1)
        pred_reward = self.rew_predictor(sag_feature)

        return pred_reward

    def pred_reward_with_feature_helper(self, feature, action, goal_feature, next_feature,
                                        forward_mode=("full", "mask", "causal")):
        bs = action.shape[:-1]

        if self.use_next_state:
            feature_input = next_feature                                    # (bs, feature_dim)
            ag_input = goal_feature                                         # (bs, goal_dim)
        else:
            feature_input = feature                                         # (bs, feature_dim)
            ag_input = torch.cat([action, goal_feature], dim=-1)            # (bs, action_dim + goal_dim)

        feature_input = feature_input.T.unsqueeze(dim=-1)                   # (feature_dim, bs, 1)
        state_feature = forward_network(feature_input, self.state_feature_weights, self.state_feature_biases)
        ag_feature = self.ag_feature_fcs(ag_input)                          # (bs, ag_feature_dim)

        full_pred = mask_pred = causal_pred = None

        if "full" in forward_mode:
            full_state_feature, _ = state_feature.max(dim=0)
            # (bs, num_samples)
            full_pred = self.pred_reward_with_sag_feature(full_state_feature, ag_feature)

        if "mask" in forward_mode:
            mask_state_feature = state_feature.clone()
            if self.training:
                mask = self.get_training_mask(bs).T                         # (feature_dim, bs)
                mask = mask.unsqueeze(dim=-1)                               # (feature_dim, bs, 1)
            else:
                mask = self.training_masks                                  # (feature_dim, feature_dim, 1, 1)
                mask_state_feature = mask_state_feature.unsqueeze(dim=0)    # (1, feature_dim, bs, out_dim)

            # (feature_dim, feature_dim, bs, out_dim) or (feature_dim, bs, out_dim)
            mask_state_feature = mask_state_feature * mask
            mask_state_feature, _ = mask_state_feature.max(dim=-3)

            # (bs, num_samples) or (feature_dim, bs, num_samples)
            mask_pred = self.pred_reward_with_sag_feature(mask_state_feature, ag_feature)

        if "causal" in forward_mode:
            causal_state_feature = state_feature.clone()                    # (feature_dim, bs, out_dim)
            causal_mask = self.mask.detach().view(self.feature_dim, 1, 1)
            causal_state_feature = causal_state_feature * causal_mask       # (feature_dim, bs, out_dim)

            causal_state_feature, _ = causal_state_feature.max(dim=0)       # (feature_dim, bs, out_dim)

            # (bs, num_samples)
            causal_pred = self.pred_reward_with_sag_feature(causal_state_feature, ag_feature)

        return full_pred, mask_pred, causal_pred

    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs feature_dim), idxes of state variable to drop
            notice that bs can be a multi-dimensional batch size
        :return: (bs, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.feature_dim)
        bool_mask = int_mask == 0
        return bool_mask

    def get_training_mask(self, bs):
        # uniformly select one state variable to omit when predicting the next time step value
        if isinstance(bs, int):
            bs = (bs,)

        idxes = torch.randint(self.feature_dim, bs, device=self.device)
        return self.get_mask_by_id(idxes)  # (bs, feature_dim)

    def get_eval_mask(self, bs, i):
        # omit i-th state variable or the action when predicting the next time step value

        if isinstance(bs, int):
            bs = (bs,)

        feature_dim = self.feature_dim
        idxes = torch.full(size=bs, fill_value=i, dtype=torch.int64, device=self.device)
        return self.get_mask_by_id(idxes)  # (bs, feature_dim)

    def update_loss_and_logging(self, pred_reward, reward, logging_prefix, loss, loss_detail):
        # pred_reward: (bs, 1)
        pred_loss = torch.abs(pred_reward - reward).sum(dim=-1).mean()

        loss += pred_loss
        loss_detail[logging_prefix + "_pred_loss"] = pred_loss

        return loss, loss_detail

    def update(self, obs, action, next_obs, reward, eval=False):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param next_obses: {obs_i_key: (bs, obs_i_shape)}
        :param reward: (bs, 1)
        :return: {"loss_name": loss_value}
        """
        if eval:
            return self.update_mask(obs, action, next_obs, reward)

        self.update_num += 1

        feature = self.encoder(obs)
        next_feature = self.encoder(next_obs)
        goal_feature = self.extract_goal_feature(obs)

        # decide which loss to compute
        forward_mode = []

        opt_mask = self.mask_opt_freq > 0 and self.update_num % self.mask_opt_freq == 0
        opt_full = self.full_opt_freq > 0 and self.update_num % self.full_opt_freq == 0
        opt_causal = self.causal_opt_freq > 0 and self.update_num % self.causal_opt_freq == 0

        if opt_mask:
            forward_mode.append("mask")
        if opt_full:
            forward_mode.append("full")
        if self.use_prioritized_buffer or opt_causal:
            forward_mode.append("causal")

        # (bs, 1 + num_negative_samples)
        full_pred, mask_pred, causal_pred = \
            self.pred_reward_with_feature_helper(feature, action, goal_feature, next_feature, forward_mode)

        loss = 0
        loss_detail = {}

        if "full" in forward_mode:
            loss, loss_detail = self.update_loss_and_logging(full_pred, reward, "full", loss, loss_detail)

        if "mask" in forward_mode:
            loss, loss_detail = self.update_loss_and_logging(mask_pred, reward, "mask", loss, loss_detail)

        if "causal" in forward_mode:
            if opt_causal:
                loss, loss_detail = self.update_loss_and_logging(causal_pred, reward, "causal", loss, loss_detail)

            if self.use_prioritized_buffer:
                priority = torch.abs(causal_pred - reward)[..., 0]      # (bs,)
                loss_detail["priority"] = priority

        self.backprop(loss)

        return loss_detail

    def update_mask(self, obs, action, next_obs, reward):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param next_obses: {obs_i_key: (bs, obs_i_shape)}
        :param reward: (bs, 1)
        :return: {"loss_name": loss_value}
        """
        eval_details = {}

        cmi = []
        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obs)
            goal_feature = self.extract_goal_feature(obs)

            forward_mode = ("full", "mask", "causal")
            full_pred, mask_pred, causal_pred = \
                self.pred_reward_with_feature_helper(feature, action, goal_feature, next_feature, forward_mode)

            full_pred_loss = torch.abs(full_pred - reward).sum(dim=-1).mean()
            mask_pred_loss = torch.abs(mask_pred - reward).sum(dim=(0, -1)).mean()
            causal_pred_loss = torch.abs(causal_pred - reward).sum(dim=-1).mean()
            eval_details = {"mask_pred_loss": mask_pred_loss,
                            "full_pred_loss": full_pred_loss,
                            "causal_pred_loss": causal_pred_loss}

            cmi = torch.abs(mask_pred - reward) - torch.abs(full_pred - reward)
            cmi = cmi.view(self.feature_dim, -1).mean(dim=-1)

        eval_tau = self.eval_tau
        self.mask_CMI = self.mask_CMI * eval_tau + cmi * (1 - eval_tau)
        self.mask = self.mask_CMI >= self.CMI_threshold

        if (self.mask != self.prev_mask).any():
            self.mask_need_update = True
            self.prev_mask = self.mask

        return eval_details

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return:
            pred_next_feature: (bs, feature_dim)
        """
        with torch.no_grad():
            forward_mode = ("causal",)
            _, _, causal_pred = \
                self.pred_reward_with_feature_helper(feature, action, goal_feature, next_feature, forward_mode)

        return causal_pred

    def get_mask(self, return_bool=False):
        if return_bool:
            return self.mask_CMI >= self.CMI_threshold
        else:
            return self.mask_CMI

    def get_threshold(self):
        return self.CMI_threshold

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("reward predictor loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.CMI_threshold = self.cmi_params.CMI_threshold
            self.mask = self.mask_CMI >= self.CMI_threshold
