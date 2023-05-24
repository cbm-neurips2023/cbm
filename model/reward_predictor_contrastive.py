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


class RewardPredictorContrastive(RewardPredictorDense):
    def __init__(self, encoder, params):
        self.init_graph(params)
        super(RewardPredictorContrastive, self).__init__(encoder, params)
        contrastive_params = self.contrastive_params
        self.num_negative_samples = contrastive_params.num_negative_samples
        self.eval_num_negative_samples = contrastive_params.eval_num_negative_samples
        self.eval_tau = contrastive_params.eval_tau

        self.mask_opt_freq = contrastive_params.mask_opt_freq
        self.full_opt_freq = contrastive_params.full_opt_freq
        self.causal_opt_freq = contrastive_params.causal_opt_freq

        self.energy_norm_reg_coef = contrastive_params.energy_norm_reg_coef
        self.sag_grad_reg_coef = contrastive_params.sag_grad_reg_coef
        self.reward_grad_reg_coef = contrastive_params.reward_grad_reg_coef

        self.num_pred_samples = contrastive_params.num_pred_samples
        self.num_pred_iters = contrastive_params.num_pred_iters
        self.pred_sigma_init = contrastive_params.pred_sigma_init
        self.pred_sigma_shrink = contrastive_params.pred_sigma_shrink

        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer

        self.update_num = 0

    def init_model(self):
        params = self.params
        self.contrastive_params = contrastive_params = self.reward_predictor_params.contrastive_params

        self.feature_dim = feature_dim = params.feature_dim
        self.action_dim = action_dim = params.action_dim

        self.goal_keys = self.encoder.goal_keys
        goal_dim = params.goal_dim

        # state feature extractor
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()

        in_dim = 1
        for out_dim in contrastive_params.feature_fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # goal (+ action) feature extractor
        if self.use_next_state:
            in_dim = goal_dim
        else:
            in_dim = goal_dim + action_dim

        self.ag_feature_fcs = self.get_mlp(in_dim, contrastive_params.feature_fc_dims)

        self.sag_encoder = self.get_mlp(2 * contrastive_params.feature_fc_dims[-1], contrastive_params.sag_encoding_fc_dims)
        self.rew_encoder = self.get_mlp(1, contrastive_params.rew_encoding_fc_dims)

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
        self.mask = torch.ones(feature_dim, dtype=torch.bool, device=device)

    def compute_energy(self, state_feature, ag_feature, rew_encoding):
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
        # (bs, encoding_dim) or (feature_dim, bs, encoding_dim)
        sag_encoding = self.sag_encoder(sag_feature)
        # (bs, 1, encoding_dim) or (feature_dim, bs, 1, encoding_dim)
        sag_encoding = sag_encoding.unsqueeze(dim=-2)

        # (bs, num_samples) or (feature_dim, bs, num_samples)
        energy = (sag_encoding * rew_encoding).sum(dim=-1)

        return energy

    def pred_energy_with_feature(self, feature, action, goal_feature, next_feature, rewards,
                                 forward_mode=("full", "mask", "causal")):
        bs = action.shape[:-1]

        if self.use_next_state:
            feature_input = next_feature                                    # (bs, feature_dim)
            ag_input = goal_feature                                         # (bs, goal_dim)
        else:
            feature_input = feature                                         # (bs, feature_dim)
            ag_input = torch.cat([action, goal_feature], dim=-1)            # (bs, action_dim + goal_dim)

        feature_input = feature_input.detach()
        feature_input.requires_grad = True
        self.input_feature = feature_input
        ag_input = ag_input.detach()
        ag_input.requires_grad = True
        self.input_action_goal = ag_input
        rewards = rewards.detach()
        rewards.requires_grad = True
        self.input_rewards = rewards

        feature_input = feature_input.T.unsqueeze(dim=-1)                   # (feature_dim, bs, 1)
        state_feature = forward_network(feature_input, self.state_feature_weights, self.state_feature_biases, use_bias=True)
        ag_feature = self.ag_feature_fcs(ag_input)                          # (bs, ag_feature_dim)
        rew_encoding = self.rew_encoder(rewards)                            # (bs, num_samples, rew_encoding_dim)

        full_energy = mask_energy = causal_energy = None

        if "full" in forward_mode:
            full_state_feature, _ = state_feature.max(dim=0)
            # (bs, num_samples)
            full_energy = self.compute_energy(full_state_feature, ag_feature, rew_encoding)

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
            mask_energy = self.compute_energy(mask_state_feature, ag_feature, rew_encoding)

        if "causal" in forward_mode:
            causal_state_feature = state_feature.clone()                    # (feature_dim, bs, out_dim)
            causal_mask = self.mask.detach().view(self.feature_dim, 1, 1)
            causal_state_feature = causal_state_feature * causal_mask       # (feature_dim, bs, out_dim)

            causal_state_feature, _ = causal_state_feature.max(dim=0)       # (feature_dim, bs, out_dim)

            # (bs, num_samples)
            causal_energy = self.compute_energy(causal_state_feature, ag_feature, rew_encoding)

        return full_energy, mask_energy, causal_energy

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

    @staticmethod
    def nce_loss(energy):
        """
        :param energy: (bs, 1 + num_negative_samples) or 
                       (feature_dim, bs, 1 + num_negative_samples)
        :return:
            loss: scalar
        """
        if energy.ndim == 2:
            return -F.log_softmax(energy, dim=-1)[..., 0].mean()
        elif energy.ndim == 3:
            return -F.log_softmax(energy, dim=-1)[..., 0].sum(dim=0).mean()
        else:
            raise NotImplementedError

    def energy_norm_loss(self, energy):
        """
        :param energy: (bs, 1 + num_negative_samples)
        :return:
            loss: scalar
        """
        energy_sq = (energy ** 2).sum(dim=-1).mean()
        energy_abs = energy.abs().sum(dim=-1).mean()

        return energy_sq * self.energy_norm_reg_coef, energy_abs

    def grad_loss(self, energy):
        """
        :param energy: (bs, 1 + num_negative_samples)
        :return:
            loss: scalar
        """
        feature = self.input_feature                        # (bs, feature_dim)
        action_goal = self.input_action_goal                # (bs, goal_dim / action_dim + goal_dim)
        rewards = self.input_rewards                        # (bs, 1 + num_samples, 1)

        grads_penalty = sag_grad_abs = 0

        nce = F.log_softmax(energy, dim=-1)[..., 0]         # (bs)
        feature_grad, action_goal_grad = torch.autograd.grad(nce.sum(), [feature, action_goal], create_graph=True)

        grads_penalty += feature_grad.pow(2).sum(dim=-1).mean() * self.sag_grad_reg_coef
        sag_grad_abs += feature_grad.abs().sum(dim=-1).mean()

        grads_penalty += action_goal_grad.pow(2).sum(dim=-1).mean() * self.sag_grad_reg_coef
        sag_grad_abs += action_goal_grad.abs().sum(dim=-1).mean()

        reward_grad = torch.autograd.grad(energy.sum(), rewards, create_graph=True)[0]
        grads_penalty += reward_grad.pow(2).sum(dim=(-2, -1)).mean() * self.reward_grad_reg_coef
        reward_grad_abs = reward_grad.abs().sum(dim=(-2, -1)).mean()

        return grads_penalty, sag_grad_abs, reward_grad_abs

    def update_loss_and_logging(self, energy, logging_prefix, loss, loss_detail):
        nce_loss = self.nce_loss(energy)
        energy_norm_loss, energy_norm = self.energy_norm_loss(energy)
        energy_grad_loss, sa_grad_norm, delta_grad_norm = self.grad_loss(energy)

        loss += nce_loss + energy_norm_loss + energy_grad_loss

        loss_detail[logging_prefix + "_nce_loss"] = nce_loss
        loss_detail[logging_prefix + "_energy_norm"] = energy_norm
        loss_detail[logging_prefix + "_sa_grad_norm"] = sa_grad_norm
        loss_detail[logging_prefix + "_delta_grad_norm"] = delta_grad_norm

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

        bs = action.shape[:-1]

        feature = self.encoder(obs)
        next_feature = self.encoder(next_obs)
        goal_feature = self.extract_goal_feature(obs)

        # (bs, num_negative_samples, 1)
        neg_rewards = torch.rand(bs + (self.num_negative_samples, 1), device=self.device)
        reward = reward.unsqueeze(dim=-2)                           # (bs, 1, 1)
        rewards = torch.cat([reward, neg_rewards], dim=-2)          # (bs, 1 + num_negative_samples, 1)

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
        full_energy, mask_energy, causal_energy = \
            self.pred_energy_with_feature(feature, action, goal_feature, next_feature, rewards, forward_mode)

        loss = 0
        loss_detail = {}

        if "mask" in forward_mode:
            loss, loss_detail = self.update_loss_and_logging(mask_energy, "mask", loss, loss_detail)

        if "full" in forward_mode:
            loss, loss_detail = self.update_loss_and_logging(full_energy, "full", loss, loss_detail)

        if "causal" in forward_mode:
            if opt_causal:
                loss, loss_detail = self.update_loss_and_logging(causal_energy, "causal", loss, loss_detail)

            if self.use_prioritized_buffer:
                priority = 1 - F.softmax(causal_energy, dim=-1)[..., 0]     # (bs,)
                loss_detail["priority"] = priority

        self.backprop(loss, loss_detail)

        return loss_detail

    @staticmethod
    def compute_cmi(energy, cond_energy, unbiased=True):
        """
        https://arxiv.org/pdf/2106.13401, proposition 3
        :param energy: (feature_dim, bs, 1 + num_negative_samples)
            notice that bs can be a multi-dimensional batch size
        :param cond_energy: (feature_dim, bs, 1 + num_negative_samples)
        :return: cmi: (feature_dim,)
        """
        feature_dim = energy.shape[0]
        pos_cond_energy = cond_energy[..., 0]               # (feature_dim, bs)

        K = energy.shape[-1]                                # 1 + num_negative_samples
        neg_energy = energy[..., 1:]                        # (feature_dim, bs, num_negative_samples)
        neg_cond_energy = cond_energy[..., 1:]              # (feature_dim, bs, num_negative_samples)

        log_w_neg = F.log_softmax(neg_energy, dim=-1)       # (feature_dim, bs, num_negative_samples)
        # (feature_dim, bs, num_negative_samples)
        weighted_neg_cond_energy = np.log(K - 1) + log_w_neg + neg_cond_energy
        # (feature_dim, bs, 1 + num_negative_samples)
        cond_energy = torch.cat([pos_cond_energy.unsqueeze(dim=-1), weighted_neg_cond_energy], dim=-1)
        log_denominator = -np.log(K) + torch.logsumexp(cond_energy, dim=-1)         # (feature_dim, bs)
        cmi = pos_cond_energy - log_denominator                                     # (feature_dim, bs)

        cmi = cmi.view(feature_dim, -1).mean(dim=-1)
        return cmi

    def update_mask(self, obs, action, next_obs, reward):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param next_obses: {obs_i_key: (bs, obs_i_shape)}
        :param reward: (bs, 1)
        :return: {"loss_name": loss_value}
        """
        bs = action.shape[:-1]

        # (bs, num_negative_samples, 1)
        neg_rewards = torch.rand(bs + (self.eval_num_negative_samples, 1), device=self.device)
        reward = reward.unsqueeze(dim=-2)                           # (bs, 1, 1)
        rewards = torch.cat([reward, neg_rewards], dim=-2)          # (bs, 1 + num_negative_samples, 1)

        eval_details = {}

        cmi = []
        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obs)
            goal_feature = self.extract_goal_feature(obs)

            full_energy, mask_energy, causal_energy = \
                self.pred_energy_with_feature(feature, action, goal_feature, next_feature, rewards)

            mask_nce_loss = self.nce_loss(mask_energy)
            full_nce_loss = self.nce_loss(full_energy)
            causal_nce_loss = self.nce_loss(causal_energy)
            eval_details = {"mask_nce_loss": mask_nce_loss,
                            "full_nce_loss": full_nce_loss,
                            "causal_nce_loss": causal_nce_loss}

            mask_cond_energy = full_energy.unsqueeze(dim=0) - mask_energy

            cmi = self.compute_cmi(mask_energy, mask_cond_energy)               # (feature_dim)

        eval_tau = self.eval_tau
        self.mask_CMI = self.mask_CMI * eval_tau + cmi * (1 - eval_tau)
        self.mask = self.mask_CMI >= self.CMI_threshold

        return eval_details

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return:
            pred_next_feature: (bs, feature_dim)
        """
        bs = action.shape[:-1]
        feature_dim = self.feature_dim
        action_dim = self.action_dim
        num_pred_samples = self.num_pred_samples
        sigma = self.pred_sigma_init

        rewards = torch.rand(bs + (num_pred_samples, 1), device=self.device)
        rewards, _ = torch.sort(rewards, dim=1)

        for i in range(self.num_pred_iters):
            forward_mode = ("causal",)

            full_energy, mask_energy, causal_energy = \
                self.pred_energy_with_feature(feature, action, goal_feature, next_feature, rewards, forward_mode)
            energy = causal_energy                                          # (bs, num_pred_samples)

            if i != self.num_pred_iters - 1:
                dist = Categorical(logits=energy)
                idxes = dist.sample([num_pred_samples])                     # (num_pred_samples, bs)
                idxes = idxes.T                                             # (bs, num_pred_samples)

                # (bs, num_pred_samples)
                idxes = idxes.unsqueeze(dim=-1)
                rewards = torch.gather(rewards, -2, idxes)
                noise = torch.rand(bs + (num_pred_samples, 1), device=self.device)
                rewards += noise * sigma
                rewards = torch.clip(rewards, 0, 1)

                sigma *= self.pred_sigma_shrink

        argmax_idx = torch.argmax(energy, dim=-1, keepdim=True)             # (bs, 1)
        argmax_idx = argmax_idx.unsqueeze(dim=-1)
        # (bs, feature_dim)
        rewards = torch.gather(rewards, -2, argmax_idx)[..., 0, :]

        return rewards

    def get_mask(self):
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
            self.CMI_threshold = self.contrastive_params.CMI_threshold
            self.mask = self.mask_CMI >= self.CMI_threshold
