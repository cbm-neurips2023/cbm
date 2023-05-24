import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.contrastive import Contrastive
from model.contrastive_cmi import ContrastiveCMI
from model.inference_utils import reset_layer, forward_network, get_state_abstraction
from utils.utils import to_numpy


class ContrastiveMask(ContrastiveCMI):
    def __init__(self, encoder, decoder, mask, params):
        super(ContrastiveMask, self).__init__(encoder, decoder, params)
        assert not self.parallel_sample
        self.mask = mask
        parameters = [parameter for parameter in self.sa_encoder_weights] + \
                     [parameter for parameter in self.sa_encoder_biases] + \
                     list(self.mask.parameters())
        self.optimizer = optim.Adam(parameters, lr=self.contrastive_params.lr)

    def setup_annealing(self, step):
        self.mask.setup_annealing(step)

    def compute_energy(self, sa_feature, delta_encoding):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_encoding: (feature_dim, bs, num_samples, encoding_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        feature_dim = self.feature_dim
        bs = sa_feature.shape[-2]

        # (feature_dim, bs, out_dim)
        sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)

        sa_encoding = sa_encoding.unsqueeze(dim=-2)
        energy = (sa_encoding * delta_encoding).sum(dim=-1)         # (feature_dim, bs, num_samples)
        energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    def extract_sa_feature(self, sa_feature):
        if self.aggregation == "max":
            sa_feature, _ = sa_feature.max(dim=1)                       # (feature_dim, bs, out_dim)
        elif self.aggregation == "mean":
            sa_feature = sa_feature.mean(dim=1)                         # (feature_dim, bs, out_dim)
        else:
            raise NotImplementedError
        return sa_feature

    def forward_step(self, feature, action, delta_features, mask=None):
        """
        :param feature: (bs, feature_dim)
            notice that bs must be 1D
        :param action: (bs, action_dim)
        :param delta_features: (bs, num_samples, feature_dim)
        :param mask: (bs, feature_dim, num_partitions)
        :return: energy (bs, num_samples, feature_dim)
        """
        bs, _, feature_dim = delta_features.shape

        # extract features from the action, the states, and the next state candidates
        action_feature = self.extract_action_feature(action)            # (feature_dim, 1, bs, out_dim)
        state_feature = self.extract_state_feature(feature)             # (feature_dim, feature_dim, bs, out_dim)
        # (feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = self.extract_delta_state_encoding(delta_features)

        if mask is None:
            # created for evaluation
            assert not self.training
            assert self.training == self.mask.training
            mask = self.mask(bs)

        num_partitions = mask.shape[-1]
        if num_partitions == 2:
            re_mask, ir_mask = mask.unbind(dim=-1)                      # (bs, feature_dim), (bs, feature_dim)
            re_mask = re_mask.T.unsqueeze(dim=-1)                       # (feature_dim, bs, 1)
            ir_mask = ir_mask.T.unsqueeze(dim=-1)                       # (feature_dim, bs, 1)

            # (feature_dim, feature_dim + 1, bs, out_dim)
            re_sa_feature = torch.cat([state_feature * re_mask, action_feature], dim=1)
            # (feature_dim, bs, out_dim)
            re_sa_feature = self.extract_sa_feature(re_sa_feature)
            # (bs, num_samples, feature_dim)
            re_sa_energy = self.compute_energy(re_sa_feature, delta_encoding)

            # (feature_dim, feature_dim + 1, bs, out_dim)
            ir_sa_feature = torch.cat([state_feature * ir_mask, action_feature], dim=1)
            # (feature_dim, bs, out_dim)
            ir_sa_feature = self.extract_sa_feature(ir_sa_feature)
            # (bs, num_samples, feature_dim)
            ir_sa_energy = self.compute_energy(ir_sa_feature, delta_encoding)

            re_mask = re_mask.detach().permute(1, 2, 0)                          # (bs, 1, feature_dim)
            ir_mask = ir_mask.detach().permute(1, 2, 0)                          # (bs, 1, feature_dim)

            energy = re_sa_energy * re_mask + ir_sa_energy * ir_mask
        elif num_partitions == 3:
            x_mask, y_mask, z_mask = mask.unbind(dim=-1)                # (bs, feature_dim), (bs, feature_dim)
            x_mask = x_mask.T.unsqueeze(dim=-1)                         # (feature_dim, bs, 1)
            y_mask = y_mask.T.unsqueeze(dim=-1)                         # (feature_dim, bs, 1)
            z_mask = z_mask.T.unsqueeze(dim=-1)                         # (feature_dim, bs, 1)

            # (feature_dim, feature_dim + 1, bs, out_dim)
            x_sa_feature = torch.cat([state_feature * x_mask, action_feature], dim=1)
            # (feature_dim, bs, out_dim)
            x_sa_feature = self.extract_sa_feature(x_sa_feature)
            # (bs, num_samples, feature_dim)
            x_sa_energy = self.compute_energy(x_sa_feature, delta_encoding)

            # (feature_dim, feature_dim + 1, bs, out_dim)
            y_sa_feature = state_feature * y_mask
            # (feature_dim, bs, out_dim)
            y_sa_feature = self.extract_sa_feature(y_sa_feature)
            # (bs, num_samples, feature_dim)
            y_sa_energy = self.compute_energy(y_sa_feature, delta_encoding)

            # (feature_dim, feature_dim + 1, bs, out_dim)
            z_sa_feature = state_feature * z_mask
            # (feature_dim, bs, out_dim)
            z_sa_feature = self.extract_sa_feature(z_sa_feature)
            # (bs, num_samples, feature_dim)
            z_sa_energy = self.compute_energy(z_sa_feature, delta_encoding)

            x_mask = x_mask.detach().permute(1, 2, 0)                            # (bs, 1, feature_dim)
            y_mask = y_mask.detach().permute(1, 2, 0)                            # (bs, 1, feature_dim)
            z_mask = z_mask.detach().permute(1, 2, 0)                            # (bs, 1, feature_dim)

            energy = x_sa_energy * x_mask + y_sa_energy * y_mask + z_sa_energy * z_mask
        else:
            raise NotImplementedError

        return energy

    def forward_with_feature(self, feature, actions, next_features, neg_delta_features, eval=False):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param actions:  (bs, num_pred_steps, action_dim)
        :param next_features: (bs, feature_dim)
        :param neg_delta_features: (bs, num_pred_steps, num_negative_samples, feature_dim)
        :return: energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        """
        feature_dim = self.feature_dim
        action_dim = self.action_dim

        bs = actions.shape[:-2]

        num_pred_steps = actions.shape[-2]
        num_negative_samples = neg_delta_features.shape[-2]
        if num_pred_steps > 1:
            raise NotImplementedError

        assert self.training == self.mask.training
        mask = self.mask(bs)

        flatten_bs = len(bs) > 1
        if flatten_bs:
            feature = feature.view(-1, feature_dim)
            actions = actions.view(-1, num_pred_steps, action_dim)
            next_features = next_features.view(-1, num_pred_steps, feature_dim)
            mask = mask.view(-1, feature_dim, mask.shape[-1])
            neg_delta_features = neg_delta_features.view(-1, num_pred_steps, num_negative_samples, feature_dim)

        # (bs, action_dim) or (feature_dim, bs, action_dim)
        action = actions[..., 0, :]

        # (bs, feature_dim) or (feature_dim, bs, feature_dim)
        next_feature = next_features[..., 0, :]

        # (bs, feature_dim) or (feature_dim, bs, feature_dim)
        delta_feature = (next_feature - feature).detach()
        delta_feature = delta_feature.unsqueeze(dim=-2)                             # (bs, 1, feature_dim)

        # (bs, num_negative_samples, feature_dim)
        neg_delta_features = neg_delta_features[:, 0]

        # (bs, 1 + num_negative_samples, feature_dim)
        delta_features = torch.cat([delta_feature, neg_delta_features], dim=-2)

        energy = self.forward_step(feature, action, delta_features, mask)
        energy = energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)

        return energy

    def forward(self, obs, actions, next_obses, neg_delta_features):
        feature = self.encoder(obs)
        next_features = self.encoder(next_obses)
        return self.forward_with_feature(feature, actions, next_features, neg_delta_features)

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs:
            if self.parallel_sample and self.training: {obs_i_key: (feature_dim, bs, obs_i_shape)}
            else: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param actions: 
            if self.parallel_sample and self.training: (feature_dim, bs, num_pred_steps, action_dim)
            else: (bs, num_pred_steps, action_dim)
        :param next_obses:
            if self.parallel_sample and self.training: {obs_i_key: (feature_dim, bs, obs_i_shape)}
            else: {obs_i_key: (bs, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """

        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]

        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_features = self.sample_delta_feature(bs + (num_pred_steps,), self.num_negative_samples)

        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        energy = self.forward(obs, actions, next_obses, neg_delta_features)

        loss = 0
        loss_detail = {}
        loss, loss_detail = self.update_loss_and_logging(energy, "full", loss, loss_detail)
        reward_predictor_params = self.params.reward_predictor_params
        if self.mask.num_partitions == 2:
            loss += self.mask.get_prob()[..., 0].mean() * reward_predictor_params.tia_params.relevant_reg_coef
        elif self.mask.num_partitions == 3:
            loss += self.mask.get_prob()[..., 0].mean() * reward_predictor_params.denoised_params.x_reg_coef
        self.backprop(loss, loss_detail)

        return loss_detail

    def get_adjacency(self):
        return None

    def get_intervention_mask(self):
        return None
