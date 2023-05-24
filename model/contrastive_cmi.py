import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contrastive import Contrastive
from model.inference_utils import reset_layer, forward_network, get_state_abstraction
from utils.utils import to_numpy


class ContrastiveCMI(Contrastive):
    def __init__(self, encoder, decoder, params):
        self.cmi_params = params.contrastive_params.cmi_params
        self.init_graph(params, encoder)
        super(ContrastiveCMI, self).__init__(encoder, decoder, params)
        self.aggregation = self.cmi_params.aggregation
        self.mask_opt_freq = self.cmi_params.mask_opt_freq
        self.full_opt_freq = self.cmi_params.full_opt_freq
        self.causal_opt_freq = self.cmi_params.causal_opt_freq

        self.energy_norm_reg_coef = self.cmi_params.energy_norm_reg_coef
        self.sa_grad_reg_coef = self.cmi_params.sa_grad_reg_coef
        self.delta_grad_reg_coef = self.cmi_params.delta_grad_reg_coef

        replay_buffer_params = params.training_params.replay_buffer_params
        self.parallel_sample = replay_buffer_params.prioritized_buffer

        self.update_num = 0

    def init_model(self):
        params = self.params
        cmi_params = self.cmi_params
        self.learn_bo = learn_bo = cmi_params.learn_bo

        # model params
        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = params.feature_dim

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()
        self.delta_state_feature_weights = nn.ParameterList()
        self.delta_state_feature_biases = nn.ParameterList()

        self.sa_encoder_weights = nn.ParameterList()
        self.sa_encoder_biases = nn.ParameterList()
        self.d_encoder_weights = nn.ParameterList()
        self.d_encoder_biases = nn.ParameterList()
        self.cond_sa_encoder_weights = nn.ParameterList()
        self.cond_sa_encoder_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        # action feature extractor
        in_dim = action_dim
        for out_dim in cmi_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        in_dim = 1
        for out_dim in cmi_params.feature_fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, 1, out_dim)))
            in_dim = out_dim

        # delta state feature extractor
        in_dim = 1
        for out_dim in cmi_params.feature_fc_dims:
            self.delta_state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.delta_state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # sa_feature encoder
        in_dim = cmi_params.feature_fc_dims[-1]
        for out_dim in cmi_params.encoding_fc_dims:
            self.sa_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.sa_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # delta feature encoder
        in_dim = cmi_params.feature_fc_dims[-1]
        for out_dim in cmi_params.encoding_fc_dims:
            self.d_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.d_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # conditional sa_feature encoder
        if learn_bo:
            in_dim = 2 * cmi_params.feature_fc_dims[-1]
            for out_dim in cmi_params.encoding_fc_dims:
                self.cond_sa_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.cond_sa_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

        training_masks = []
        for i in range(feature_dim):
            training_masks.append(self.get_eval_mask((1,), i))

        # 1st feature_dim: variable to predict, 2nd feature_dim: input variable to ignore
        training_masks = torch.stack(training_masks, dim=2)         # (1, feature_dim, feature_dim, feature_dim + 1)
        self.training_masks = training_masks.view(feature_dim, feature_dim, feature_dim + 1, 1, 1)

    def reset_params(self):
        feature_dim = self.feature_dim
        module_weights = [self.action_feature_weights,
                          self.state_feature_weights,
                          self.delta_state_feature_weights,
                          self.sa_encoder_weights,
                          self.d_encoder_weights,
                          self.cond_sa_encoder_weights]
        module_biases = [self.action_feature_biases,
                         self.state_feature_biases,
                         self.delta_state_feature_biases,
                         self.sa_encoder_biases,
                         self.d_encoder_biases,
                         self.cond_sa_encoder_biases]
        for weights, biases in zip(module_weights, module_biases):
            for w, b in zip(weights, biases):
                assert w.ndim == b.ndim == 3
                for i in range(w.shape[0]):
                    reset_layer(w[i], b[i])

    def init_graph(self, params, encoder):
        feature_dim = encoder.feature_dim
        device = params.device
        self.CMI_threshold = self.cmi_params.CMI_threshold

        # used for masking diagonal elementss
        self.mask_CMI = torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        self.mask = torch.ones(feature_dim, feature_dim + 1, dtype=torch.bool, device=device)

    def extract_action_feature(self, action):
        """
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
            notice that bs must be 1D
        :return: (feature_dim, 1, bs, out_dim)
        """
        if not (self.parallel_sample and self.training):
            action = action.unsqueeze(dim=0)                                # (1, bs, action_dim)
            action = action.expand(self.feature_dim, -1, -1)                # (feature_dim, bs, action_dim)
        
        action = action.detach()
        action.requires_grad = True
        self.input_action = action

        action_feature = forward_network(action, self.action_feature_weights, self.action_feature_biases)
        return action_feature.unsqueeze(dim=1)                              # (feature_dim, 1, bs, out_dim)

    def extract_state_feature(self, feature):
        """
        :param feature:
            if self.parallel_sample and self.training: (feature_dim, bs, feature_dim)
            else: (bs, feature_dim)
            notice that bs must be 1D
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim
        if self.parallel_sample and self.training:
            bs = feature.shape[1]
            x = feature.permute(0, 2, 1).unsqueeze(dim=-1)         # (feature_dim, feature_dim, bs, 1)
        else:
            bs = feature.shape[0]
            x = feature.T[None, :, :, None]                         # (1, feature_dim, bs, 1)
            x = x.repeat(feature_dim, 1, 1, 1)                      # (feature_dim, feature_dim, bs, 1)

        x = x.detach()
        x.requires_grad = True
        self.input_feature = x

        x = x.reshape(feature_dim * feature_dim, bs, 1)             # (feature_dim * feature_dim, bs, 1)

        state_feature = forward_network(x, self.state_feature_weights, self.state_feature_biases)
        state_feature = state_feature.view(feature_dim, feature_dim, bs, -1)
        return state_feature                                        # (feature_dim, feature_dim, bs, out_dim)

    def extract_delta_state_encoding(self, delta_feature):
        """
        :param delta_feature: (bs, num_samples, feature_dim).
            notice that bs must be 1D
        :return: (feature_dim, bs, num_samples, encoding_dim)
        """
        delta_feature = delta_feature.detach()
        delta_feature.requires_grad = True
        self.input_delta_features = delta_feature

        bs, num_samples, feature_dim = delta_feature.shape
        x = delta_feature.view(-1, feature_dim).T                   # (feature_dim, bs * num_samples)
        x = x.unsqueeze(dim=-1)                                     # (feature_dim, bs * num_samples, 1)

        # (feature_dim, bs * num_samples, out_dim)
        delta_state_feature = forward_network(x, self.delta_state_feature_weights, self.delta_state_feature_biases)

        # (feature_dim, bs * num_samples, encoding_dim)
        delta_encoding = forward_network(delta_state_feature, self.d_encoder_weights, self.d_encoder_biases)
        # (feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = delta_encoding.view(feature_dim, bs, num_samples, -1)
        return delta_encoding                                       # (feature_dim, bs, num_samples, encoding_dim)

    @staticmethod
    def dot_product(sa_encoding, delta_encoding):
        """
        compute the dot product between sa_encoding and delta_encoding
        :param sa_encoding: (feature_dim, bs, encoding_dim) or (feature_dim, feature_dim, bs, encoding_dim),
            notice that bs must be 1D
        :param delta_encoding: (feature_dim, bs, num_samples, encoding_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        # (feature_dim, bs, 1, out_dim) or (feature_dim, feature_dim, bs, 1, out_dim)
        sa_encoding = sa_encoding.unsqueeze(dim=-2)

        if sa_encoding.ndim == 5:
            num_samples = delta_encoding.shape[-2]
            if num_samples < 5000:
                delta_encoding = delta_encoding.unsqueeze(dim=1)        # (feature_dim, 1, bs, num_samples, out_dim)
                energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (feature_dim, feature_dim, bs, num_samples)
            else:
                # likely to have out of memory issue, so need to compute energy in batch
                energy = []
                for sa_encoding in torch.unbind(sa_encoding, dim=1):
                    energy.append((sa_encoding * delta_encoding).sum(dim=-1))
                energy = torch.stack(energy, dim=1)
            energy = energy.permute(2, 3, 0, 1)                         # (bs, num_samples, feature_dim, feature_dim)
        else:
            energy = (sa_encoding * delta_encoding).sum(dim=-1)         # (feature_dim, bs, num_samples)
            energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    def compute_energy(self, sa_feature, delta_encoding, full_sa_feature=None):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim) or (feature_dim, feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_encoding: (feature_dim, bs, num_samples, encoding_dim), global feature used for prediction,
            notice that bs must be 1D
        :param full_sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        assert sa_feature.ndim in [3, 4]
        is_mask_feature = sa_feature.ndim == 4

        feature_dim = self.feature_dim
        bs = sa_feature.shape[-2]

        if is_mask_feature:
            # (feature_dim, feature_dim * bs, sa_feature_dim)
            sa_feature = sa_feature.view(feature_dim, feature_dim * bs, -1)

        # (feature_dim, bs, out_dim) or (feature_dim, feature_dim * bs, out_dim)
        sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            sa_encoding = sa_encoding.view(feature_dim, feature_dim, bs, -1)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        energy = self.dot_product(sa_encoding, delta_encoding)

        if full_sa_feature is None:
            return energy

        if not self.learn_bo:
            return energy, torch.zeros_like(energy)     # placeholder

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            full_sa_feature = full_sa_feature.unsqueeze(dim=1).expand(-1, feature_dim, -1, -1)
            # (feature_dim, feature_dim * bs, out_dim)
            full_sa_feature = full_sa_feature.reshape(feature_dim, feature_dim * bs, -1)
            # (feature_dim, feature_dim * bs, 2 * out_dim)
            cond_sa_feature = torch.cat([sa_feature, full_sa_feature], dim=-1)
        else:
            # (feature_dim, bs, 2 * out_dim)
            cond_sa_feature = torch.cat([sa_feature, full_sa_feature], dim=-1)

        # (feature_dim, bs, out_dim) or (feature_dim, feature_dim * bs, out_dim)
        cond_sa_encoding = forward_network(cond_sa_feature, self.cond_sa_encoder_weights, self.sa_encoder_biases)

        if is_mask_feature:
            # (feature_dim, feature_dim, bs, out_dim)
            cond_sa_encoding = cond_sa_encoding.view(feature_dim, feature_dim, bs, -1)

        # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
        cond_energy = self.dot_product(cond_sa_encoding, delta_encoding)

        return energy, cond_energy

    def forward_step(self, feature, action, delta_features, forward_mode=("full", "mask", "causal")):
        """
        compute energy for the following combinations
        if using (1) next_feature + neg_delta_features for training
            a. feature from randomly masking one state variable + conditional feature from all variables
            b. feature from all variables
            c. feature from causal parents + conditional feature from all variable (? probably for eval only)
        elif using (2) pred_delta_features for evaluation
            a. feature from causal parents + conditional feature from all variable (? probably for eval only)
        :param feature:
            if self.parallel_sample and self.training: (feature_dim, bs, feature_dim)
            else: (bs, feature_dim)
            notice that bs must be 1D
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
        :param delta_features: (bs, num_samples, feature_dim)
        :param forward_mode: which energy to compute
        :return: energy (bs, num_samples, feature_dim)
        """
        bs, _, feature_dim = delta_features.shape

        # extract features from the action, the states, and the next state candidates
        action_feature = self.extract_action_feature(action)            # (feature_dim, 1, bs, out_dim)
        state_feature = self.extract_state_feature(feature)             # (feature_dim, feature_dim, bs, out_dim)
        # (feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = self.extract_delta_state_encoding(delta_features)

        sa_feature = torch.cat([state_feature, action_feature], dim=1)  # (feature_dim, feature_dim + 1, bs, out_dim)

        if self.aggregation == "max":
            full_sa_feature, _ = sa_feature.max(dim=1)                  # (feature_dim, bs, out_dim)
        elif self.aggregation == "mean":
            full_sa_feature = sa_feature.mean(dim=1)                    # (feature_dim, bs, out_dim)
        else:
            raise NotImplementedError

        # (bs, num_samples, feature_dim)
        full_energy = mask_energy = mask_cond_energy = causal_energy = causal_cond_energy = None

        if "full" in forward_mode:
            full_energy = self.compute_energy(full_sa_feature, delta_encoding)   # (bs, num_samples, feature_dim)

        if "mask" in forward_mode:
            mask_sa_feature = sa_feature.clone()                    # (feature_dim, feature_dim + 1, bs, out_dim)
            if self.training:
                mask = self.get_training_mask(bs)                   # (bs, feature_dim, feature_dim + 1)
                mask = torch.permute(mask, (1, 2, 0))               # (feature_dim, feature_dim + 1, bs)
                mask = mask.unsqueeze(dim=-1)                       # (feature_dim, feature_dim + 1, bs, 1)
            else:
                mask = self.training_masks                          # (feature_dim, feature_dim, feature_dim + 1, 1, 1)
                mask_sa_feature = mask_sa_feature.unsqueeze(dim=1)  # (feature_dim, 1, feature_dim + 1, bs, out_dim)

            # (feature_dim, feature_dim, feature_dim + 1, bs, out_dim) or (feature_dim, feature_dim + 1, bs, out_dim)
            mask_sa_feature = mask_sa_feature * mask                

            # (feature_dim, feature_dim, bs, out_dim) or (feature_dim, bs, out_dim)
            if self.aggregation == "max":
                mask_sa_feature, _ = mask_sa_feature.max(dim=-3)            
            elif self.aggregation == "mean":
                mask_sa_feature = mask_sa_feature.sum(dim=-3) / feature_dim
            else:
                raise NotImplementedError

            # (bs, num_samples, feature_dim) or (bs, num_samples, feature_dim, feature_dim)
            mask_energy, mask_cond_energy = self.compute_energy(mask_sa_feature, delta_encoding, full_sa_feature)

        if "causal" in forward_mode:
            causal_sa_feature = sa_feature.clone()                      # (feature_dim, feature_dim + 1, bs, out_dim)
            causal_mask = self.mask.detach().view(feature_dim, feature_dim + 1, 1, 1)
            causal_sa_feature = causal_sa_feature * causal_mask         # (feature_dim, feature_dim + 1, bs, out_dim)

            if self.aggregation == "max":
                causal_sa_feature, _ = causal_sa_feature.max(dim=1)             # (feature_dim, bs, out_dim)
            elif self.aggregation == "mean":
                num_parents = causal_mask.sum(dim=1)
                causal_sa_feature = causal_sa_feature.sum(dim=1) / num_parents  # (feature_dim, bs, out_dim)
            else:
                raise NotImplementedError

            # (bs, num_samples, feature_dim)
            causal_energy, causal_cond_energy = self.compute_energy(causal_sa_feature, delta_encoding, full_sa_feature)

        return full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy

    def forward_with_feature(self, feature, actions, next_features, neg_delta_features,
                             forward_mode=("full", "mask", "causal")):
        """
        :param feature:
            if self.parallel_sample and self.training: (feature_dim, bs, feature_dim)
            else: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param actions: 
            if self.parallel_sample and self.training: (feature_dim, bs, num_pred_steps, action_dim)
            else: (bs, num_pred_steps, action_dim)
        :param next_features:
            if self.parallel_sample and self.training: (feature_dim, bs, num_pred_steps, feature_dim)
            else: (bs, feature_dim)
        :param neg_delta_features: (bs, num_pred_steps, num_negative_samples, feature_dim)
        :param forward_mode: which energy to compute
        :return: energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        """
        feature_dim = self.feature_dim
        action_dim = self.action_dim

        bs = actions.shape[:-2]
        if self.parallel_sample and self.training:
            bs = bs[1:]

        num_pred_steps = actions.shape[-2]
        num_negative_samples = neg_delta_features.shape[-2]
        if num_pred_steps > 1:
            raise NotImplementedError

        flatten_bs = len(bs) > 1
        if flatten_bs:
            assert not (self.parallel_sample and self.training)
            feature = feature.view(-1, feature_dim)
            actions = actions.view(-1, num_pred_steps, action_dim)
            next_features = next_features.view(-1, num_pred_steps, feature_dim)
            neg_delta_features = neg_delta_features.view(-1, num_pred_steps, num_negative_samples, feature_dim)

        # (bs, action_dim) or (feature_dim, bs, action_dim)
        action = actions[..., 0, :]

        # (bs, feature_dim) or (feature_dim, bs, feature_dim)
        next_feature = next_features[..., 0, :]

        # (bs, feature_dim) or (feature_dim, bs, feature_dim)
        delta_feature = (next_feature - feature).detach()

        if self.parallel_sample and self.training:
            eye = torch.eye(feature_dim, device=self.device).unsqueeze(dim=1)       # (feature_dim, 1, feature_dim)
            delta_feature = (delta_feature * eye).sum(dim=-1).T                     # (bs, feature_dim)

        delta_feature = delta_feature.unsqueeze(dim=-2)                             # (bs, 1, feature_dim)
        # (bs, num_negative_samples, feature_dim)
        neg_delta_features = neg_delta_features[:, 0]
        # (bs, 1 + num_negative_samples, feature_dim)
        delta_features = torch.cat([delta_feature, neg_delta_features], dim=-2)

        full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
            self.forward_step(feature, action, delta_features, forward_mode)

        # (bs, 1, 1 + num_negative_samples, feature_dim)
        if "full" in forward_mode:
            full_energy = full_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)

        if "mask" in forward_mode:
            if mask_energy.ndim == 4:
                mask_energy = mask_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
                mask_cond_energy = mask_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
            elif mask_energy.ndim == 3:
                mask_energy = mask_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
                mask_cond_energy = mask_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
            else:
                raise NotImplementedError

        if "causal" in forward_mode:
            causal_energy = causal_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
            causal_cond_energy = causal_cond_energy.view(*bs, num_pred_steps, 1 + num_negative_samples, feature_dim)

        return full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy

    def forward(self, obs, actions, next_obses, neg_delta_features, forward_mode=("full", "mask", "causal")):
        feature = self.encoder(obs)
        next_features = self.encoder(next_obses)
        return self.forward_with_feature(feature, actions, next_features, neg_delta_features, forward_mode)

    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs feature_dim), idxes of state variable to drop
            notice that bs can be a multi-dimensional batch size
        :return: (bs, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.feature_dim + 1)
        bool_mask = int_mask < 1
        return bool_mask

    def get_training_mask(self, bs):
        # uniformly select one state variable to omit when predicting the next time step value
        if isinstance(bs, int):
            bs = (bs,)

        idxes = torch.randint(self.feature_dim, bs + (self.feature_dim,), device=self.device)
        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    def get_eval_mask(self, bs, i):
        # omit i-th state variable or the action when predicting the next time step value

        if isinstance(bs, int):
            bs = (bs,)

        feature_dim = self.feature_dim
        idxes = torch.full(size=bs + (feature_dim,), fill_value=i, dtype=torch.int64, device=self.device)
        self_mask = torch.arange(feature_dim, device=self.device)
        # each state variable must depend on itself when predicting the next time step value
        idxes[idxes >= self_mask] += 1

        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    def bo_loss(self, energy, cond_energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        :param cond_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        return self.nce_loss(energy.detach() + cond_energy)

    def energy_norm_loss(self, energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        energy_sq = (energy ** 2).sum(dim=(-3, -1)).mean()
        energy_abs = energy.abs().sum(dim=(-3, -1)).mean()

        return energy_sq * self.energy_norm_reg_coef, energy_abs

    def grad_loss(self, energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """

        feature = self.input_feature                        # (feature_dim, feature_dim, bs, 1)
        action = self.input_action                          # (feature_dim, bs, action_dim)
        delta_features = self.input_delta_features          # (bs, num_samples, feature_dim)

        grads_penalty = sa_grad_abs = 0

        nce = F.log_softmax(energy, dim=-2)[..., 0, :]      # (bs, num_pred_steps, feature_dim)
        feature_grad, action_grad = torch.autograd.grad(nce.sum(), [feature, action], create_graph=True)

        feature_grad_norm = feature_grad.abs().mean(dim=2).sum()
        grads_penalty += feature_grad.pow(2).mean(dim=2).sum() * self.sa_grad_reg_coef
        sa_grad_abs += feature_grad_norm

        action_grad_norm = action_grad.abs().mean(dim=1).sum()
        grads_penalty += action_grad.pow(2).mean(dim=1).sum() * self.sa_grad_reg_coef
        sa_grad_abs += action_grad_norm

        delta_grad = torch.autograd.grad(energy.sum(), delta_features, create_graph=True)[0]
        grads_penalty += delta_grad.pow(2).mean(dim=(0, 1)).sum() * self.delta_grad_reg_coef
        delta_grad_abs = delta_grad.abs().mean(dim=0).sum()

        return grads_penalty, sa_grad_abs, delta_grad_abs

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
        if eval:
            return self.update_mask(obs, actions, next_obses)

        self.update_num += 1

        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]
        if self.parallel_sample:
            bs = bs[1:]

        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_features = self.sample_delta_feature(bs + (num_pred_steps,), self.num_negative_samples)

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

        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
            self.forward(obs, actions, next_obses, neg_delta_features, forward_mode)

        loss = 0
        loss_detail = {}

        if "mask" in forward_mode:
            # mask_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
            loss, loss_detail = self.update_loss_and_logging(mask_energy, "mask", loss, loss_detail)

            if self.learn_bo:
                mask_bo_loss = self.bo_loss(mask_energy, mask_cond_energy)
                loss += mask_bo_loss
                loss_detail["mask_bo_gain"] = mask_nce_loss - mask_bo_loss

        if "full" in forward_mode:
            loss, loss_detail = self.update_loss_and_logging(full_energy, "full", loss, loss_detail)

        if "causal" in forward_mode:
            if opt_causal:
                loss, loss_detail = self.update_loss_and_logging(causal_energy, "causal", loss, loss_detail)

            if self.use_prioritized_buffer:
                priority = 1 - F.softmax(causal_energy, dim=-2)[..., 0, :].mean(dim=-2)         # (bs, feature_dim)
                if self.parallel_sample:
                    priority = priority.T
                else:
                    priority = priority.mean(dim=-1)
                loss_detail["priority"] = priority

            if self.learn_bo:
                # for eval only
                causal_bo_loss = self.bo_loss(causal_energy, causal_cond_energy)
                loss_detail["causal_bo_gain"] = causal_nce_loss - causal_bo_loss

        self.backprop(loss, loss_detail)

        return loss_detail

    @staticmethod
    def compute_cmi(energy, cond_energy, unbiased=True):
        """
        https://arxiv.org/pdf/2106.13401, proposition 3
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param cond_energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return: cmi: (feature_dim,)
        """
        pos_cond_energy = cond_energy[..., 0, :, :]         # (bs, num_pred_steps, feature_dim, feature_dim)

        if unbiased:
            K = energy.shape[-3]                            # num_negative_samples
            neg_energy = energy[..., 1:, :, :]              # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
            neg_cond_energy = cond_energy[..., 1:, :, :]    # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        else:
            K = energy.shape[-3] + 1                        # num_negative_samples
            neg_energy = energy                             # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
            neg_cond_energy = cond_energy                   # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)

        log_w_neg = F.log_softmax(neg_energy, dim=-3)       # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        # (bs, num_pred_steps, num_negative_samples, feature_dim, feature_dim)
        weighted_neg_cond_energy = np.log(K - 1) + log_w_neg + neg_cond_energy
        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        cond_energy = torch.cat([pos_cond_energy.unsqueeze(dim=-3), weighted_neg_cond_energy], dim=-3)
        log_denominator = -np.log(K) + torch.logsumexp(cond_energy, dim=-3)         # (bs, num_pred_steps, feature_dim, feature_dim)
        cmi = pos_cond_energy - log_denominator                                     # (bs, num_pred_steps, feature_dim, feature_dim)

        feature_dim = cmi.shape[-1]
        cmi = cmi.sum(dim=-3).view(-1, feature_dim, feature_dim).mean(dim=0)
        return cmi

    def update_mask(self, obs, actions, next_obses):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_obses: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]
        feature_dim = self.feature_dim

        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_features = self.sample_delta_feature(bs + (num_pred_steps,), self.cmi_params.eval_num_negative_samples)

        eval_details = {}

        cmi = []
        with torch.no_grad():
            feature = self.encoder(obs)
            next_features = self.encoder(next_obses)

            full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
                self.forward_with_feature(feature, actions, next_features, neg_delta_features)

            mask_nce_loss = self.nce_loss(mask_energy)
            full_nce_loss = self.nce_loss(full_energy)
            causal_nce_loss = self.nce_loss(causal_energy)
            eval_details = {"mask_nce_loss": mask_nce_loss,
                            "full_nce_loss": full_nce_loss,
                            "causal_nce_loss": causal_nce_loss}

            if self.learn_bo:
                mask_bo_loss = self.bo_loss(mask_energy, mask_cond_energy)
                causal_bo_loss = self.bo_loss(causal_energy, causal_cond_energy)
                eval_details["mask_bo_gain"] = mask_nce_loss - mask_bo_loss
                eval_details["causal_bo_gain"] = causal_nce_loss - causal_bo_loss
            else:
                mask_cond_energy = full_energy.unsqueeze(dim=-1) - mask_energy

            cmi = self.compute_cmi(mask_energy, mask_cond_energy)                       # (feature_dim, feature_dim)

        diag = torch.eye(feature_dim, feature_dim + 1, dtype=torch.float32, device=self.device)
        diag *= self.CMI_threshold

        # (feature_dim, feature_dim), (feature_dim, feature_dim)
        upper_tri, lower_tri = torch.triu(cmi), torch.tril(cmi, diagonal=-1)
        diag[:, 1:] += upper_tri
        diag[:, :-1] += lower_tri

        eval_tau = self.cmi_params.eval_tau
        self.mask_CMI = self.mask_CMI * eval_tau + diag * (1 - eval_tau)
        self.mask = self.mask_CMI >= self.CMI_threshold

        return eval_details

    def get_state_abstraction(self):
        abstraction_mask = np.zeros(self.feature_dim, dtype=bool)
        abstraction_graph = get_state_abstraction(to_numpy(self.get_mask()))
        abstraction_idxes = list(abstraction_graph.keys())
        abstraction_mask[abstraction_idxes] = True
        return abstraction_mask

    def get_mask(self):
        return self.mask

    def get_adjacency(self):
        return self.mask_CMI[:, :-1]

    def get_intervention_mask(self):
        return self.mask_CMI[:, -1:]

    def get_threshold(self):
        return self.CMI_threshold

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("contrastive loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.mask = self.mask_CMI >= self.CMI_threshold
