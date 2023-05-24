import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contrastive import Contrastive
from model.inference_utils import reset_layer, forward_network, forward_network_batch


class ContrastiveModular(Contrastive):
    def __init__(self, encoder, decoder, params):
        self.modular_params = params.contrastive_params.modular_params
        self.aggregation = self.modular_params.aggregation
        super(ContrastiveModular, self).__init__(encoder, decoder, params)

    def init_model(self):
        params = self.params
        modular_params = self.modular_params
        self.dot_product_energy = dot_product_energy = modular_params.dot_product_energy

        # model params
        continuous_state = self.continuous_state
        if not continuous_state:
            raise NotImplementedError

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()
        self.delta_state_feature_weights = nn.ParameterList()
        self.delta_state_feature_biases = nn.ParameterList()

        self.energy_weights = nn.ParameterList()
        self.energy_biases = nn.ParameterList()
        self.cond_energy_weights = nn.ParameterList()
        self.cond_energy_biases = nn.ParameterList()

        self.sa_encoder_weights = nn.ParameterList()
        self.sa_encoder_biases = nn.ParameterList()
        self.d_encoder_weights = nn.ParameterList()
        self.d_encoder_biases = nn.ParameterList()
        self.cond_sa_encoder_weights = nn.ParameterList()
        self.cond_sa_encoder_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        # action feature extractor
        in_dim = action_dim
        for out_dim in modular_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        in_dim = 1 * self.num_observation_steps
        for out_dim in modular_params.feature_fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, 1, out_dim)))
            in_dim = out_dim

        # delta state feature extractor
        in_dim = 1
        for out_dim in modular_params.feature_fc_dims:
            self.delta_state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.delta_state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        if dot_product_energy:
            # sa_feature encoder
            in_dim = modular_params.feature_fc_dims[-1]
            for out_dim in modular_params.enery_fc_dims:
                self.sa_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.sa_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

            # delta feature encoder
            in_dim = modular_params.feature_fc_dims[-1]
            for out_dim in modular_params.enery_fc_dims:
                self.d_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.d_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim
        else:
            # energy
            in_dim = 2 * modular_params.feature_fc_dims[-1]
            for out_dim in modular_params.enery_fc_dims:
                self.energy_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.energy_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim
            self.energy_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, 1)))
            self.energy_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, 1)))

    def reset_params(self):
        feature_dim = self.feature_dim
        module_weights = [self.action_feature_weights,
                          self.state_feature_weights,
                          self.delta_state_feature_weights,
                          self.energy_weights,
                          self.cond_energy_weights,
                          self.sa_encoder_weights,
                          self.d_encoder_biases,
                          self.cond_sa_encoder_weights]
        module_biases = [self.action_feature_biases,
                         self.state_feature_biases,
                         self.delta_state_feature_biases,
                         self.energy_biases,
                         self.cond_energy_biases,
                         self.sa_encoder_biases,
                         self.d_encoder_biases,
                         self.cond_sa_encoder_biases]
        for weights, biases in zip(module_weights, module_biases):
            for w, b in zip(weights, biases):
                assert w.ndim == b.ndim == 3
                for i in range(w.shape[0]):
                    reset_layer(w[i], b[i])

    def extract_action_feature(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: (feature_dim, 1, bs, out_dim)
        """
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action = action.expand(self.feature_dim, -1, -1)                    # (feature_dim, bs, action_dim)
        action_feature = forward_network(action, self.action_feature_weights, self.action_feature_biases)
        return action_feature.unsqueeze(dim=1)                              # (feature_dim, 1, bs, out_dim)

    def extract_state_feature(self, features):
        """
        :param features:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim
        if self.continuous_state:
            bs = features.shape[0]
            x = features.permute(2, 0, 1).unsqueeze(dim=0)      # (1, feature_dim, bs, num_observation_steps)
            x = x.repeat(feature_dim, 1, 1, 1)                  # (feature_dim, feature_dim, bs, num_observation_steps)
            x = x.view(feature_dim * feature_dim, bs, -1)       # (feature_dim * feature_dim, bs, 1)
        else:
            bs = features[0].shape[0]
            # [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            reshaped_feature = []
            for f_i in features:
                f_i = f_i.repeat(feature_dim, 1, 1)                         # (feature_dim, bs, feature_i_dim)
                reshaped_feature.append(f_i)
            x = forward_network_batch(reshaped_feature,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases)
            x = torch.stack(x, dim=1)                                       # (feature_dim, feature_dim, bs, out_dim)
            x = x.view(feature_dim * feature_dim, *x.shape[2:])             # (feature_dim * feature_dim, bs, out_dim)

        state_feature = forward_network(x, self.state_feature_weights, self.state_feature_biases)
        state_feature = state_feature.view(feature_dim, feature_dim, bs, -1)
        return state_feature                                                # (feature_dim, feature_dim, bs, out_dim)

    def extract_delta_state_feature(self, delta_feature):
        """
        :param delta_feature:
            if state space is continuous: (bs, num_samples, feature_dim).
            else: [(bs, num_samples, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (feature_dim, bs, num_samples, out_dim)
        """
        feature_dim = self.feature_dim
        if self.continuous_state:
            bs, num_samples, _ = delta_feature.shape
            x = delta_feature.view(-1, feature_dim).T               # (feature_dim, bs * num_samples)
            x = x.unsqueeze(dim=-1)                                 # (feature_dim, bs * num_samples, 1)
        else:
            raise NotImplementedError

        delta_state_feature = forward_network(x, self.delta_state_feature_weights, self.delta_state_feature_biases)
        delta_state_feature = delta_state_feature.view(feature_dim, bs, num_samples, -1)
        return delta_state_feature                                  # (feature_dim, bs, num_samples, out_dim)

    @staticmethod
    def dot_product(sa_encoding, delta_encoding):
        """
        compute the dot product between sa_encoding and delta_encoding
        :param sa_encoding: (feature_dim, bs, encoding_dim),
            notice that bs must be 1D
        :param delta_encoding: (feature_dim, bs, num_samples, encoding_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """

        return energy

    def compute_energy_dot(self, sa_feature, delta_feature):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        feature_dim, bs, num_samples, delta_feature_dim = delta_feature.shape
        # (feature_dim, bs * num_samples, delta_feature_dim)
        delta_feature = delta_feature.view(feature_dim, bs * num_samples, -1)

        # (feature_dim, bs * num_samples, out_dim)
        delta_encoding = forward_network(delta_feature, self.d_encoder_weights, self.d_encoder_biases)
        # (feature_dim, bs, num_samples, out_dim)
        delta_encoding = delta_encoding.view(feature_dim, bs, num_samples, -1)

        # (feature_dim, bs, out_dim)
        sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)
        sa_encoding = sa_encoding.unsqueeze(dim=-2)                 # (feature_dim, bs, 1, out_dim)

        energy = (sa_encoding * delta_encoding).sum(dim=-1)         # (feature_dim, bs, num_samples)
        energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    @staticmethod
    def unsqueeze_expand_tensor(tensor, dim, expand_size):
        tensor = tensor.unsqueeze(dim=dim)
        expand_sizes = [-1] * tensor.ndim
        expand_sizes[dim] = expand_size
        tensor = tensor.expand(*expand_sizes)
        return tensor

    def compute_energy_net(self, sa_feature, delta_feature):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """

        sa_feature_dim = sa_feature.shape[-1]
        feature_dim, bs, num_samples, delta_feature_dim = delta_feature.shape
        
        # (feature_dim, bs, num_samples, sa_feature_dim)
        sa_feature = self.unsqueeze_expand_tensor(sa_feature, -2, num_samples)

        sad_feature = torch.cat([sa_feature, delta_feature], dim=-1)
        sad_feature_dim = sad_feature.shape[-1]

        # (feature_dim, bs * num_samples, sad_feature_dim)
        sad_feature = sad_feature.view(feature_dim, -1, sad_feature_dim)

        # (feature_dim, bs * num_samples, 1)
        energy = forward_network(sad_feature, self.energy_weights, self.energy_biases)

        energy = energy.view(feature_dim, bs, num_samples)          # (feature_dim, bs, num_samples)
        energy = energy.permute(1, 2, 0)                            # (bs, num_samples, feature_dim)

        return energy

    def compute_energy(self, sa_feature, delta_feature):
        """
        compute the conditional energy from the conditional and total state-action-delta features
        :param sa_feature: (feature_dim, bs, sa_feature_dim) or (feature_dim, feature_dim, bs, sa_feature_dim),
            notice that bs must be 1D
        :param delta_feature: (feature_dim, bs, num_samples, delta_feature_dim), global feature used for prediction,
            notice that bs must be 1D
        :return: energy: (bs, num_samples, feature_dim)
        """
        if self.dot_product_energy:
            return self.compute_energy_dot(sa_feature, delta_feature)
        else:
            return self.compute_energy_net(sa_feature, delta_feature)

    def forward_step(self, features, action, delta_features):
        """
        :param features:
            if state space is continuous: (bs, num_observation_steps, feature_dim).
            else: NotImplementedError
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param delta_features:
            if observation space is continuous: (bs, num_samples, feature_dim).
            else: NotImplementedError
        :param forward_mode: which energy to compute
        :return:
        energy
            for training, (bs, 1 + num_negative_samples, feature_dim)
            for evaluation, (bs, num_pred_samples, feature_dim)
        """
        bs = action.shape[:-1]
        if len(bs) > 1:
            num_observation_steps, feature_dim = features.shape[-2:]
            num_samples = delta_features.shape[-2]
            features = features.view(-1, num_observation_steps, feature_dim)
            delta_features = delta_features.view(-1, num_samples, feature_dim)

        # extract features from the action, the states, and the next state candidates
        action_feature = self.extract_action_feature(action)        # (feature_dim, 1, bs, out_dim)
        state_feature = self.extract_state_feature(features)        # (feature_dim, feature_dim, bs, out_dim)
        # (feature_dim, bs, num_samples, out_dim)
        delta_feature = self.extract_delta_state_feature(delta_features)

        sa_feature = torch.cat([state_feature, action_feature], dim=1)  # (feature_dim, feature_dim + 1, bs, out_dim)

        if self.aggregation == "max":
            sa_feature, _ = sa_feature.max(dim=1)                   # (feature_dim, bs, out_dim)
        elif self.aggregation == "mean":
            sa_feature = sa_feature.mean(dim=1)                     # (feature_dim, bs, out_dim)
        else:
            raise NotImplementedError

        energy = self.compute_energy(sa_feature, delta_feature)
        if len(bs) > 1:
            energy = energy.view(*bs, num_samples, feature_dim)

        return energy
