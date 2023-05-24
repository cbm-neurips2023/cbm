import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.modules import Attention, MHAttention
from model.inference import Inference
from model.inference_utils import reset_layer, forward_network, forward_network_batch


class InferenceAttn(Inference):
    def __init__(self, encoder, decoder, params):
        self.attn_params = params.inference_params.attn_params
        super(InferenceAttn, self).__init__(encoder, decoder, params)

    def init_model(self):
        params = self.params
        attn_params = self.attn_params

        # model params
        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = self.encoder.feature_inner_dim
        num_observation_steps = self.num_observation_steps

        self.obs_spec = obs_spec = params.obs_spec
        self.obs_keys = obs_keys = params.obs_keys
        self.num_objs = num_objs = len(obs_spec)

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()
        self.generative_weights = nn.ParameterList()
        self.generative_biases = nn.ParameterList()

        # attention
        num_queries = num_objs if self.object_level_obs else feature_dim
        num_keys = num_queries + 1

        feature_embed_dim = attn_params.feature_fc_dims[-1]
        attn_dim = attn_params.attn_dim
        num_heads = attn_params.num_heads
        attn_out_dim = attn_params.attn_out_dim
        use_bias = attn_params.attn_use_bias
        self.gumbel_select = attn_params.gumbel_select
        if num_heads == 1:
            self.attn = Attention(attn_dim, num_queries, feature_embed_dim, num_keys, feature_embed_dim,
                                  out_dim=attn_out_dim, use_bias=use_bias)
        else:
            self.attn = MHAttention(attn_dim, num_heads, num_queries, feature_embed_dim, num_keys, feature_embed_dim,
                                    out_dim=attn_out_dim, use_bias=use_bias)

        # only needed for discrete observation space or when using object_level_obs
        self.state_feature_1st_layer_weights = nn.ParameterList()
        self.state_feature_1st_layer_biases = nn.ParameterList()
        self.generative_last_layer_weights = nn.ParameterList()
        self.generative_last_layer_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        # action feature extractor
        in_dim = action_dim
        for out_dim in attn_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if self.object_level_obs:
            out_dim = attn_params.feature_fc_dims[0]
            for key in self.obs_keys:
                in_dim = len(self.obs_spec[key]) * num_observation_steps
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))

            in_dim = out_dim
            for out_dim in attn_params.feature_fc_dims[1:]:
                self.state_feature_weights.append(nn.Parameter(torch.zeros(num_objs, in_dim, out_dim)))
                self.state_feature_biases.append(nn.Parameter(torch.zeros(num_objs, 1, out_dim)))
                in_dim = out_dim
        else:
            if continuous_state:
                in_dim = num_observation_steps
                fc_dims = attn_params.feature_fc_dims
            else:
                out_dim = attn_params.feature_fc_dims[0]
                fc_dims = attn_params.feature_fc_dims[1:]
                for feature_i_dim in self.feature_inner_dim:
                    in_dim = feature_i_dim
                    self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
                    self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))
                in_dim = out_dim

            for out_dim in fc_dims:
                self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

        # generator mlps
        if self.object_level_obs:
            in_dim = attn_out_dim
            for out_dim in attn_params.generative_fc_dims:
                self.generative_weights.append(nn.Parameter(torch.zeros(num_objs, in_dim, out_dim)))
                self.generative_biases.append(nn.Parameter(torch.zeros(num_objs, 1, out_dim)))
                in_dim = out_dim

            for key in self.obs_keys:
                final_dim = 2 * len(self.obs_spec[key])
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))
        else:
            in_dim = attn_out_dim
            for out_dim in attn_params.generative_fc_dims:
                self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

            if continuous_state:
                self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, 2)))
                self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, 2)))
            else:
                for feature_i_dim in self.feature_inner_dim:
                    final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                    self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                    self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        feature_dim = self.feature_dim
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            reset_layer(w, b)
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            reset_layer(w, b)
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(len(w)):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(len(w)):
                reset_layer(w[i], b[i])

    def forward_step(self, features, action):
        """
        :param features:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if observation space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        bs = action.shape[:-1]
        feature_dim = self.feature_dim

        bs_dims = np.arange(len(bs))

        # extract action feature
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action_feature = forward_network(action,
                                         self.action_feature_weights,
                                         self.action_feature_biases)        # (1, bs, feature_embed_dim)

        # extract state feature
        if self.continuous_state:
            x = features.permute(-1, *bs_dims, -2)                          # (feature_dim, bs, num_observation_steps)
        else:
            # [(1, bs, num_observation_steps * feature_i_dim)] * feature_dim
            x = [feature_i.view(*bs, -1).unsqueeze(dim=0) for feature_i in features]
            x = forward_network_batch(x,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases)  # [(1, bs, layer_out_dim)] * feature_dim
            x = torch.cat(x, dim=0)                                         # (feature_dim, bs, layer_out_dim)

        state_feature = forward_network(x,
                                        self.state_feature_weights,
                                        self.state_feature_biases)          # (feature_dim, bs, feature_embed_dim)
        sa_feature = torch.cat([action_feature, state_feature], dim=0)      # (feature_dim + 1, bs, feature_embed_dim)

        # (feature_dim, bs, attn_out_dim)
        sa_feature = self.attn(state_feature, sa_feature, gumbel_select=self.gumbel_select)

        x = forward_network(sa_feature, self.generative_weights, self.generative_biases)

        def normal_helper(mean_, base_, log_std_):
            if self.residual:
                mean_ = mean_ + base_
            log_std_ = torch.clip(log_std_, min=self.log_std_min, max=self.log_std_max)
            std_ = torch.exp(log_std_)
            return Normal(mean_, std_)

        if self.continuous_state:
            x = x.permute(*(1 + bs_dims), 0, -1)                                    # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)                                          # (bs, feature_dim) * 2
            return normal_helper(mu, features[..., -1, :], log_std)
        else:
            x = F.relu(x)                                                           # (feature_dim, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]            # [(1, bs, out_dim)] * feature_dim
            x = forward_network_batch(x,
                                      self.generative_last_layer_weights,
                                      self.generative_last_layer_biases,
                                      activation=None)

            dist = []
            if self.object_level_obs:
                for base_i, dist_i in zip(features, x):
                    dist_i = dist_i.squeeze(dim=0)
                    mu, log_std = torch.split(dist_i, base_i.shape[-1], dim=-1)     # (obj_i_dim, 1), (obj_i_dim, 1)
                    dist.append(normal_helper(mu, base_i[..., -1, :], log_std))
            else:
                for base_i, feature_i_inner_dim, dist_i in zip(residual_base, self.feature_inner_dim, x):
                    dist_i = dist_i.squeeze(dim=0)
                    if feature_i_inner_dim == 1:
                        mu, log_std = torch.split(dist_i, 1, dim=-1)                # (bs, 1), (bs, 1)
                        dist.append(normal_helper(mu, base_i, log_std))
                    else:
                        dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_step_abstraction(self, abstraction_feature, action):
        """
        :param abstraction_feature: (bs, abstraction_feature_dim) or [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all abstraction state variables in the format of distribution,
            if observation space is continuous, distribution is a tuple for (sample + mean + log_std),
            otherwise, distribution is a tuple for (sample + logits),
        """
        return self.forward_step(abstraction_feature, action)

    def get_state_abstraction(self):
        abstraction_graph = {i: np.arange(self.feature_dim + 1) for i in range(self.feature_dim)}
        return abstraction_graph

    def get_adjacency(self):
        return torch.ones(self.feature_dim, self.feature_dim)

    def get_intervention_mask(self):
        return torch.ones(self.feature_dim, 1)
