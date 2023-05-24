import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from model.modules import get_backbone
from model.disentangle_metrics import compute_sap, compute_dci, compute_mig, compute_modularity
from utils.utils import to_numpy


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        self.goal_keys = [key for key in params.goal_keys if params.obs_spec[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.goal_dim = np.sum([len(params.obs_spec[key]) for key in self.goal_keys]).astype(int)

        self.continuous_state = params.continuous_state
        self.object_level_obs = params.training_params.object_level_obs
        self.feature_inner_dim = None
        if not self.continuous_state and not self.object_level_obs:
            self.feature_inner_dim = np.concatenate([params.obs_dims[key] for key in self.keys])

        self.to(params.device)

    def forward(self, obs, detach=False, include_goal=False):
        if self.object_level_obs:
            return [obs[k] for k in self.keys]

        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            keys = self.keys
            if include_goal:
                keys = keys + self.goal_keys
            obs = torch.cat([obs[k] for k in keys], dim=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                obs = [obs_i if obs_i.shape[-1] > 1 else torch.randn_like(obs_i) * test_scale for obs_i in obs]
            return obs

    def evaluate_disentanglement(self, replay_buffer):
        return {}


class ConvEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.encoder_params = encoder_params = params.encoder_params
        self.feature_dim = encoder_params.feature_dim

        self.factor_keys = params.factor_keys
        self.image_keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 3]
        self.backbone, in_dim = get_backbone(params, encoding=True)
        self.final_layer = nn.Linear(in_dim, self.feature_dim * 2)

        self.log_std_min = encoder_params.log_std_min
        self.log_std_max = encoder_params.log_std_max

        self.to(params.device)

    def forward(self, obs, detach=False):
        stacked_image = torch.cat([obs[key] for key in self.image_keys], dim=-3)

        reshaped = False
        if stacked_image.ndim > 4:
            reshaped = True
            bs_shape, image_shape = stacked_image.shape[:-3], stacked_image.shape[-3:]
            stacked_image = stacked_image.view(-1, *image_shape)

        feature = self.backbone(stacked_image)
        feature = self.final_layer(feature)

        if reshaped:
            feature = feature.view(*bs_shape, self.feature_dim * 2)

        if detach:
            feature = feature.detach()

        mu, log_std = torch.split(feature, self.feature_dim, dim=-1)
        log_std = torch.clip(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        return dist

    def evaluate_disentanglement(self, replay_buffer):
        default_batch_size = 128
        eval_num_train = self.params.disentangle_params.eval_num_train
        eval_num_test = self.params.disentangle_params.eval_num_test

        def get_factor_and_encoding(train):
            num_samples = eval_num_train if train else eval_num_test
            use_part = "train" if train else "eval"

            factors = []
            encoding_dists = []
            with torch.no_grad():
                for i in range(0, num_samples, default_batch_size):
                    batch_size = min(num_samples - i, default_batch_size)
                    obs_batch = replay_buffer.sample_encoder(batch_size, use_part=use_part)

                    factor = torch.cat([obs_batch[key] for key in self.factor_keys], dim=-1)
                    encoding_dist = self(obs_batch)

                    factors.append(factor)
                    encoding_dists.append(encoding_dist)

            factors = torch.cat(factors)
            mu = torch.cat([dist.mean for dist in encoding_dists], dim=-2)        # (num_samples, feature_dim)
            std = torch.cat([dist.stddev for dist in encoding_dists], dim=-2)     # (num_samples, feature_dim)
            return factors, Normal(mu, std)

        factor_train, encoding_dist_train = get_factor_and_encoding(train=True)
        factor_test, encoding_dist_test = get_factor_and_encoding(train=False)
        mu_train, mu_test = encoding_dist_train.mean, encoding_dist_test.mean

        factor_train, factor_test = to_numpy(factor_train), to_numpy(factor_test)
        mu_train, mu_test = to_numpy(mu_train), to_numpy(mu_test)

        params = self.params
        disentangle_score = {"sap": compute_sap(params, factor_train, mu_train, factor_test, mu_test),
                             "dci": compute_dci(params, factor_train, mu_train, factor_test, mu_test)}

        if not params.continuous_factor and (params.fac_inner_dim > 1).all():
            disentangle_score["mig"] = compute_mig(params, factor_train, mu_train, factor_test, mu_test)
            disentangle_score["mod"] = compute_modularity(params, factor_train, mu_train, factor_test, mu_test)

        return disentangle_score


_AVAILABLE_ENCODERS = {"identity": IdentityEncoder,
                       "conv": ConvEncoder}


def make_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    return _AVAILABLE_ENCODERS[encoder_type](params)
