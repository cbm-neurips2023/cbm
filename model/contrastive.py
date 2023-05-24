import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class Contrastive(nn.Module):
    def __init__(self, encoder, decoder, params):
        super(Contrastive, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.params = params
        self.device = device = params.device

        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action
        assert self.continuous_state, "contrastive learning doesn't support discrete state space"

        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer

        self.contrastive_params = contrastive_params = params.contrastive_params
        self.num_pred_steps = contrastive_params.num_pred_steps
        self.gradient_through_pred_steps = contrastive_params.gradient_through_pred_steps

        self.num_negative_samples = contrastive_params.num_negative_samples
        # (feature_dim,)
        self.delta_feature_min = self.encoder({key: val[0] for key, val in self.params.obs_delta_range.items()})
        self.delta_feature_max = self.encoder({key: val[1] for key, val in self.params.obs_delta_range.items()})

        self.num_pred_samples = contrastive_params.num_pred_samples
        self.num_pred_iters = contrastive_params.num_pred_iters
        self.pred_sigma_init = contrastive_params.pred_sigma_init
        self.pred_sigma_shrink = contrastive_params.pred_sigma_shrink

        self.init_model()
        self.reset_params()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=contrastive_params.lr)

        self.load(params.training_params.load_inference, device)
        self.train()

    def init_model(self):
        raise NotImplementedError

    def reset_params(self):
        pass

    def setup_annealing(self, step):
        pass

    def sample_delta_feature(self, shape, num_samples):
        # (bs, num_pred_samples, feature_dim)
        uniform_noise = torch.rand(*shape, num_samples, self.feature_dim, dtype=torch.float32, device=self.device)
        delta_feature = uniform_noise * (self.delta_feature_max - self.delta_feature_min) + self.delta_feature_min
        return delta_feature

    def forward_step(self, feature, action, delta_features):
        """
        compute energy
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :param delta_features: (bs, num_samples, feature_dim)
        :return: energy: (bs, num_samples, feature_dim)
        """
        raise NotImplementedError

    def forward_with_feature(self, feature, actions, next_features, neg_delta_features=None):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_features: (bs, num_pred_steps, feature_dim)
        :param neg_delta_features: (bs, num_pred_steps, num_negative_samples, feature_dim)
        :return:
            energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        """
        energies = []
        actions = torch.unbind(actions, dim=-2)
        next_features = torch.unbind(next_features, dim=-2)
        neg_delta_features = torch.unbind(neg_delta_features, dim=-3)
        for i, (action, next_feature, neg_delta_features_i) in enumerate(zip(actions, next_features, neg_delta_features)):
            delta_feature = next_feature - feature                                              # (bs, feature_dim)
            delta_feature = delta_feature.unsqueeze(dim=-2)                                     # (bs, 1, feature_dim)
            # (bs, 1 + num_negative_samples, feature_dim)
            delta_features = torch.cat([delta_feature, neg_delta_features_i], dim=-2)
            energy = self.forward_step(feature, action, delta_features)
            energies.append(energy)

            if i == len(actions) - 1:
                break

            # (bs, num_negative_samples, feature_dim)
            if self.gradient_through_pred_steps:
                # (bs, 1 + num_negative_samples, feature_dim)
                delta_feature_select = F.gumbel_softmax(energy, dim=-2, hard=True)
                delta_feature = (delta_features * delta_feature_select).sum(dim=-2)             # (bs, feature_dim)
                feature += delta_feature
            else:
                feature = next_feature

        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        energies = torch.stack(energies, dim=-3)
        return energies

    def forward(self, obs, actions, next_obses, neg_delta_feature=None):
        feature = self.encoder(obs)
        next_features = self.encoder(next_obses)
        return self.forward_with_feature(feature, actions, next_features, neg_delta_feature)

    @staticmethod
    def nce_loss(energy):
        """
        :param energy: (bs, num_pred_steps, 1 + num_negative_samples, feature_dim) or 
                       (bs, num_pred_steps, 1 + num_negative_samples, feature_dim, feature_dim)
        :return:
            loss: scalar
        """
        if energy.ndim == 4:
            return -F.log_softmax(energy, dim=-2)[..., 0, :].sum(dim=(-2, -1)).mean()
        elif energy.ndim == 5:
            return -F.log_softmax(energy, dim=-3)[..., 0, :, :].sum(dim=(-3, -2, -1)).mean()
        else:
            raise NotImplementedError

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.contrastive_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        self.optimizer.step()
        return loss_detail

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_obs: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]
        # (bs, num_pred_steps, num_negative_samples, feature_dim)
        neg_delta_feature = self.sample_delta_feature(bs + (num_pred_steps,), self.num_negative_samples)
        # (bs, num_pred_steps, 1 + num_negative_samples, feature_dim)
        energy = self.forward(obs, actions, next_obses, neg_delta_feature)

        loss_detail = {}
        loss = self.nce_loss(energy)
        loss_detail["nce_loss"] = loss

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def predict_step_with_feature(self, feature, action):
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
        delta_feature_max = self.delta_feature_max
        delta_feature_min = self.delta_feature_min
        sigma = self.pred_sigma_init

        delta_feature_candidates = self.sample_delta_feature(bs, num_pred_samples)

        for i in range(self.num_pred_iters):
            # (bs, num_pred_samples, feature_dim)
            if self.params.training_params.inference_algo == "contrastive_cmi":
                forward_mode = ("causal",)
                # forward_mode = ("full",)

                if len(bs) > 1:
                    feature = feature.reshape(-1, feature_dim)
                    action = action.view(-1, action_dim)
                    delta_feature_candidates = delta_feature_candidates.view(-1, num_pred_samples, feature_dim)

                full_energy, mask_energy, mask_cond_energy, causal_energy, causal_cond_energy = \
                    self.forward_step(feature, action, delta_feature_candidates, forward_mode)
                energy = causal_energy

                if len(bs) > 1:
                    feature = feature.view(*bs, feature_dim)
                    delta_feature_candidates = delta_feature_candidates.view(*bs, num_pred_samples, feature_dim)
                    energy = energy.view(*bs, num_pred_samples, feature_dim)
            else:
                energy = self.forward_step(feature, action, delta_feature_candidates)

            if i != self.num_pred_iters - 1:
                energy = energy.transpose(-2, -1)                           # (bs, feature_dim, num_pred_samples)
                dist = Categorical(logits=energy)
                idxes = dist.sample([num_pred_samples])                     # (num_pred_samples, bs, feature_dim)
                idxes = idxes.permute(*(np.arange(len(bs)) + 1), 0, -1)     # (bs, num_pred_samples, feature_dim)

                # (bs, num_pred_samples, feature_dim)
                delta_feature_candidates = torch.gather(delta_feature_candidates, -2, idxes)
                noise = torch.randn_like(delta_feature_candidates) * sigma * (delta_feature_max - delta_feature_min)
                delta_feature_candidates += noise
                delta_feature_candidates = torch.clip(delta_feature_candidates, delta_feature_min, delta_feature_max)

                sigma *= self.pred_sigma_shrink

        argmax_idx = torch.argmax(energy, dim=-2, keepdim=True)             # (bs, 1, feature_dim)
        # (bs, feature_dim)
        delta_feature = torch.gather(delta_feature_candidates, -2, argmax_idx)[..., 0, :]
        pred_next_feature = feature + delta_feature

        return pred_next_feature

    def predict_with_feature(self, feature, actions):
        pred_next_features = []
        for action in torch.unbind(actions, dim=-2):
            feature = self.predict_step_with_feature(feature, action)
            pred_next_features.append(feature)
        return torch.stack(pred_next_features, dim=-2)

    def eval_prediction(self, obs, actions, next_obses):
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)

        feature = self.encoder(obs)
        next_features = self.encoder(next_obses)
        self.next_features = next_features
        pred_next_features = self.predict_with_feature(feature, actions)

        return feature, next_features, pred_next_features

    def preprocess(self, obs, actions, next_obses):
        if isinstance(actions, int):
            actions = np.array([actions])

        if isinstance(actions, np.ndarray):
            if self.continuous_action and actions.dtype != np.float32:
                actions = actions.astype(np.float32)
            if not self.continuous_action and actions.dtype != np.int64:
                actions = actions.astype(np.int64)
            actions = torch.from_numpy(actions).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            next_obses = postprocess_obs(preprocess_obs(next_obses, self.params))
            next_obses = {k: torch.from_numpy(v).to(self.device) for k, v in next_obses.items()}

        need_squeeze = False
        if actions.ndim == 1:
            need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}                          # (bs, obs_spec)
            actions = actions[None, None]                                       # (bs, num_pred_steps, action_dim)
            next_obses = {k: v[None, None] for k, v in next_obses.items()}      # (bs, num_pred_steps, obs_spec)
        elif self.params.env_params.num_envs > 1 and actions.ndim == 2:
            need_squeeze = True
            actions = actions[:, None]                                          # (bs, num_pred_steps, action_dim)
            next_obses = {k: v[:, None] for k, v in next_obses.items()}         # (bs, num_pred_steps, obs_spec)

        return obs, actions, next_obses, need_squeeze

    @staticmethod
    def reward_postprocess(reward, need_squeeze, output_numpy):
        if need_squeeze:
            reward = torch.squeeze(reward)                                      # scalar
        if output_numpy:
            reward = to_numpy(reward)
        return reward

    def get_state_abstraction(self):
        raise NotImplementedError

    def get_adjacency(self):
        return None

    def get_intervention_mask(self):
        return None

    def train(self, training=True):
        self.training = training
        super(Contrastive, self).train(training)

    def eval(self):
        self.train(False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("contrastive loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

