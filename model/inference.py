import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.kl import kl_divergence

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class Inference(nn.Module):
    def __init__(self, encoder, decoder, params):
        super(Inference, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.params = params
        self.device = device = params.device
        self.inference_params = inference_params = params.inference_params
        self.training_params = training_params = params.training_params

        self.log_std_min = inference_params.log_std_min
        self.log_std_max = inference_params.log_std_max
        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action

        self.object_level_obs = training_params.object_level_obs

        self.init_model()
        self.reset_params()

        self.abstraction_quested = False

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=inference_params.lr)

        self.load(params.training_params.load_inference, device)
        self.train()

    def init_model(self):
        raise NotImplementedError

    def reset_params(self):
        pass

    def forward_step(self, feature, action):
        """
        :param feature:
            if observation space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if observation space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        raise NotImplementedError

    def forward_step_abstraction(self, abstraction_features, action):
        """
        :param abstraction_features:
            if observation space is continuous: (bs, abstraction_feature_dim).
            else: [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all abstracted state variables in the format of distribution,
            if observation space is continuous: a Normal distribution of shape (bs, abstraction_feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * abstraction_feature_dim,
                each of shape (bs, feature_i_dim)
        """
        raise NotImplementedError

    def stack_dist(self, dist_list):
        """
        list of distribution at different time steps to a single distribution stacked at dim=-2
        :param dist_list:
            if observation space is continuous: [Normal] * num_pred_steps, each of shape (bs, feature_dim)
            else: [[OneHotCategorical / Normal]  * feature_dim] * num_pred_steps, each of shape (bs, feature_i_dim)
            notice that bs can be a multi-dimensional batch size
        :return:
            if observation space is continuous: Normal distribution of shape (bs, num_pred_steps, feature_dim)
            else: [OneHotCategorical / Normal]  * feature_dim, each of shape (bs, num_pred_steps, feature_i_dim)
        """
        if self.continuous_state:
            mu = torch.stack([dist.mean for dist in dist_list], dim=-2)         # (bs, num_pred_steps, feature_dim)
            std = torch.stack([dist.stddev for dist in dist_list], dim=-2)      # (bs, num_pred_steps, feature_dim)
            return Normal(mu, std)
        else:
            # [(bs, num_pred_steps, feature_i_dim)]
            stacked_dist_list = []
            for i, dist_i in enumerate(dist_list[0]):
                if isinstance(dist_i, Normal):
                    # (bs, num_pred_steps, feature_i_dim)
                    mu = torch.stack([dist[i].mean for dist in dist_list], dim=-2)
                    std = torch.stack([dist[i].stddev for dist in dist_list], dim=-2)
                    stacked_dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    # (bs, num_pred_steps, feature_i_dim)
                    logits = torch.stack([dist[i].logits for dist in dist_list], dim=-2)
                    stacked_dist_i = OneHotCategorical(logits=logits)
                else:
                    raise NotImplementedError
                stacked_dist_list.append(stacked_dist_i)

            return stacked_dist_list

    def sample_from_distribution(self, dist):
        """
        sample from the distribution
        :param dist:
            if observation space is continuous: Normal distribution of shape (bs, feature_dim).
            else: [OneHotCategorical / Normal]  * feature_dim, each of shape (bs, feature_i_dim)
            notice that bs can be a multi-dimensional batch size
        :return:
            if observation space is continuous: (bs, feature_dim)
            else: [(bs, feature_i_dim)]  * feature_dim
        """
        if dist is None:
            # for cmi
            return None

        if self.continuous_state:
            return dist.rsample() if self.training else dist.mean
        else:
            sample = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    sample_i = dist_i.rsample() if self.training else dist_i.mean
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits
                    if self.training:
                        sample_i = F.gumbel_softmax(logits, hard=True)
                    else:
                        sample_i = F.one_hot(torch.argmax(logits, dim=-1), logits.size(-1)).float()
                else:
                    raise NotImplementedError
                sample.append(sample_i)
            return sample

    def log_prob_from_distribution(self, dist, value):
        """
        calculate log_prob of value from the distribution
        :param dist:
            if observation space is continuous: Normal distribution of shape (bs, feature_dim).
            else: [OneHotCategorical / Normal]  * feature_dim, each of shape (bs, feature_i_dim)
            notice that bs can be a multi-dimensional batch size
        :param value:
            if observation space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)]  * feature_dim
        :return: (bs, feature_dim)
        """
        if self.continuous_state:
            return dist.log_prob(value)
        else:
            log_prob = []
            for dist_i, val_i in zip(dist, value):
                log_prob_i = dist_i.log_prob(val_i)
                if isinstance(dist_i, Normal) and not self.object_level_obs:
                    log_prob_i = log_prob_i.squeeze(dim=-1)
                log_prob.append(log_prob_i)
            return torch.cat(log_prob, dim=-1) if self.object_level_obs else torch.stack(log_prob, dim=-1)

    def forward_with_feature(self, feature, actions, abstraction_mode=False):
        """
        :param feature:
            if observation space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param actions:
            if observation space is continuous: (bs, num_pred_steps, action_dim)
            else: (bs, num_pred_steps, 1)
        :param abstraction_mode: whether to only forward controllable & action-relevant state variables,
            used for model-based roll-out
        :return: next step value for all (abstracted) state variables in the format of distribution,
            if observation space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """

        if abstraction_mode and self.abstraction_quested:
            if self.continuous_state:
                feature = feature[..., self.abstraction_idxes]
            else:
                feature = [feature[idx] for idx in self.abstraction_idxes]

        if not self.continuous_action:
            actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()   # (bs, num_pred_steps, action_dim)
        actions = torch.unbind(actions, dim=-2)                                     # [(bs, action_dim)] * num_pred_steps

        dists = []
        for action in actions:
            if abstraction_mode and self.abstraction_quested:
                dist = self.forward_step_abstraction(feature, action)
            else:
                dist = self.forward_step(feature, action)

            next_feature = self.sample_from_distribution(dist)
            feature = next_feature
            dists.append(dist)
        dists = self.stack_dist(dists)

        return dists

    def forward(self, obs, actions, abstraction_mode=False):
        feature = self.encoder(obs)
        return self.forward_with_feature(feature, actions, abstraction_mode)

    def setup_annealing(self, step):
        pass

    def prediction_loss_from_dist(self, pred_dist, next_feature, keep_variable_dim=False):
        """
        calculate prediction loss from the prediction distribution
        if use a CNN encoder: prediction loss = KL divergence
        else: prediction loss = -log_prob
        :param pred_dist: next step value for all state variables in the format of distribution,
            if observation space is continuous:
                a Normal distribution of shape (bs, num_pred_steps, feature_dim)
            else: 
                a list of distributions, [OneHotCategorical / Normal] * feature_dim, 
                each of shape (bs, num_pred_steps, feature_i_dim)
        :param next_feature:
            if use a CNN encoder:
                a Normal distribution of shape (bs, num_pred_steps, feature_dim)
            elif observation space is continuous:
                a tensor of shape (bs, num_pred_steps, feature_dim)
            else:
                a list of tensors, [(bs, num_pred_steps, feature_i_dim)] * feature_dim
        :param keep_variable_dim: whether to keep the dimension of state variables which is dim=-1
        :return: (bs, num_pred_steps, feature_dim) if keep_variable_dim else (bs, num_pred_steps)
        """
        if isinstance(next_feature, Distribution):
            assert isinstance(next_feature, Normal)
            next_feature = Normal(next_feature.mean.detach(), next_feature.stddev.detach())
            pred_loss = kl_divergence(next_feature, pred_dist)                          # (bs, num_pred_steps, feature_dim)
        else:
            if self.continuous_state:
                next_feature = next_feature.detach()
            else:
                next_feature = [next_feature_i.detach() for next_feature_i in next_feature]
            pred_loss = -self.log_prob_from_distribution(pred_dist, next_feature)       # (bs, num_pred_steps, feature_dim)

        if not keep_variable_dim:
            pred_loss = pred_loss.sum(dim=-1)                                           # (bs, num_pred_steps)

        return pred_loss

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.inference_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        self.optimizer.step()
        return loss_detail

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_obses: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        feature = self.encoder(obs)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(feature, actions)

        # prediction loss in the state / latent space
        pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)    # (bs, num_pred_steps)
        loss = pred_loss = pred_loss.sum(dim=-1).mean()
        loss_detail = {"pred_loss": pred_loss}

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def update_mask(self, obs, actions, next_obses):
        raise NotImplementedError

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

    def eval_prediction(self, obs, actions, next_obses):
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)

        with torch.no_grad():
            feature = self.encoder(obs)
            next_features = self.encoder(next_obses)
            pred_next_dist = self.forward_with_feature(feature, actions)

            # prediction loss in the state / latent space
            # (bs, num_pred_steps, feature_dim)
            pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features, keep_variable_dim=True)

            accuracy = None
            if not self.continuous_state and not self.object_level_obs:
                accuracy = []
                for dist_i, next_feature_i in zip(pred_next_dist, next_features):
                    if not isinstance(dist_i, OneHotCategorical):
                        continue
                    logits = dist_i.logits                                 # (bs, num_pred_steps, feature_i_inner_dim)
                    # (bs, num_pred_steps)
                    accuracy_i = logits.argmax(dim=-1) == next_feature_i.argmax(dim=-1)
                    accuracy.append(accuracy_i)
                accuracy = torch.stack(accuracy, dim=-1)
                accuracy = to_numpy(accuracy)

            if self.object_level_obs:
                pred_next_dist = Normal(torch.cat([dist.mean for dist in pred_next_dist], dim=-1),
                                        torch.cat([dist.stddev for dist in pred_next_dist], dim=-1))
                next_features = torch.cat(next_features, dim=-1)

        return pred_next_dist, next_features, pred_loss, accuracy

    def get_state_abstraction(self):
        raise NotImplementedError

    def get_adjacency(self):
        return None

    def get_intervention_mask(self):
        return None

    def get_threshold(self):
        raise NotImplementedError

    def get_mask(self):
        return torch.cat([self.get_adjacency(), self.get_intervention_mask()], dim=-1)

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("inference loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
