import os
import numpy as np

import torch

from model.inference import Inference
from model.inference_cmi import InferenceCMI
from model.gumbel import GumbelMatrix, ConditionalGumbelMatrix
from utils.utils import to_numpy, preprocess_obs, postprocess_obs

EPS = 1e-4


class InferenceReg(InferenceCMI):
    def __init__(self, encoder, decoder, params):
        self.reg_params = reg_params = params.inference_params.reg_params
        super(InferenceReg, self).__init__(encoder, decoder, params)
        self.use_mask = reg_params.use_mask

        # regularization params
        self.lambda_M = reg_params.lambda_M_init
        self.lambda_I = reg_params.lambda_I_init

    def init_model(self):
        super(InferenceReg, self).init_model()

        reg_params = self.reg_params
        device = self.device

        feature_dim = self.encoder.feature_dim

        self.adjacency = GumbelMatrix((feature_dim, feature_dim), reg_params.adjacency_init, device)
        self.interv_mask = GumbelMatrix((feature_dim, 1), reg_params.interv_mask_init, device)

        self.eye_mask = torch.eye(feature_dim, device=device).to(device)
        self.adjacency_mask = torch.ones(feature_dim, feature_dim, device=device) - self.eye_mask

        mask = None
        if feature_dim == 20:
            mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                                dtype=torch.float32, device=device)
        elif feature_dim == 25:
            mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],],
                                dtype=torch.float32, device=device)
        self.ground_truth_mask = mask

    def init_graph(self, *args):
        pass

    def init_abstraction(self):
        self.abstraction_quested = False
        self.abstraction_graph = None
        self.should_update_abstraction = False
        self.should_update_abstracted_dynamics = False

    def init_cache(self):
        pass

    def reset_causal_graph_eval(self):
        pass

    def setup_training_mask_distribution(self):
        pass

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
        # 1. extract features of all state variables and the action
        action_feature = self.extract_action_feature(action)            # (feature_dim, 1, bs, out_dim)
        state_feature = self.extract_state_feature(features)            # (feature_dim, feature_dim, bs, out_dim)

        mask = None
        if self.use_mask:
            # bs = action.shape[:-1]
            # # adjacency mask
            # M = self.adjacency(bs)                                      # (feature_dim, feature_dim, bs)
            # M = M * self.adjacency_mask + self.eye_mask                 # (feature_dim, feature_dim, bs)

            # # intervention mask
            # I = self.interv_mask(bs)                                    # (feature_dim, 1, bs)

            # mask = torch.cat([M, I], dim=-1)                            # (feature_dim, feature_dim + 1, bs)
            # mask = mask.unsqueeze(dim=-1)                               # (feature_dim, feature_dim + 1, bs, 1)

            if self.ground_truth_mask is not None:
                mask = self.ground_truth_mask[..., None, None]

        # mask out unused features
        sa_feature = torch.cat([state_feature, action_feature], dim=1)  # (feature_dim, feature_dim + 1, bs, out_dim)
        if mask is not None:
            sa_feature = sa_feature * mask                              # (feature_dim, feature_dim + 1, bs, out_dim)
        sa_feature, sa_indices = sa_feature.max(dim=1)                  # (feature_dim, bs, out_dim)

        # 3. predict the distribution of next time step value
        dist = self.predict_from_sa_feature(sa_feature, features[..., -1, :])

        return dist

    def forward_with_feature(self, features, actions, abstraction_mode=False):
        """
        :param features:
            if observation space is continuous: (bs, num_observation_steps, feature_dim).
            else: [(bs, num_observation_steps, feature_i_dim)] * feature_dim
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
        return Inference.forward_with_feature(self, features, actions, abstraction_mode)

    def eval_prediction(self, obses, actions, next_obses):
        return Inference.eval_prediction(self, obses, actions, next_obses)

    def forward(self, obses, actions, abstraction_mode=False):
        features = self.encoder(obses)
        return self.forward_with_feature(features, actions, abstraction_mode)

    def setup_annealing(self, step):
        reg_params = self.reg_params
        reg_annealing_start = reg_params.reg_annealing_start
        reg_annealing_end = reg_params.reg_annealing_end
        reg_annealing_coef = np.clip((step - reg_annealing_start) / (reg_annealing_end - reg_annealing_start), 0, 1)

        lambda_M_init = reg_params.lambda_M_init
        lambda_M_final = reg_params.lambda_M_final
        lambda_I_init = reg_params.lambda_I_init
        lambda_I_final = reg_params.lambda_I_final

        self.lambda_M = lambda_M_init + reg_annealing_coef * (lambda_M_final - lambda_M_init)
        self.lambda_I = lambda_I_init + reg_annealing_coef * (lambda_I_final - lambda_I_init)

    def loss(self, next_obses, pred_next_dist):
        """
        loss = reconstruction loss + adjacency regularization + interv_mask regularization
        :param next_obses: (bs, num_pred_steps, obs_spec)
        :param pred_next_dist: (full_dists, masked_dists, causal_dists) or (causal_dists,) only. For each distribution,
            if continuous state space, its form is a (sample, mean, log_std) tuple,
                each is a (bs, num_pred_steps, feature_dim) tensor
            if state space is discrete, its form is a (sample, logits) tuple,
                each is a [(bs, num_pred_steps, feature_i_dim)] * feature_dim list
        :return:
        """

        log_prob = self.eval_log_prob_from_dist(next_obses, pred_next_dist)[0]            # (bs, num_pred_steps)
        loss = -log_prob.sum(dim=-1).mean()

        loss_detail = {"pred_loss": loss}

        if self.use_mask:
            # L1 reg for adjacency M and intervention mask I
            reg_M = self.lambda_M * self.adjacency.get_prob().sum()     # scalar
            reg_I = self.lambda_I * self.interv_mask.get_prob().sum()   # scalar
            loss += reg_M + reg_I
            loss_detail["reg_M"] = reg_M
            loss_detail["reg_I"] = reg_I

        loss_detail["loss"] = loss

        return loss, loss_detail

    def update(self, obses, actions, next_obses, eval=False):
        loss_detail = Inference.update(self, obses, actions, next_obses, eval)

        if self.abstraction_quested:
            self.should_update_abstraction = True
            self.should_update_abstracted_dynamics = True

        return loss_detail

    def eval_log_prob_from_obs(self, obses, actions, next_obses):
        obses, actions, next_obses, _ = self.preprocess(obses, actions, next_obses)

        with torch.no_grad():
            pred_next_dist = self.forward(obses, actions)
            # if state space is continuous, distribution is a tuple for (sample, mean, log_std), where each element is
            # of shape (bs, feature_dim)
            # otherwise, distribution is a tuple for (sample, logits), where each element is of shape
            # [(bs, feature_i_dim)] * feature_dim
            next_feature = self.target_encoder(next_obses)
            # (bs, num_pred_steps, feature_dim)
            log_prob = self.eval_log_prob_from_dist(next_obses, pred_next_dist, keep_last_dim=True)[0]

        if self.continuous_state:
            _, mu, log_std = pred_next_dist
            mu, log_std = to_numpy(mu), to_numpy(log_std)
            dist = (mu, log_std)
            next_feature = to_numpy(next_feature)
        else:
            _, dist = pred_next_dist
            dist = [(to_numpy(dist_i[0]), to_numpy(dist_i[1])) if isinstance(dist_i, tuple) else to_numpy(dist_i)
                    for dist_i in dist]
            next_feature = [to_numpy(next_feature_i) for next_feature_i in next_feature]
        log_prob = to_numpy(log_prob)

        return dist, next_feature, log_prob

    def get_adjacency(self):
        if self.use_mask:
            M = self.adjacency.get_prob()
            M = M * self.adjacency_mask + self.eye_mask    # (feature_dim, feature_dim)
            return M
        else:
            return None

    def get_intervention_mask(self):
        if self.use_mask:
            return self.interv_mask.get_prob()
        else:
            return None

    def get_threshold(self):
        return 1.0

    def get_mask(self, boolean=True):
        if not self.use_mask:
            return None

        M = self.get_adjacency()
        I = self.get_intervention_mask()
        mask = torch.cat([M, I], dim=-1)
        if boolean:
            mask = mask > self.reg_params.mask_threshold

        return mask

    def train(self, training=True):
        self.training = training

        self.adjacency.training = training
        self.interv_mask.training = training

    def save(self, path):
        mask = self.get_mask(boolean=False)
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask": mask,
                    }, path)

    def load(self, path, device):
        Inference.load(self, path, device)
