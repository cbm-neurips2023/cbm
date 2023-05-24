import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class RewardPredictorDense(nn.Module):
    def __init__(self, encoder, params, denoised_mask=None, tia_mask=None):
        super(RewardPredictorDense, self).__init__()

        self.encoder = encoder
        self.denoised_mask = denoised_mask
        self.tia_mask = tia_mask

        self.params = params
        self.continuous_state = params.continuous_state
        if not self.continuous_state:
            raise NotImplementedError

        self.device = device = params.device
        self.reward_predictor_params = reward_predictor_params = params.reward_predictor_params

        self.use_next_state = reward_predictor_params.use_next_state

        self.init_model()
        self.reset_params()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=reward_predictor_params.lr)

        self.load(params.training_params.load_reward_predictor, device)
        self.train()

    @staticmethod
    def get_mlp(in_dim, fc_dims, activations=None):
        if activations is None:
            activations = ["relu"] * len(fc_dims)

        modules = []
        for i, (out_dim, activation) in enumerate(zip(fc_dims, activations)):
            modules.append(nn.Linear(in_dim, out_dim))
            if i != len(fc_dims) - 1:
                if activation == "relu":
                    activation = nn.ReLU()
                elif activation == "leaky_relu":
                    activation = nn.LeakyReLU()
                elif activation == "tanh":
                    activation = nn.Tanh()
                else:
                    raise ValueError("Unknown activation: {}".format(activation))
                modules.append(activation)
            in_dim = out_dim
        return nn.Sequential(*modules)

    def init_model(self):
        params = self.params
        dense_params = self.reward_predictor_params.dense_params

        self.feature_dim = feature_dim = params.feature_dim
        self.action_dim = action_dim = params.action_dim

        self.goal_keys = self.encoder.goal_keys
        goal_dim = params.goal_dim

        if self.use_next_state:
            in_dim = feature_dim + goal_dim
        else:
            in_dim = feature_dim + goal_dim + action_dim

        self.fcs = self.get_mlp(in_dim, dense_params.fc_dims + [1], dense_params.activations + ["relu"])

    def reset_params(self):
        pass

    def setup_annealing(self, step):
        pass

    def extract_goal_feature(self, obs):
        if not self.goal_keys:
            return None

        goal = torch.cat([obs[k] for k in self.goal_keys], dim=-1)
        if self.continuous_state:
            return goal
        else:
            raise NotImplementedError

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature):
        if not self.continuous_state:
            raise NotImplementedError

        if self.use_next_state:
            inputs = [next_feature]
        else:
            inputs = [feature, action]

        if goal_feature is not None:
            inputs.append(goal_feature)

        inputs = torch.cat(inputs, dim=-1)

        pred_reward = self.fcs(inputs)
        return pred_reward

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, obs, action, next_obs, reward, eval=False):
        feature = self.encoder(obs)
        goal_feature = self.extract_goal_feature(obs)
        next_feature = self.encoder(next_obs)

        pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature)

        pred_error = torch.abs(pred_reward - reward).squeeze(dim=-1)
        loss = pred_error.mean()
        loss_detail = {"pred_loss": loss,
                       "priority": to_numpy(pred_error)}

        if not eval:
            self.backprop(loss)

        return loss_detail

    def preprocess(self, obs, action, next_obs):
        if isinstance(action, np.ndarray):
            if action.dtype != np.float32:
                action = action.astype(np.float32)
            action = torch.from_numpy(action).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            next_obs = postprocess_obs(preprocess_obs(next_obs, self.params))
            next_obs = {k: torch.from_numpy(v).to(self.device) for k, v in next_obs.items()}

        need_squeeze = False
        if action.ndim == 1:
            need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}                              # (bs, obs_spec)
            action = action[None]                                                   # (bs, action_dim)
            next_obs = {k: v[None] for k, v in next_obs.items()}                    # (bs, obs_spec)

        return obs, action, next_obs, need_squeeze

    def pred_reward(self, obs, action, next_obs, output_numpy=False):
        obs, action, next_obs, need_squeeze = self.preprocess(obs, action, next_obs)
        feature = self.encoder(obs)
        goal_feature = self.extract_goal_feature(obs)
        next_feature = self.encoder(next_obs)
        pred_reward = self.pred_reward_with_feature(feature, action, goal_feature, next_feature)

        if need_squeeze:
            pred_reward = torch.squeeze(pred_reward)                                # scalar

        if output_numpy:
            pred_reward = to_numpy(pred_reward)

        return pred_reward

    def get_mask(self, return_bool=False):
        return None

    def train(self, training=True):
        self.training = training
        super(RewardPredictorDense, self).train(training)

    def eval(self):
        self.train(False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("reward predictor loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])


class TrueReward(nn.Module):
    def __init__(self, encoder, params):
        super(TrueReward, self).__init__()

        self.encoder = encoder

        self.params = params
        self.device = device = params.device
        self.reward_predictor_params = reward_predictor_params = params.reward_predictor_params

        self.goal_keys = params.goal_keys
        self.continuous_state = params.continuous_state

        global_low = np.array([-0.5, -0.5, 0.7])
        global_high = np.array([0.5, 0.5, 1.1])
        self.global_mean = torch.tensor((global_high + global_low) / 2, device=device)
        self.global_scale = torch.tensor((global_high - global_low) / 2, device=device)

    def extract_goal_feature(self, obs):
        if not self.goal_keys:
            return None

        goal = torch.cat([obs[k] for k in self.goal_keys], dim=-1)
        if self.continuous_state:
            return goal
        else:
            raise NotImplementedError

    def pred_reward_with_feature(self, feature, action, goal_feature, next_feature):
        if not self.continuous_state:
            feature = torch.cat(feature, dim=-1)

        env_name = self.params.env_params.env_name
        if env_name == "CausalReach":
            eef_pos = feature[..., 0:3] * self.global_scale + self.global_mean
            goal_pos = goal_feature[..., 0:3] * self.global_scale + self.global_mean
            dist = torch.abs(eef_pos - goal_pos).sum(dim=-1, keepdim=True)
            pred_reward = 1 - torch.tanh(5 * dist)
        elif env_name == "CausalPush":
            reach_mult = 0.5
            push_mult = 1.0

            eef_pos = feature[..., 0:3]
            mov_pos = feature[..., 10:13]
            goal_pos = goal_feature[..., 0:3]

            dist1 = torch.norm(eef_pos - mov_pos, dim=-1, keepdim=True)
            dist2 = torch.norm(mov_pos - goal_pos, dim=-1, keepdim=True)
            pred_reward = (1 - torch.tanh(5.0 * dist1)) * reach_mult + (1 - torch.tanh(5.0 * dist2)) * push_mult
        elif env_name == "CausalPick":
            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 0.5

            reward = 0

            eef_pos = feature[..., 0:3] * self.global_scale + self.global_mean
            mov_pos = feature[..., 10:13] * self.global_scale + self.global_mean
            goal_pos = goal_feature[..., 0:3] * self.global_scale + self.global_mean
            gripper_open = action[..., -1:] < 0
            gripper_close = action[..., -1:] > 0

            dist = torch.norm(eef_pos - mov_pos, dim=-1, keepdim=True)
            r_reach = (1 - torch.tanh(5.0 * dist)) * reach_mult

            grasping_cubeA = (dist < 0.02) * gripper_close
            r_reach += grasp_mult * grasping_cubeA

            reward += r_reach

            dist = torch.norm(goal_pos - mov_pos, dim=-1, keepdim=True)
            r_lift = (1 - torch.tanh(5.0 * dist)) * lift_mult
            reward += r_lift

            reward /= (reach_mult + grasp_mult + lift_mult)
        elif env_name == "CausalStack":
            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 1.0
            stack_mult = 2.0

            lift_height = 0.95
            xy_max_dist = 1.0
            z_max_dist = 0.2

            eef_pos = feature[..., 0:3] * self.global_scale + self.global_mean
            eef_xy = eef_pos[..., 0:2]
            eef_z = eef_pos[..., 2:3]

            mov_pos = feature[..., 10:13] * self.global_scale + self.global_mean
            mov_xy = mov_pos[..., 0:2]
            mov_z = mov_pos[..., 2:3]

            unmov_pos = feature[..., 15:18] * self.global_scale + self.global_mean
            unmov_xy = feature[..., 0:2]
            unmov_z = feature[..., 2:3]

            gripper_open = action[..., -1:] < 0
            gripper_close = action[..., -1:] > 0

            dist = torch.norm(eef_pos - mov_pos, dim=-1, keepdim=True)
            r_reach = (1 - np.tanh(5.0 * dist)) * reach_mult

            # grasping reward
            grasping_cubeA = (dist < 0.02) * gripper_close
            r_reach += grasp_mult * grasping_cubeA

            # lifting is successful when the cube is above the table top by a margin
            table_height = 0.8
            cubeA_lifted = mov_z > table_height + 0.1
            r_lift = lift_mult * cubeA_lifted

            # Aligning is successful when cubeA is right above cubeB
            horiz_dist = torch.norm(mov_xy - unmov_xy, dim=-1, keepdim=True)
            r_lift += lift_mult * (1 - np.tanh(5.0 * horiz_dist)) * cubeA_lifted

            # stacking is successful when the block is lifted and the gripper is not holding the object
            vert_dist = mov_z - unmov_z
            r_stack = stack_mult * (horiz_dist < 0.01) * (vert_dist > 0.04) * gripper_open

            reward = (r_reach + r_lift + r_stack) / (reach_mult + grasp_mult + 2 * lift_mult + stack_mult)
        else:
            raise NotImplementedError

        return pred_reward

    def preprocess(self, obs, action):
        if isinstance(action, np.ndarray):
            if action.dtype != np.float32:
                action = action.astype(np.float32)
            action = torch.from_numpy(action).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}

        need_squeeze = False
        if action.ndim == 1:
            need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}                              # (bs, obs_spec)
            action = action[None]                                                   # (bs, action_dim)

        return obs, action, need_squeeze

    def pred_reward(self, obs, action):
        feature = self.encoder(obs)
        goal_feature = self.extract_goal_feature(obs)
        pred_reward = self.pred_reward_with_feature(feature, action, goal_feature)

        if reward_need_squeeze:
            pred_reward = torch.squeeze(pred_reward)                                # scalar

        if output_numpy:
            pred_reward = to_numpy(pred_reward)

        return pred_reward

    def save(self, path):
        pass