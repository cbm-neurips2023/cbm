import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box

from tianshou.policy.modelfree.sac import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import ReplayBuffer, Batch

from utils.utils import to_numpy, preprocess_obs, postprocess_obs
from utils.imagination_replay_buffer import ImaginationReplayBuffer
from model.inference_utils import get_task_abstraction


class AlphaSchedule():
    def __init__(self, 
                 alpha_start, alpha_finish, horizon,
                 total_train_ts,
                 alpha_decay,
                 ):
        self.alpha_start = alpha_start
        self.alpha_finish = alpha_finish
        self.alpha_decay = alpha_decay  # higher values -> faster decay
        self.horizon = horizon

        self.total_train_ts = total_train_ts
        self.current_ts = 0

        self.step()

    def step(self):
        # multiplication by 10 is to maintain compatibility with hyperparams discovered before step bug fix
        self._alpha = (self.alpha_start - self.alpha_finish)*np.exp(-self.alpha_decay * 10 * (self.current_ts / self.total_train_ts)) + self.alpha_finish
        self.current_ts += 1 

class CustomSACPolicy(SACPolicy):
    '''override update function to convert batch sampled from replay buffer into Batch
    '''
    def __init__(self,
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            tau, gamma, alpha, action_space,
            grad_clip_norm, target_update_freq):
        super(CustomSACPolicy, self).__init__(actor, actor_optim,
                                              critic1, critic1_optim, critic2, critic2_optim,
                                              tau=tau, gamma=gamma, alpha=alpha,
                                              action_space=action_space)
        self.grad_clip_norm = grad_clip_norm
        self.target_update_freq = target_update_freq
        self.num_updates = 0
        self._is_alpha_schedule  = False
        if isinstance(alpha, AlphaSchedule):
            self._is_alpha_schedule = True
            self.alpha_schedule = alpha
            self._alpha = self.alpha_schedule._alpha

    def update(self,
               batch: Batch,
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network
        Assumption: batch contains the nstep reward, and has been sampled from the replay buffer
        """
        self.updating = True
        # batch = self.process_fn(batch, buffer, indices)
        losses = self.learn(batch, **kwargs)
        # self.post_process_fn(batch, buffer, indices) # TODO: figure out what this does

        # if self.lr_scheduler is not None:
            # self.lr_scheduler.step()
        self.updating = False
        return losses

    def _mse_optimizer(
        self, batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()

        grad_clip_norm = self.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)

        optimizer.step()
        return current_q.mean(), target_q.mean(), grad_norm, td, critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        cq1, tq1, grad_norm1, td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        cq2, tq2, grad_norm2, td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten() -
            torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        elif self._is_alpha_schedule:
            self.alpha_schedule.step()
            self._alpha = self.alpha_schedule._alpha

        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/grad_norm_q": grad_norm1.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore
        elif self._is_alpha_schedule:
            result["alpha"] = self._alpha

        return result


class ModelBasedSAC(nn.Module):
    def __init__(self, encoder, inference, reward_predictor, params, horizon, start_step=0):
        super(ModelBasedSAC, self).__init__()
        self.encoder = encoder
        self.inference = inference
        self.reward_predictor = reward_predictor
        self.horizon = horizon
        
        self.params = params
        self.device = device = params.device

        policy_params = params.policy_params
        self.num_td_steps = policy_params.num_td_steps
        self.batch_size = policy_params.batch_size
        self.sac_params = params.policy_params.sac_params
        self.use_imagination = self.sac_params.use_imagination
        if self.use_imagination:
            self.imag_replay_buffer = ImaginationReplayBuffer(params)

        self.use_dynamics_abstraction = self.sac_params.use_dynamics_abstraction
        self.use_reward_abstraction = self.sac_params.use_reward_abstraction
        self.use_partition_abstraction = self.sac_params.use_partition_abstraction
        self.use_abstraction = self.use_dynamics_abstraction or self.use_reward_abstraction or self.use_partition_abstraction

        assert self.params.continuous_state, "SAC can only handle continuous state"
        self.action_low, self.action_high = params.action_spec
        self.action_mean = (self.action_low + self.action_high) / 2
        self.action_scale = (self.action_high - self.action_low) / 2

        self.init_sac_model()
        if self.use_abstraction:
            self.update_abstraction()

        self.to(device)
        self.load(params.training_params.load_policy, device, start_step)
        self.train()

    def init_sac_model(self):
        action_dim = self.params.action_dim
        self.obs_dim = obs_dim = self.params.feature_dim + self.params.goal_dim
        action_space = Box(low=self.action_low,
                           high=self.action_high,
                           shape=(action_dim,),
                           dtype=np.float32)

        net_a = Net(obs_dim, hidden_sizes=self.sac_params.actor_dims, device=self.device)
        self.actor = ActorProb(
            net_a,
            action_dim,
            max_action=1.0, # TODO look into this later
            device=self.device,
            unbounded=True,
            conditioned_sigma=True
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.sac_params.actor_lr)
        net_c1 = Net(
            obs_dim,
            action_dim,
            hidden_sizes=self.sac_params.critic_dims,
            concat=True,
            device=self.device
        )
        net_c2 = Net(
            obs_dim,
            action_dim,
            hidden_sizes=self.sac_params.critic_dims,
            concat=True,
            device=self.device
        )
        self.critic1 = Critic(net_c1, device=self.device)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=self.sac_params.critic_lr)
        self.critic2 = Critic(net_c2, device=self.device)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=self.sac_params.critic_lr)

        if self.sac_params.auto_alpha:
            target_entropy = -np.prod(action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.sac_params.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        elif self.sac_params.alpha_schedule:
            alpha = AlphaSchedule(alpha_start=self.sac_params.alpha_start, 
                                  alpha_finish=self.sac_params.alpha_finish, 
                                  horizon=self.horizon,
                                  total_train_ts=self.params.training_params.total_step,
                                  alpha_decay=self.sac_params.alpha_decay)
            self.alpha_schedule = alpha
        else:
            alpha = self.sac_params.alpha

        self.policy = CustomSACPolicy(
            self.actor,
            self.actor_optim,
            self.critic1,
            self.critic1_optim,
            self.critic2,
            self.critic2_optim,
            tau=self.sac_params.tau,
            gamma=self.sac_params.gamma,  # TODO: where to pass in batch size, rest of args, etc.?
            alpha=alpha,
            action_space=action_space,
            grad_clip_norm=self.params.policy_params.grad_clip_norm,
            target_update_freq=self.sac_params.target_update_freq)

        self.abstraction_mask = torch.ones(self.obs_dim, device=self.device)

    def update_abstraction(self):
        prev_abstraction = self.abstraction_mask.clone()

        if self.use_partition_abstraction:
            partition = self.inference.mask
            partition.training = False
            abstraction_mask = partition(bs=[])[..., 0]
        elif self.use_reward_abstraction:
            dynamics_mask = to_numpy(self.inference.get_mask())
            reward_mask = to_numpy(self.reward_predictor.get_mask(return_bool=True))
            abstraction_mask = get_task_abstraction(reward_mask, dynamics_mask)
        elif self.use_dynamics_abstraction:
            abstraction_mask = self.inference.get_state_abstraction()

            # mask = np.zeros(40)
            # mask[0:3] = 1
            # mask[6:8] = 1
            # mask[10:13] = 1
            # mask[24:27] = 1
            # abstraction_mask = mask

        if isinstance(abstraction_mask, np.ndarray):
            abstraction_mask = torch.tensor(abstraction_mask, dtype=torch.float32, device=self.device)
        self.abstraction_mask[:len(abstraction_mask)] = abstraction_mask

        if (self.abstraction_mask != prev_abstraction).any():
            self.reset_network()

    def reset_network(self):
        for module in self.policy.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        if self.sac_params.alpha_schedule and self.sac_params.reset_alpha:
            self.alpha_schedule.current_ts = 0

    def setup_annealing(self, step):
        if step > self.params.training_params.reduce_reward_opt_step:
            self.sac_params.abstraction_update_freq = self.params.training_params.total_step
        if step % self.sac_params.abstraction_update_freq == 0 and self.use_abstraction:
            self.update_abstraction()

    def act_randomly(self):
        num_envs = self.params.env_params.num_envs
        action_dim = self.params.action_dim

        if self.params.continuous_action:
            return self.action_mean + self.action_scale * np.random.uniform(-1, 1, (num_envs, action_dim))
        else:
            return np.random.randint(action_dim, size=num_envs)

    def act(self, obs, deterministic=False):
        """
        Sample from Tianshou SAC policy.
        :param obs: (obs_spec)

        """
        # self.inference.eval()
        # self.policy.eval()
        self.policy.train(mode=not deterministic) # eval mode controls sampling in Tianshou

        obs = postprocess_obs(preprocess_obs(obs, self.params))
        eef_pos = obs["robot0_eef_pos"]
        obs = {k: torch.tensor(v, device=self.device) for k, v in obs.items()}
        feat = self.encoder(obs, include_goal=True)
        if self.use_abstraction:
            feat = feat * self.abstraction_mask

        batch = Batch(obs=feat, info=None)
        forward_res = self.policy(batch) # TODO: check if anything else is needed to pred action
        action = to_numpy(forward_res.act)

        return action

    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        feat_batch = self.encoder(obs_batch, include_goal=True)
        next_feat_batch = self.encoder(next_obs_batch, include_goal=True)
        if self.use_abstraction:
            feat_batch = feat_batch * self.abstraction_mask
            next_feat_batch = next_feat_batch * self.abstraction_mask

        # compute n-step return
        nstep = self.num_td_steps
        gamma = self.sac_params.gamma
        target_q_batch = self.compute_target_q(next_feat_batch[:, -1, :]) # compute target q at step t+n
        nstep_return = self.compute_nstep_return(nstep, gamma, rew_batch, target_q_batch, done_batch)

        batch = Batch(obs=feat_batch[:, 0, :],
                      act=act_batch[:, 0, :],
                      obs_next=next_feat_batch[:, 0, :],
                      returns=nstep_return,
                      done=done_batch[:, 0, :],
                      info=None)
        loss_detail = self.policy.update(batch) # dictionary of loss updates
        # print("LOSS IS ", loss_detail)

        # generate data
        if self.use_imagination:
            perm = torch.randperm(self.batch_size)
            imag_idxs = perm[:self.sac_params.imag_gen_batch_size]
            
            feat_batch = self.encoder(obs_batch)
            feat_goal_batch = self.reward_predictor.extract_goal_feature(obs_batch)
            next_feat_goal_batch = self.reward_predictor.extract_goal_feature(next_obs_batch)

            gen_feat_batch, gen_act_batch, gen_rew_batch, gen_next_feat_batch = \
                self.generate_data(feat_batch[imag_idxs, 0, :], 
                                   feat_goal_batch[imag_idxs], 
                                   next_feat_goal_batch[imag_idxs])

            self.imag_replay_buffer.add(gen_feat_batch, 
                                        gen_act_batch, 
                                        gen_rew_batch, 
                                        done_batch[imag_idxs], 
                                        gen_next_feat_batch)

            # perform multiple updates on imagination buffer
            for _ in range(self.sac_params.num_imagination_updates):
                feat_batch, act_batch, rew_batch, done_batch, next_feat_batch = self.imag_replay_buffer.sample(self.sac_params.imag_update_batch_size)
                if self.use_abstraction:
                    feat_batch = feat_batch * self.abstraction_mask
                    next_feat_batch = next_feat_batch * self.abstraction_mask

                target_q_batch = self.compute_target_q(next_feat_batch[:, -1, :]) # compute target q at step t+n
                nstep_return = self.compute_nstep_return(nstep, gamma, rew_batch, target_q_batch, done_batch)
                batch = Batch(obs=feat_batch[:, 0, :],
                            act=act_batch[:, 0, :],
                            obs_next=next_feat_batch[:, 0, :],
                            returns=nstep_return,
                            done=done_batch[:, 0, :],
                            info=None)

                imag_loss_detail = self.policy.update(batch) # dictionary of loss updates
            # log stats for last update
            for k, v in imag_loss_detail.items():
                loss_detail[k + "_imag"] = v
        return loss_detail

    def compute_nstep_return(self, nstep, gamma, rew_batch, target_q_batch, done_batch):
        '''
        :param rew_batch: shape (batch_size, nstep, 1)
        :param target_q_batch: shape (batch_size, 1, 1)
        :param done_batch: (batch_size, nstep, 1)
        :return nstep_return: (batch_size, 1)
        '''
        # assumption: done_batch is 1 if done, 0 if not done
        # done batch also has -1, which can be treated as 1
        done_batch[done_batch == -1] = 1
        done_mask = 1 - done_batch

        nstep_return = target_q_batch # computed for next_feat[:, nstep-1, :]
        for n in range(nstep - 1, -1, -1):
            nstep_return = rew_batch[:, n, :] + gamma * nstep_return * done_mask[:, n, :]

        return nstep_return

    def compute_target_q(self, next_feat_batch):
        '''Compute target q value n steps ahead
        '''
        with torch.no_grad():
            obs_next_result = self.policy(Batch(obs=next_feat_batch, info=None))
            act_ = obs_next_result.act
            target_q = torch.min(
                self.policy.critic1_old(next_feat_batch, act_),
                self.policy.critic2_old(next_feat_batch, act_),
            ) - self.policy._alpha * obs_next_result.log_prob
        return target_q

    def generate_data(self, feat_batch, feat_goal_batch, next_feat_goal_batch):
        '''starting from obs_batch, generate data using SAC policy
        Generate rollouts for at least n_step/num_td_steps
        so that we can compute the n_step return
        feat_batch: (batch_size, feat_size)
        '''
        self.inference.eval()
        self.reward_predictor.eval()
        with torch.no_grad():
            total_feat = []
            total_act = []
            total_rew = []
            total_next_feat = []

            goal_batch = None
            if feat_goal_batch is not None:
                goal_batch = feat_goal_batch[:, 0, :]

            for i in range(self.num_td_steps):
                policy_feat_batch = feat_batch
                if goal_batch is not None:
                    policy_feat_batch = torch.cat([feat_batch, goal_batch], dim=-1)
                act_batch = self.policy(Batch(obs=policy_feat_batch, info=None)).act
                # input of inference.forward_with_feature should be (batch_size, 1, feature_dim)
                # assume output of inference.forward_with_feature will be (batch_size, num_step=1, feature_dim)
                next_feat_batch = self.inference.predict_with_feature(feat_batch, act_batch.unsqueeze(1))[:, 0, :]
                pred_rew = self.reward_predictor.pred_reward_with_feature(feat_batch,
                    act_batch, goal_batch, next_feat_batch)

                # generate new act
                total_feat.append(feat_batch)
                total_act.append(act_batch)
                total_rew.append(pred_rew)
                total_next_feat.append(next_feat_batch)

                feat_batch = next_feat_batch

            total_feat = torch.stack(total_feat, dim=1)
            total_act = torch.stack(total_act, dim=1)
            total_rew = torch.stack(total_rew, dim=1)
            total_next_feat = torch.stack(total_next_feat, dim=1)

            if feat_goal_batch is not None:
                total_feat = torch.cat([total_feat, feat_goal_batch], dim=-1)
                total_next_feat = torch.cat([total_next_feat, next_feat_goal_batch], dim=-1)

        return total_feat, total_act, total_rew, total_next_feat


    def save(self, path):
        if self.use_imagination:
            self.imag_replay_buffer.save()
        torch.save({"actor": self.actor.state_dict(),
                    "actor_optim": self.actor_optim.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic2": self.critic2.state_dict(),
                    "critic1_optim": self.critic1_optim.state_dict(),
                    "critic2_optim": self.critic2_optim.state_dict()
                    }, path)

    def load(self, path, device, start_step=0):
        if path is not None and os.path.exists(path):
            print("ModelFree SAC loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic1.load_state_dict(checkpoint["critic1"])
            self.critic2.load_state_dict(checkpoint["critic2"])

            # load optimizers
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
            self.critic1_optim.load_state_dict(checkpoint["critic1_optim"])
            self.critic2_optim.load_state_dict(checkpoint["critic2_optim"])

            # set alpha schedule to correct step 
            if self.sac_params.alpha_schedule:
                self.alpha_schedule.current_ts = start_step * self.params.training_params.num_policy_opt_steps

