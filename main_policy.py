import os
import numpy as np
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_default_tensor_type(torch.FloatTensor)

from model.encoder import make_encoder
from model.decoder import Decoder

from model.gumbel import GumbelPartition

from model.inference_mlp import InferenceMLP
from model.inference_gnn import InferenceGNN
from model.inference_reg import InferenceReg
from model.inference_nps import InferenceNPS
from model.inference_attention import InferenceAttn
from model.inference_cmi import InferenceCMI

from model.contrastive_mod import ContrastiveModular
from model.contrastive_cmi import ContrastiveCMI
from model.contrastive_mask import ContrastiveMask

# data collection policy for dynamics training
from model.random_policy import RandomPolicy
from model.hippo import HiPPO

from model.reward_predictor import RewardPredictorDense, TrueReward
from model.reward_predictor_cmi import RewardPredictorCMI
from model.reward_predictor_contrastive import RewardPredictorContrastive
from model.reward_predictor_tia import RewardPredictorTIA
from model.reward_predictor_denoised import RewardPredictorDenoised

# policy to solve downstream tasks
from model.cem import CEM
from model.collocation import Collocation
from model.model_based_sac import ModelBasedSAC

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, get_start_step_from_model_loading, to_float
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.plot import plot_dynamics_mask, plot_reward_mask, plot_partition, plot_abstraction
from utils.scripted_policy import get_scripted_policy, get_is_demo

from env.chemical_env import Chemical


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args, unknown = parser.parse_known_args()
    return args

def run_test_episodes(env, writer, rl_algo, policy, inference,
                      num_test_envs, num_test_eps, 
                      curr_train_ep):
    inference.eval()
    policy.eval()

    success = np.zeros(num_test_envs, dtype=bool)
    episode_reward = np.zeros(num_test_envs)
    tot_success = np.zeros(num_test_eps, dtype=bool)
    tot_ep_rew = np.zeros(num_test_eps)
    done = np.zeros(num_test_envs, dtype=bool)
    episode_step = np.zeros(num_test_envs)

    obs = env.reset()
    episode_num = 0
    while episode_num <= num_test_eps:
        if done.any():
            for i, done_i in enumerate(done):
                if not done_i:
                    continue
                if episode_num < num_test_eps:
                    assert tot_ep_rew[episode_num] == 0
                    tot_ep_rew[episode_num] = episode_reward[i]
                    tot_success[episode_num] = success[i]

                episode_reward[i] = 0
                episode_step[i] = 0
                success[i] = False
                episode_num += 1

        # get action (num_envs, action_dim)
        action = policy.act(obs, deterministic=True)

        # (num_envs, obs_spec), (num_envs,), (num_envs,), [info] * num_envs
        next_obs, reward, done, info = env.step(action)

        success_step = np.array([info_i["success"] for info_i in info])
        success = success | success_step

        episode_reward += reward
        episode_step += 1
        obs = next_obs
    # log mean reward and success
    writer.add_scalar("policy_stat/test_mean_episode_reward", np.mean(tot_ep_rew), curr_train_ep)
    writer.add_scalar("policy_stat/test_mean_success", np.mean(tot_success), curr_train_ep)


def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(params.seed)

    params.device = device
    env_params = params.env_params
    training_params = params.training_params
    testing_params = params.testing_params
    replay_buffer_params = training_params.replay_buffer_params
    inference_params = params.inference_params
    contrastive_params = params.contrastive_params
    policy_params = params.policy_params

    # init environment
    render = False
    num_envs = params.env_params.num_envs
    num_test_envs = testing_params.num_test_envs

    env = get_env(params, num_envs, render)
    horizon = env.horizon


    if env_params.env_name == "Chemical":
        torch.save(env.get_save_information(), os.path.join(params.rslts_dir, "chemical_env_params"))

    # init model
    update_obs_act_spec(env, params)
    encoder = make_encoder(params)
    params.feature_dim = encoder.feature_dim
    params.goal_dim = encoder.goal_dim
    use_decoder = params.decoder_params.use_decoder
    decoder = Decoder(params) if use_decoder else None

    rew_pred_algo = params.training_params.rew_pred_algo

    partition = None
    need_partition = rew_pred_algo in ["tia", "denoised"]
    if need_partition:
        if rew_pred_algo == "tia":
            num_partitions = 2
        elif rew_pred_algo == "denoised":
            num_partitions = 3

        partition = GumbelPartition((params.feature_dim, num_partitions), params)

    if rew_pred_algo == "dense":
        RewardPredictor = RewardPredictorDense
    elif rew_pred_algo == "cmi":
        RewardPredictor = RewardPredictorCMI
    elif rew_pred_algo == "contrastive":
        RewardPredictor = RewardPredictorContrastive
    elif rew_pred_algo == "tia":
        RewardPredictor = RewardPredictorTIA
    elif rew_pred_algo == "denoised":
        RewardPredictor = RewardPredictorDenoised
    elif rew_pred_algo == "true":
        RewardPredictor = TrueReward
    else:
        raise NotImplementedError

    if need_partition:
        reward_predictor = RewardPredictor(encoder, partition, params)
    else:
        reward_predictor = RewardPredictor(encoder, params)

    inference_algo = params.training_params.inference_algo
    if inference_algo == "mlp":
        Inference = InferenceMLP
    elif inference_algo == "gnn":
        Inference = InferenceGNN
    elif inference_algo == "reg":
        Inference = InferenceReg
    elif inference_algo == "nps":
        Inference = InferenceNPS
    elif inference_algo == "attn":
        Inference = InferenceAttn
    elif inference_algo == "cmi":
        Inference = InferenceCMI
    elif inference_algo == "contrastive_mod":
        Inference = ContrastiveModular
    elif inference_algo == "contrastive_cmi":
        Inference = ContrastiveCMI
    elif inference_algo == "contrastive_mask":
        Inference = ContrastiveMask
    else:
        raise NotImplementedError

    if need_partition:
        inference = Inference(encoder, decoder, partition, params)
    else:
        inference = Inference(encoder, decoder, params)

    scripted_policy = get_scripted_policy(env, params)

    rl_algo = params.training_params.rl_algo
    is_task_learning = rl_algo in ["cem", "collocation", "sac"]
    start_step = get_start_step_from_model_loading(params)

    if rl_algo == "random":
        policy = RandomPolicy(params)
    elif rl_algo == "hippo":
        policy = HiPPO(params)
    elif rl_algo == "cem":
        policy = CEM(encoder, inference, reward_predictor, params)
    elif rl_algo == "collocation":
        policy = Collocation(encoder, inference, reward_predictor, params)
    elif rl_algo == "sac":
        policy = ModelBasedSAC(encoder, inference, reward_predictor, params, horizon, start_step)
    else:
        raise NotImplementedError

    # init replay buffer
    use_prioritized_buffer = replay_buffer_params.prioritized_buffer
    if use_prioritized_buffer:
        replay_buffer = PrioritizedReplayBuffer(params)
    else:
        replay_buffer = ReplayBuffer(params)

    total_step = training_params.total_step
    collect_transitions = training_params.collect_transitions
    num_inference_opt_steps = training_params.num_inference_opt_steps
    num_reward_opt_steps = training_params.num_reward_opt_steps
    num_policy_opt_steps = training_params.num_policy_opt_steps
    train_prop = training_params.train_prop
    if is_task_learning and num_policy_opt_steps:
        test_env = get_env(params, num_test_envs, render=False)

    # init saving
    writer = SummaryWriter(params.tb_logdir)
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    # init episode variables
    num_envs = params.env_params.num_envs
    # mult by num_envs necessary if restarting from before step bug fixed
    episode_num = start_step / horizon
    obs = env.reset()
    scripted_policy.reset(obs)

    done = np.zeros(num_envs, dtype=bool)
    success = np.zeros(num_envs, dtype=bool)
    episode_reward = np.zeros(num_envs)
    episode_step = np.zeros(num_envs)
    is_train = np.random.rand(num_envs) < train_prop
    is_demo = get_is_demo(start_step, params, num_envs)

    for step in range(start_step, total_step, num_envs):
        is_init_stage = step < training_params.init_step
        print("{}/{}, init_stage: {}".format(step, total_step, is_init_stage))

        # env interaction and transition saving
        if collect_transitions:
            # reset in the beginning of an episode
            if done.any():
                for i, done_i in enumerate(done):
                    if not done_i:
                        continue

                    if rl_algo == "hippo":
                        policy.reset(i)
                    scripted_policy.reset(obs, i)

                    if is_task_learning and not is_demo[i] and (episode_num * horizon) % training_params.log_freq == 0:
                        writer.add_scalar("policy_stat/episode_reward", episode_reward[i], episode_num)
                        writer.add_scalar("policy_stat/success", float(success[i]), episode_num)

                    is_train[i] = np.random.rand() < train_prop
                    is_demo[i] = get_is_demo(step, params)

                    episode_reward[i] = 0
                    episode_step[i] = 0
                    success[i] = False
                    episode_num += 1

            # get action (num_envs, action_dim)
            inference.eval()
            policy.eval()
            if is_init_stage:
                action = policy.act_randomly()
            else:
                action = policy.act(obs)
                if is_demo.any():
                    demo_action = scripted_policy.act(obs)
                    action[is_demo] = demo_action[is_demo]

            # (num_envs, obs_spec), (num_envs,), (num_envs,), [info] * num_envs
            next_obs, reward, done, info = env.step(action)

            if is_task_learning:
                success_step = np.array([info_i["success"] for info_i in info])
                success = success | success_step

            episode_reward += reward
            episode_step += 1

            # is_train: if the transition is training data or evaluation data for inference_cmi
            replay_buffer.add(obs, action, reward, next_obs, done, is_train, is_demo, info)

            obs = next_obs

        # training and logging
        if is_init_stage:
            continue

        # perform test episodes 
        if step % testing_params.policy_eval_freq == 0 and is_task_learning and num_policy_opt_steps:
            run_test_episodes(env=test_env, writer=writer, 
                              rl_algo=rl_algo, policy=policy, inference=inference,
                              num_test_envs=num_test_envs, 
                              num_test_eps=testing_params.num_test_eps,
                              curr_train_ep=episode_num)

        loss_details = {"inference": [],
                        "inference_eval": [],
                        "reward": [],
                        "reward_eval": [],
                        "policy": []}

        if num_inference_opt_steps:
            inference.train()
            inference.setup_annealing(step)
            for i_grad_step in range(num_inference_opt_steps):
                obs_batch, actions_batch, next_obses_batch, idxes_batch = \
                    replay_buffer.sample_inference(use_part="train")
                loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch)
                if use_prioritized_buffer:
                    replay_buffer.update_priorities(idxes_batch, loss_detail["priority"], "inference")
                loss_details["inference"].append(loss_detail)

            inference.eval()
            if step % testing_params.eval_freq == 0:
                obs_batch, actions_batch, next_obses_batch, _ = \
                    replay_buffer.sample_inference(use_part="eval")
                loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch, eval=True)
                loss_details["inference_eval"].append(loss_detail)

        if step > training_params.reduce_reward_opt_step:
            # can train reward predictor for fewer steps after it converges
            num_reward_opt_steps = num_inference_opt_steps = 0

        if reward_predictor is not None and num_reward_opt_steps and rew_pred_algo != "true":
            reward_predictor.train()
            reward_predictor.setup_annealing(step)
            for i_grad_step in range(num_reward_opt_steps):
                obs_batch, action_batch, next_obs_batch, reward_batch, idxes_batch = \
                    replay_buffer.sample_reward(use_part="train")
                loss_detail = reward_predictor.update(obs_batch, action_batch, next_obs_batch, reward_batch)
                if use_prioritized_buffer:
                    replay_buffer.update_priorities(idxes_batch, loss_detail["priority"], "reward")
                loss_details["reward"].append(loss_detail)

            reward_predictor.eval()
            if step % testing_params.eval_freq == 0:
                obs_batch, action_batch, next_obs_batch, reward_batch, idxes_batch = \
                    replay_buffer.sample_reward(use_part="eval")
                loss_detail = reward_predictor.update(obs_batch, action_batch, next_obs_batch, reward_batch, eval=True)
                loss_details["reward_eval"].append(loss_detail)

        if num_policy_opt_steps:
            assert is_task_learning and rl_algo == "sac"

            policy.train()
            inference.eval()
            reward_predictor.eval()
            policy.setup_annealing(step)

            for i_grad_step in range(num_policy_opt_steps):
                obses_batch, actions_batch, rewards_batch, next_obses_batch, dones_batch = replay_buffer.sample_policy()
                loss_detail = policy.update(obses_batch, actions_batch, rewards_batch, next_obses_batch, dones_batch)
                loss_details["policy"].append(loss_detail)
            policy.eval()

        # logging
        if step % training_params.log_freq == 0:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [to_float(dic[k]) for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name), np.mean(loss_values), step)

        if step % training_params.plot_freq == 0:
            if num_inference_opt_steps:
                plot_dynamics_mask(params, inference, writer, step)

            if num_reward_opt_steps:
                plot_reward_mask(params, reward_predictor, writer, step)

            if need_partition:
                plot_partition(params, partition, writer, step)

            if num_policy_opt_steps:
                plot_abstraction(params, policy, writer, step)

        if step % training_params.saving_freq == 0:
            if num_inference_opt_steps:
                inference.save(os.path.join(model_dir, "inference_{}".format(step)))
            if num_reward_opt_steps and rew_pred_algo != "true":
                reward_predictor.save(os.path.join(model_dir, "reward_{}".format(step)))
            if num_policy_opt_steps:
                policy.save(os.path.join(model_dir, "policy_{}".format(step)))
            if collect_transitions:
                replay_buffer.save()
            


if __name__ == "__main__":
    args = argparser()
    params = TrainingParams(training_params_fname="policy_params.json", train=True, seed=args.seed)
    train(params)
