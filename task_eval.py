import os
import pickle
import numpy as np

import torch
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)


from model.encoder import make_encoder
from model.decoder import Decoder

from model.inference_mlp import InferenceMLP
from model.inference_gnn import InferenceGNN
from model.inference_reg import InferenceReg
from model.inference_nps import InferenceNPS
from model.inference_cmi import InferenceCMI

from model.contrastive_mod import ContrastiveModular
from model.contrastive_cmi import ContrastiveCMI

# data collection policy for dynamics training
from model.random_policy import RandomPolicy
from model.hippo import HiPPO

from model.reward_predictor import RewardPredictorDense, TrueReward
from model.reward_predictor_cmi import RewardPredictorCMI
from model.reward_predictor_contrastive import RewardPredictorContrastive

# policy to solve downstream tasks
from model.cem import CEM
from model.collocation import Collocation
from model.model_based_sac import ModelBasedSAC

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, to_numpy
from utils.scripted_policy import get_scripted_policy


if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    load_paths = ["reward_test_cmi_pick_2022_12_21_01_31_57"]

    save_fname = "test_task_ood.p"
    num_epi = 1
    cuda_id = 2

    use_scripted_policy = True

    manipulation_train = False
    manipulation_test_scale = 10

    seed = 0

    # ======================================= params to overwrite begins ======================================= #
    # contrastive params
    num_pred_samples = 16384
    num_pred_iters = 1
    pred_iter_sigma_init = 0.33
    pred_iter_sigma_shrink = 0.5

    # cem params
    std_scale = 0.3
    num_horizon_steps = 3
    num_iters = 5
    num_candidates = 64
    num_top_candidates = 32
    # ======================================== params to overwrite ends ======================================== #

    device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")

    performances = {}
    for path in load_paths:
        load_path = os.path.join(repo_path, "interesting_models", path)

        params_fname = os.path.join(load_path, "params.json")
        params = TrainingParams(training_params_fname=params_fname, train=False)

        policy_fname = None
        if not use_scripted_policy:
            policy_fnames = [fname for fname in os.listdir(load_path) if "policy" in fname]
            assert len(policy_fnames) == 1
            policy_fname = policy_fnames[0]
            policy_fname = os.path.join(load_path, policy_fname)

        reward_predictor_fname = None
        reward_predictor_fnames = [fname for fname in os.listdir(load_path) if "reward" in fname]
        assert len(reward_predictor_fnames) == 1
        reward_predictor_fname = reward_predictor_fnames[0]
        reward_predictor_fname = os.path.join(load_path, reward_predictor_fname)

        params.seed = seed
        params.training_params.load_policy = policy_fname
        params.training_params.load_reward_predictor = reward_predictor_fname
        env_params = params.env_params
        env_params.num_envs = 1

        params.reward_predictor_params.contrastive_params.num_pred_samples = num_pred_samples
        params.reward_predictor_params.contrastive_params.num_pred_iters = num_pred_iters
        params.reward_predictor_params.contrastive_params.pred_iter_sigma_init = pred_iter_sigma_init
        params.reward_predictor_params.contrastive_params.pred_iter_sigma_shrink = pred_iter_sigma_shrink

        params.policy_params.imagination_replay_buffer_params.saving_freq = 0

        set_seed_everywhere(seed)
        params.device = device
        training_params = params.training_params

        env = get_env(params, env_params.num_envs, render=False)
        update_obs_act_spec(env, params)

        encoder = make_encoder(params)
        params.feature_dim = encoder.feature_dim
        params.goal_dim = encoder.goal_dim
        encoder.manipulation_train = manipulation_train
        encoder.manipulation_test_scale = manipulation_test_scale

        decoder = Decoder(params) if params.decoder_params.use_decoder else None

        inference_algo = training_params.inference_algo
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
        else:
            raise NotImplementedError
        inference = Inference(encoder, decoder, params)
        inference.eval()

        reward_predictor = None
        rew_pred_algo = params.training_params.rew_pred_algo
        if rew_pred_algo == "dense":
            RewardPredictor = RewardPredictorDense
        elif rew_pred_algo == "cmi":
            RewardPredictor = RewardPredictorCMI
        elif rew_pred_algo == "contrastive":
            RewardPredictor = RewardPredictorContrastive
        elif rew_pred_algo == "true":
            RewardPredictor = TrueReward
        else:
            raise NotImplementedError
        reward_predictor = RewardPredictor(encoder, params)

        rl_algo = params.training_params.rl_algo
        if rl_algo == "random":
            policy = RandomPolicy(params)
        elif rl_algo == "hippo":
            policy = HiPPO(params)
        elif rl_algo == "cem":
            policy = CEM(encoder, inference, reward_predictor, params)
        elif rl_algo == "collocation":
            policy = Collocation(encoder, inference, reward_predictor, params)
        elif rl_algo == "sac":
            policy = ModelBasedSAC(encoder, inference, reward_predictor, params)
        else:
            raise NotImplementedError
        scripted_policy = get_scripted_policy(env, params)

        episode_rewards = []
        successes = []

        for e in range(num_epi):
            print(path, "{}/{}".format(e + 1, num_epi))
            done = False
            success = False
            episode_reward = 0

            obs = env.reset()
            scripted_policy.reset(obs)

            while not done:
                if use_scripted_policy:
                    action = scripted_policy.act(obs)
                else:
                    action = policy.act(obs, deterministic=True)

                next_obs, reward, done, info = env.step(action)

                reward = reward[0]
                done = done[0]
                info = info[0]

                pred_reward = reward_predictor.pred_reward(obs, action, next_obs)
                pred_reward = to_numpy(pred_reward.squeeze())
                print("reward {:.4f}, {:.4f}, {:.4f}".format(reward, pred_reward, np.abs(reward - pred_reward)))

                episode_reward += reward
                success = success | info["success"]

                obs = next_obs

            print("episode_reward:", episode_reward, "success:", success)
            episode_rewards.append(episode_reward)
            successes.append(success)

        print("episode_rewards", np.mean(episode_rewards), np.std(episode_rewards) / np.sqrt(num_epi))
        print("successes", np.mean(successes))

        performances[path] = {"episode_reward": np.array(episode_rewards),
                              "success": np.array(successes)}

        with open(save_fname, "wb") as f:
            pickle.dump(performances, f)
