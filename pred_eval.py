import os
import time
import pickle
import numpy as np

import torch
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)


from model.inference_mlp import InferenceMLP
from model.inference_gnn import InferenceGNN
from model.inference_reg import InferenceReg
from model.inference_nps import InferenceNPS
from model.inference_cmi import InferenceCMI

from model.contrastive_mod import ContrastiveModular
from model.contrastive_cmi import ContrastiveCMI

from model.random_policy import RandomPolicy
from model.hippo import HiPPO

from model.encoder import IdentityEncoder
from model.decoder import Decoder

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, to_numpy
from utils.scripted_policy import get_scripted_policy


if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    load_paths = ["causal_cmi_2022_11_26_14_19_17",
                  "causal_contrastive_cmi_sq_sa_1en6_128_2022_11_24_14_25_42"]
    load_paths = ["tooluse_contrastive_cmi_sq_sa_1en6_128_2022_12_23_00_04_15",
                  "tooluse_more_pick_contrastive_cmi_no_PER_seed_0_2023_01_09_00_45_26"]
    save_fname = "test.p"
    num_epi = 30
    num_pred_steps = 1
    batch_size = 25
    model_step = None           # 1800000, None

    cuda_id = 1
    seed = 0

    # ======================================= params to overwrite begins ======================================= #
    horizon = 400

    manipulation_train = True
    manipulation_test_scale = 100

    # contrastive params
    num_pred_samples = 256
    num_pred_iters = 1
    pred_iter_sigma_init = 0.1
    pred_iter_sigma_shrink = 0.5

    # hippo params
    skill_names = ["lift", "push"]
    skill_probs = [0.5, 0.5]
    skill_names = ["lift", "push", "pick_place", "hook"]
    skill_probs = [0.6, 0.1, 0.15, 0.15]
    # skill_names = ["hook"]
    # skill_probs = [1.0]
    disturbance_prob = 0.2

    # cem params
    std_scale = 0.3
    num_horizon_steps = 3
    num_iters = 5
    num_candidates = 64
    num_top_candidates = 32
    # ======================================== params to overwrite ends ======================================== #

    device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")

    # get test transitions
    load_path = os.path.join(repo_path, "interesting_models", load_paths[0])

    params_fname = os.path.join(load_path, "params.json")
    params = TrainingParams(training_params_fname=params_fname, train=False)

    params.seed = seed
    env_params = params.env_params
    env_params.num_envs = 1
    save_fname = env_params.env_name + "_" + save_fname
    env_params.manipulation_env_params.causal_env_params.horizon = horizon
    env_params.manipulation_env_params.tool_use_env_params.horizon = horizon
    set_seed_everywhere(seed)
    params.device = device

    env = get_env(params, 1)

    update_obs_act_spec(env, params)

    if params.continuous_action:
        hippo_params = params.policy_params.hippo_params
        hippo_params.skill_names = skill_names
        hippo_params.skill_probs = skill_probs
        hippo_params.skill_params.disturbance_prob = disturbance_prob
        policy = HiPPO(params)
    else:
        policy = get_scripted_policy(env, params)

    obs_buffer = []
    action_buffer = []
    next_obs_buffer = []
    if num_pred_steps != 1:
        raise NotImplementedError

    for e in range(num_epi):
        print("{}/{}".format(e + 1, num_epi))
        done = False

        obs = env.reset()

        step = 0

        while not done:
            step += 1
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)
            done = done[0]

            if not done:
                obs_buffer.append(obs)
                action_buffer.append(action)
                next_obs_buffer.append(next_obs)

            obs = next_obs

    obs_buffer = {k: np.concatenate([obs[k] for obs in obs_buffer], axis=0)
                  for k in obs_buffer[0].keys()}
    action_buffer = np.concatenate(action_buffer, axis=0)
    next_obs_buffer = {k: np.concatenate([next_obs[k] for next_obs in next_obs_buffer], axis=0)
                       for k in next_obs_buffer[0].keys()}

    num_test_data = action_buffer.shape[0]
    print("number of test transitions:", num_test_data)

    # eval prediction
    performances = {"obs": obs_buffer,
                    "action": action_buffer,
                    "next_obs": next_obs_buffer}
    for path in load_paths:
        print("test", path)
        load_path = os.path.join(repo_path, "interesting_models", path)

        params_fname = os.path.join(load_path, "params.json")
        params = TrainingParams(training_params_fname=params_fname, train=False)

        if model_step is None:
            inference_fnames = [fname for fname in os.listdir(load_path) if "inference" in fname]
            assert len(inference_fnames) == 1
            inference_fname = inference_fnames[0]
            inference_fname = os.path.join(load_path, inference_fname)
        else:
            inference_fname = os.path.join(load_path, "inference_" + str(model_step))
            if not os.path.exists(inference_fname):
                print("warning:", inference_fname, "doesn't exist")
                continue

        params.seed = seed
        params.training_params.load_inference = inference_fname
        env_params = params.env_params

        set_seed_everywhere(seed)
        params.device = device
        training_params = params.training_params

        update_obs_act_spec(env, params)
        encoder = IdentityEncoder(params)
        encoder.manipulation_train = manipulation_train
        encoder.manipulation_test_scale = manipulation_test_scale
        params.feature_dim = encoder.feature_dim

        decoder = Decoder(params) if params.decoder_params.use_decoder else None

        params.contrastive_params.num_pred_samples = num_pred_samples
        params.contrastive_params.num_pred_iters = num_pred_iters
        params.contrastive_params.pred_iter_sigma_init = pred_iter_sigma_init
        params.contrastive_params.pred_iter_sigma_shrink = pred_iter_sigma_shrink

        inference_algo = params.training_params.inference_algo
        if inference_algo == "mlp":
            Inference = InferenceMLP
        elif inference_algo == "gnn":
            Inference = InferenceGNN
        elif inference_algo == "reg":
            Inference = InferenceReg
        elif inference_algo == "nps":
            Inference = InferenceNPS
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

        feature_buffer = []
        next_feature_buffer = []
        pred_next_feature_buffer = []
        start = time.time()
        for i in range(0, num_test_data, batch_size):
            print("{}/{}".format(i + batch_size, num_test_data))
            obs = {k: v[i:i + batch_size] for k, v in obs_buffer.items()}
            if params.continuous_action:
                action = action_buffer[i:i + batch_size, None, :]
            else:
                action = action_buffer[i:i + batch_size, None, None]
            next_obs = {k: v[i:i + batch_size, None, :] for k, v in next_obs_buffer.items()
                        if v.ndim > 1}
            feature, next_features, pred_next_features = inference.eval_prediction(obs, action, next_obs)

            feature_buffer.append(to_numpy(feature))
            next_feature_buffer.append(to_numpy(next_features[:, 0]))
            pred_next_feature_buffer.append(to_numpy(pred_next_features[:, 0]))
        print("pred takes", time.time() - start)

        feature = np.concatenate(feature_buffer, axis=0)                        # (num_test_data, feature_dim)
        next_feature = np.concatenate(next_feature_buffer, axis=0)              # (num_test_data, feature_dim)
        pred_next_feature = np.concatenate(pred_next_feature_buffer, axis=0)    # (num_test_data, feature_dim)

        performances[path] = {"feature": feature,
                              "next_feature": next_feature,
                              "pred_next_feature": pred_next_feature}

        with open(save_fname, "wb") as f:
            pickle.dump(performances, f)
