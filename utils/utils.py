import os
import glob
import time
import json
import torch
import shutil
import random
import numpy as np

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import GymWrapper
from tianshou.env import SubprocVectorEnv
from env.physical_env import Physical
from env.chemical_env import Chemical
from utils.multiprocessing_env import SubprocVecEnv, SingleVecEnv


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TrainingParams(AttrDict):
    def __init__(self, training_params_fname="params.json", train=True, seed=None):
        config = json.load(open(training_params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        if seed: # boilerplate for slurm
            self.__dict__["seed"] = seed
        self.__dict__ = self._clean_dict(self.__dict__)

        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        training_params = self.training_params
        if training_params.load_inference is not None:
            training_params.load_inference = \
                os.path.join(repo_path, "interesting_models", training_params.load_inference)
        if training_params.load_policy is not None:
            training_params.load_policy = \
                load_latest(basepath=os.path.join(repo_path, "rslts", training_params.load_policy, "trained_models", "policy_*"))
                # os.path.join(repo_path, "interesting_models", training_params.load_policy)
        if training_params.load_reward_predictor is not None:
            training_params.load_reward_predictor = \
                load_latest(basepath=os.path.join(repo_path, "rslts", training_params.load_reward_predictor, "trained_models", "reward_*"))
        if training_params.load_replay_buffer is not None:
            training_params.load_replay_buffer = \
                os.path.join(repo_path, "replay_buffer", training_params.load_replay_buffer)
        if training_params.load_imagination_replay_buffer is not None:
            training_params.load_imagination_replay_buffer = \
                os.path.join(repo_path, "replay_buffer", training_params.load_imagination_replay_buffer)

        self.replay_buffer_dir = None
        self.imagination_replay_buffer_dir = None
        if train:
            # saving paths
            if training_params_fname == "policy_params.json":
                if hasattr(self, "sub_dirname"):
                    sub_dirname = self.sub_dirname
                elif training_params.rl_algo in ["cem", "collocation", "sac"]:
                    sub_dirname = "task"
                else:
                    sub_dirname = "dynamics"
            else:
                raise NotImplementedError

            info = self.info.replace(" ", "_")
            experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S") + f"_seed={self.seed}"
            self.rslts_dir = os.path.join(repo_path, "rslts", sub_dirname, experiment_dirname)
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, os.path.join(self.rslts_dir, "params.json"))

            self.replay_buffer_dir = os.path.join(repo_path, "replay_buffer", sub_dirname, experiment_dirname)
            if training_params.collect_transitions:
                os.makedirs(self.replay_buffer_dir)
            if training_params.rl_algo in ["sac"] and self.policy_params.sac_params.use_imagination:
                self.imagination_replay_buffer_dir = os.path.join(repo_path, "replay_buffer", sub_dirname, "imag_" + experiment_dirname)
                os.makedirs(self.imagination_replay_buffer_dir)
            
            self.tb_logdir = os.path.join(self.rslts_dir, "tensorboard")
            if training_params.load_tensorboard_log is not None:
                old_tb_logpath = load_latest(basepath=os.path.join(repo_path, "rslts", training_params.load_tensorboard_log, "tensorboard", "events*"),
                                             )
                old_tb_name = os.path.basename(old_tb_logpath)
                os.makedirs(self.tb_logdir)
                shutil.copyfile(old_tb_logpath, 
                                os.path.join(self.tb_logdir, old_tb_name))

        super(TrainingParams, self).__init__(self.__dict__)

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = self._clean_dict(v)
            _dict[k] = v
        return AttrDict(_dict)

def load_latest(basepath):
    # loads most recent available path following regex
    matched_paths = glob.glob(basepath) # these are full paths
    return max(matched_paths, key=os.path.getctime)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def to_float(x):
    if isinstance(x, (float, int)):
        return x
    else:
        return x.item()

def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def preprocess_obs(obs, params, ignore_goal=False):
    """
    filter unused obs keys, convert to np.float32 / np.uint8, resize images if applicable
    """
    def to_type(ndarray, type):
        if ndarray.dtype != type:
            ndarray = ndarray.astype(type)
        return ndarray

    obs_spec = getattr(params, "obs_spec", obs)
    new_obs = {}

    keys = params.obs_keys
    if not ignore_goal:
        keys = keys + params.goal_keys

    for k in keys:
        val = obs[k]
        val_spec = obs_spec[k]
        if isinstance(val_spec, np.ndarray):
            if val_spec.ndim == 0:
                if hasattr(params, "obs_spec"):
                    val = to_type(val[..., None], np.float32)
                else:
                    val = to_type(val, np.float32)
            elif val_spec.ndim == 1:
                val = to_type(val, np.float32)
            elif val_spec.ndim == 3:
                num_channel = val.shape[2]
                if num_channel == 1:
                    env_params = params.env_params
                    assert "Causal" in env_params.env_name
                    val = to_type(val, np.float32)
                elif num_channel == 3:
                    val = to_type(val, np.uint8)
                else:
                    raise NotImplementedError
                val = val.transpose((2, 0, 1))                  # c, h, w
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        new_obs[k] = val
    return new_obs


def postprocess_obs(obs):
    # convert images to float32 and normalize to [0, 1]
    new_obs = {}
    for k, val in obs.items():
        if val.dtype == np.uint8:
            val = val.astype(np.float32) / 255
        new_obs[k] = val
    return new_obs


def update_obs_act_spec(env, params):
    """
    get act_dim and obs_spec from env and add to params
    """
    params.continuous_state = params.continuous_action = not isinstance(env, (Physical, Chemical))

    params.action_dim = env.action_dim
    params.obs_spec = preprocess_obs(env.observation_spec(), params)

    env_params = params.env_params
    env_name = env_params.env_name
    manipulation_env_params = env_params.manipulation_env_params
    if "ToolUse" in env_name:
        params.normalization_range = manipulation_env_params.tool_use_env_params.normalization_range
    elif "Causal" in env_name:
        params.normalization_range = manipulation_env_params.causal_env_params.normalization_range

    params.action_spec = env.action_spec if params.continuous_action else None

    if params.continuous_state:
        params.obs_dims = None
        obs_delta_range = env.obs_delta_range()
        obs_delta_low, obs_delta_high = {k: v[0] for k, v in obs_delta_range.items()}, {k: v[1] for k, v in obs_delta_range.items()}
        obs_delta_low = preprocess_obs(obs_delta_low, params, ignore_goal=True)
        obs_delta_high = preprocess_obs(obs_delta_high, params, ignore_goal=True)
        params.obs_delta_range = {k: [torch.from_numpy(obs_delta_low[k]).to(params.device), torch.from_numpy(obs_delta_high[k]).to(params.device)]
                                  for k in obs_delta_low}
    else:
        params.obs_dims = env.observation_dims()

    if params.training_params.object_level_obs:
        params.continuous_state = False


def get_single_env(params, render=False, wrap_gym=False, test=False):
    env_params = params.env_params
    env_name = env_params.env_name
    if env_name == "Physical":
        env = Physical(params)
    elif env_name == "Chemical":
        env = Chemical(params)
    else:
        manipulation_env_params = env_params.manipulation_env_params
        if "ToolUse" in env_name:
            env_kwargs = manipulation_env_params.tool_use_env_params
            if env_name != "ToolUseSeries":
                env_kwargs.pop("terminal_state")
        elif "Causal" in env_name:
            env_kwargs = manipulation_env_params.causal_env_params
        else:
            raise ValueError("Unknown env_name: {}".format(env_name))
        env = suite.make(env_name=env_params.env_name,
                         robots=manipulation_env_params.robots,
                         controller_configs=
                         load_controller_config(default_controller=manipulation_env_params.controller_name),
                         gripper_types=manipulation_env_params.gripper_types,
                         has_renderer=render,
                         has_offscreen_renderer=manipulation_env_params.use_camera_obs,
                         use_camera_obs=manipulation_env_params.use_camera_obs,
                         camera_names=manipulation_env_params.camera_names,
                         camera_heights=manipulation_env_params.camera_heights,
                         camera_widths=manipulation_env_params.camera_widths,
                         camera_depths=manipulation_env_params.camera_depths,
                         ignore_done=False,
                         control_freq=manipulation_env_params.control_freq,
                         reward_scale=manipulation_env_params.reward_scale,
                         sparse_reward=manipulation_env_params.sparse_reward,
                         num_markers=manipulation_env_params.num_markers,
                         marker_x_range=manipulation_env_params.marker_x_range,
                         marker_y_range=manipulation_env_params.marker_y_range,
                         **env_kwargs)

    if wrap_gym:
        keys = params.obs_keys + params.goal_keys
        env = GymWrapper(env, keys=keys)
        env.is_async = False # hacking for tianshou, fix later in baseenv
    return env


def get_subproc_env(params, wrap_gym, test):
    def get_single_env_wrapper():
        return get_single_env(params=params, wrap_gym=wrap_gym, test=test)
    return get_single_env_wrapper

def get_env(params, num_envs, render=False, 
            wrap_gym=False, use_tianshou=False,
            test=False):
    if render:
        assert num_envs == 1
    if num_envs == 1:
        return SingleVecEnv(get_single_env(params, render, wrap_gym, test))
    else:
        assert params.env_params.env_name not in ["Physical", "Chemical"]
        if not use_tianshou:
            return SubprocVecEnv([get_subproc_env(params, wrap_gym, test) for _ in range(num_envs)])
        else:
            # tianshou subprocvecenv
            return SubprocVectorEnv([get_subproc_env(params=params, wrap_gym=wrap_gym, test=test) for _ in range(num_envs)])


def get_start_step_from_model_loading(params):
    """
    if model-based policy is loaded, return its training step;
    elif inference is loaded, return its training step;
    else, return 0
    """
    task_learning = params.training_params.rl_algo in ["cem", "collocation", "sac"]
    load_inference = params.training_params.load_inference
    load_policy = params.training_params.load_policy
    if load_policy is not None and os.path.exists(load_policy):
        model_name = load_policy.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    elif load_inference is not None and os.path.exists(load_inference) and not task_learning:
        model_name = load_inference.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    else:
        start_step = 0
    return start_step
