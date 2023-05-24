import os
import torch
import numpy as np

from utils.utils import preprocess_obs, postprocess_obs, to_numpy
from utils.sum_tree import SumTree, BatchSumTree


def take(array, start, end):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    return array[idxes]


def assign(array, start, end, value):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    array[idxes] = value


class ReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.continuous_action = params.continuous_action

        training_params = params.training_params
        replay_buffer_params = training_params.replay_buffer_params

        self.capacity = capacity = replay_buffer_params.capacity
        self.saving_dir = params.replay_buffer_dir

        self.use_contrastive = "contrastive" in params.training_params.inference_algo
        if self.use_contrastive:
            self.inference_batch_size = params.contrastive_params.batch_size
            self.num_inference_pred_steps = params.contrastive_params.num_pred_steps
        else:
            self.inference_batch_size = params.inference_params.batch_size
            self.num_inference_pred_steps = params.inference_params.num_pred_steps

        self.reward_batch_size = params.reward_predictor_params.batch_size
        self.reward_eval_demo_only = params.reward_predictor_params.eval_demo_only

        self.policy_batch_size = params.policy_params.batch_size
        self.num_policy_td_steps = params.policy_params.num_td_steps
        self.policy_use_demo = params.policy_params.use_demo

        # init data
        obs_spec = params.obs_spec
        action_dim = params.action_dim
        self.obses = {k: np.empty((capacity, *v.shape), dtype=v.dtype) for k, v in obs_spec.items()}
        if self.continuous_action:
            self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.is_trains = np.empty((capacity, 1), dtype=bool)
        self.is_demos = np.empty((capacity, 1), dtype=bool)

        # init writing
        self.idx = 0
        self.last_save = 0
        self.full = False

        # loading
        self.load(training_params.load_replay_buffer)

        # cache for vecenv
        self.num_envs = params.env_params.num_envs
        self.temp_buffer = [[] for _ in range(self.num_envs)]

        self.init_cache()

    def init_cache(self):
        # cache for faster query
        self.inference_pred_idxes_base = np.tile(np.arange(self.num_inference_pred_steps), (self.inference_batch_size, 1))
        self.policy_idxes_base = np.tile(np.arange(self.num_policy_td_steps), (self.policy_batch_size, 1))

    def add(self, obs, action, reward, next_obs, done, is_train, is_demo, info):
        for i in range(self.num_envs):
            obs_i = {key: val[i] for key, val in obs.items()}
            self.temp_buffer[i].append([obs_i, action[i], reward[i], done[i], is_train[i], is_demo[i]])
            if done[i]:
                for obs_, action_, reward_, done_, is_train_, is_demo_ in self.temp_buffer[i]:
                    self._add(obs_, action_, reward_, done_, is_train_, is_demo_)
                final_obs = info[i]["obs"]
                # use done = -1 as a special indicator that the added obs is the last obs in the episode
                self._add(final_obs, action_, 0, -1, is_train_, is_demo_)
                self.temp_buffer[i] = []

    def _add(self, obs, action, reward, done, is_train, is_demo):
        obs = preprocess_obs(obs, self.params)
        for k in obs.keys():
            self.obses[k][self.idx] = obs[k]

        if self.continuous_action and action.dtype != np.float32:
            action = action.astype(np.float32)
        elif not self.continuous_action:
            action = np.int64(action)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.is_trains[self.idx], is_train)
        np.copyto(self.is_demos[self.idx], is_demo)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def valid_idx(self, idx, num_steps, type, use_part="all"):
        if use_part != "all":
            is_train = self.is_trains[idx]
            if use_part == "train" and not is_train:
                return False
            if use_part == "eval" and is_train:
                return False

        if type == "policy":
            if not self.policy_use_demo and self.is_demos[idx:idx + num_steps].any():
                return False

        if type == "reward" and use_part == "eval":
            if self.reward_eval_demo_only and not self.is_demos[idx]:
                return False

        cross_episode = (self.dones[idx:idx + num_steps] == -1).any()

        # self.idx - 1 is the latest data point
        # idx is the first data point to use, idx + num_steps is the last data point to use (both inclusive)
        cross_newest_data = (idx < self.idx) and (idx + num_steps >= self.idx)

        return not (cross_episode or cross_newest_data)

    def sample_idx(self, type, use_part="all"):
        if type == "inference":
            num_steps = self.num_inference_pred_steps
            batch_size = self.inference_batch_size
        elif type == "reward":
            num_steps = 1
            batch_size = self.reward_batch_size
        elif type == "policy":
            num_steps = self.num_policy_td_steps
            batch_size = self.policy_batch_size
        else:
            raise NotImplementedError

        idxes = []
        for _ in range(batch_size):
            while True:
                idx = np.random.randint(len(self) - num_steps)
                if self.valid_idx(idx, num_steps, type, use_part):
                    idxes.append(idx)
                    break

        return np.array(idxes)

    def get_idxes_base(self, type, use_part):
        if type == "inference":
            return self.inference_pred_idxes_base
        elif type == "reward":
            return 0
        elif type == "policy":
            return self.policy_idxes_base
        else:
            raise NotImplementedError

    def construct_transition(self, idxes, type, use_part):
        pred_idxes_base = self.get_idxes_base(type, use_part)

        if type == "policy":
            idxes = idxes[:, None]
            obs_idxes = pred_idxes_base + idxes                         # (batch_size, num_td_steps)
        else:
            obs_idxes = idxes                                           # (batch_size,)
            if type == "inference":
                idxes = idxes[:, None]

        next_obs_idxes = pred_idxes_base + idxes + 1                    # (batch_size,) or (batch_size, num_pred_steps)
        act_rew_done_idxes = next_obs_idxes - 1                         # (batch_size,) or (batch_size, num_pred_steps)

        obses = postprocess_obs({k: v[obs_idxes] for k, v in self.obses.items()})
        obses = {k: torch.tensor(v, device=self.device) for k, v in obses.items()}

        actions = torch.tensor(self.actions[act_rew_done_idxes],
                               dtype=torch.float32 if self.continuous_action else torch.int64, device=self.device)

        next_obses = postprocess_obs({k: v[next_obs_idxes] for k, v in self.obses.items()})
        next_obses = {k: torch.tensor(v, device=self.device) for k, v in next_obses.items()}

        rewards = dones = None
        if type in ["reward", "policy"]:
            rewards = torch.tensor(self.rewards[act_rew_done_idxes], dtype=torch.float32, device=self.device)

        if type == "policy":
            dones = torch.tensor(self.dones[act_rew_done_idxes], dtype=torch.float32, device=self.device)

        return obses, actions, rewards, next_obses, dones

    def sample(self, type, use_part="all"):
        """
        Sample training data for inference model
        return:
            obses: (batch_size, obs_spec)
            actions: (batch_size, action_dim) or (batch_size, num_steps, action_dim)
            next_obses: (batch_size, action_dim) or (batch_size, num_steps, obs_spec)
        """
        idxes = self.sample_idx(type, use_part)
        obses, actions, rewards, next_obses, dones = self.construct_transition(idxes, type, use_part)
        return obses, actions, rewards, next_obses, dones, idxes

    def sample_inference(self, use_part="all"):
        obses, actions, _, next_obses, _, idxes = self.sample("inference", use_part)
        return obses, actions, next_obses, idxes

    def sample_reward(self, use_part="all"):
        obses, actions, rewards, next_obses, _, idxes = self.sample("reward", use_part)
        return obses, actions, next_obses, rewards, idxes

    def sample_policy(self, num_steps=1):
        obses, actions, rewards, next_obses, dones, _ = self.sample("policy", use_part="all")
        return obses, actions, rewards, next_obses, dones

    def save(self):
        assert self.idx != self.last_save

        for chunk in os.listdir(self.saving_dir):
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            if (self.last_save < end or (end < start < self.last_save)) and (self.idx >= end or (self.idx < self.last_save < end)):
                chunk = os.path.join(self.saving_dir, chunk)
                print("remove", chunk)
                os.remove(chunk)

        chunk = "%d_%d.p" % (self.last_save, self.idx)
        path = os.path.join(self.saving_dir, chunk)

        payload = {"obses": {k: take(v, self.last_save, self.idx) for k, v in self.obses.items()},
                   "actions": take(self.actions, self.last_save, self.idx),
                   "rewards": take(self.rewards, self.last_save, self.idx),
                   "dones": take(self.dones, self.last_save, self.idx),
                   "is_trains": take(self.is_trains, self.last_save, self.idx)}

        self.last_save = self.idx
        torch.save(payload, path)
        print("saved", chunk)

    def load(self, save_dir):
        if save_dir is None or not os.path.isdir(save_dir):
            return

        chunks = [os.path.join(save_dir, chunk) for chunk in os.listdir(save_dir)]
        chunks.sort(key=os.path.getctime)
        for chunk in chunks:
            chunk_fname = os.path.split(chunk)[1]
            start, end = [int(x) for x in chunk_fname.split(".")[0].split("_")]
            payload = torch.load(chunk)
            for k in self.obses:
                assign(self.obses[k], start, end, payload["obses"][k])
            assign(self.actions, start, end, payload["actions"])
            assign(self.rewards, start, end, payload["rewards"])
            assign(self.dones, start, end, payload["dones"])
            assign(self.is_trains, start, end, payload["is_trains"])

            self.idx = end
            if end < start or end == self.capacity:
                self.full = True

            print("loaded", chunk)

        if len(chunks):
            # episode ends
            self.dones[self.idx - 1] = -1

        print("replay buffer loaded from", save_dir)

    def __len__(self):
        return self.capacity if self.full else self.idx


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, params):
        replay_buffer_params = params.training_params.replay_buffer_params
        capacity = replay_buffer_params.capacity

        self.feature_dim = params.feature_dim

        self.use_contrastive = "contrastive" in params.training_params.inference_algo
        if self.use_contrastive:
            self.inference_batch_size = params.contrastive_params.batch_size
        else:
            self.inference_batch_size = params.inference_params.batch_size

        self.inference_train_tree = BatchSumTree(self.feature_dim, capacity, self.inference_batch_size)
        self.reward_train_tree = SumTree(capacity)

        self.inference_alpha = replay_buffer_params.inference_alpha
        self.reward_alpha = replay_buffer_params.reward_alpha
        self.max_priority = 1

        self.inference_to_add_buffer = []
        self.reward_to_add_buffer = []

        super(PrioritizedReplayBuffer, self).__init__(params)

    def init_cache(self):
        super(PrioritizedReplayBuffer, self).init_cache()

        # cache for faster query
        feature_dim = self.feature_dim
        inference_batch_size = self.inference_batch_size

        self.inference_train_pred_idxes_base = np.tile(np.arange(self.num_inference_pred_steps),
                                                       (feature_dim * inference_batch_size, 1))
        self.inference_eval_pred_idxes_base = np.tile(np.arange(self.num_inference_pred_steps),
                                                      (inference_batch_size, 1))

    def _add(self, obs, action, reward, done, is_train, is_demo):
        super(PrioritizedReplayBuffer, self)._add(obs, action, reward, done, is_train, is_demo)
        self._add_to_tree(done, is_train)

    def _add_to_tree(self, done, is_train):
        inference_prob = reward_prob = self.max_priority
        # TODO: update priorities of previous transitions too
        inference_prob = reward_prob = is_train & (done != -1)

        if self.params.training_params.num_inference_opt_steps:
            self.inference_to_add_buffer.append(inference_prob)
            if len(self.inference_to_add_buffer) == self.inference_batch_size:
                self.inference_train_tree.add(np.array(self.inference_to_add_buffer))
                self.inference_to_add_buffer = []

        if self.params.training_params.num_reward_opt_steps:
            self.reward_to_add_buffer.append(reward_prob)
            if len(self.reward_to_add_buffer) == self.reward_batch_size:
                self.reward_train_tree.add(np.array(self.reward_to_add_buffer))
                self.reward_to_add_buffer = []

    def update_priorities(self, idxes, probs, type):
        if isinstance(probs, torch.Tensor):
            probs = to_numpy(probs)

        EPS = 0.02

        if type == "reward":
            # idxes, probs: (batch_size,)
            probs = np.clip(probs ** self.reward_alpha, EPS, self.max_priority)
            tree = self.reward_train_tree
        elif type == "inference":
            # idxes, probs: (feature_dim, batch_size)
            probs = np.clip(probs ** self.inference_alpha, EPS, self.max_priority)
            tree = self.inference_train_tree
        else:
            raise NotImplementedError

        tree.update(idxes, probs)

    def get_idxes_base(self, type, use_part):
        if type == "inference":
            if use_part == "train":
                return self.inference_train_pred_idxes_base
            else:
                return self.inference_eval_pred_idxes_base
        elif type == "reward":
            return 0
        elif type == "policy":
            return self.policy_idxes_base
        else:
            raise NotImplementedError

    def sample(self, type, use_part="all"):
        if type == "inference":
            num_steps = self.num_inference_pred_steps
            batch_size = self.inference_batch_size
        elif type == "reward":
            num_steps = 1
            batch_size = self.reward_batch_size
        elif type == "policy":
            num_steps = self.num_policy_td_steps
            batch_size = self.policy_batch_size
        else:
            raise NotImplementedError

        tree_idxes, data_idxes = self.sample_idx(batch_size, num_steps, type, use_part)
        obses, actions, rewards, next_obses, dones = self.construct_transition(data_idxes, type, use_part)

        if type == "inference" and use_part == "train":
            feature_dim = self.feature_dim

            obses = {k: v.view(feature_dim, batch_size, -1)
                     for k, v in obses.items()}
            actions = actions.view(feature_dim, batch_size, num_steps, -1)
            next_obses = {k: v.view(feature_dim, batch_size, num_steps, -1)
                          for k, v in next_obses.items()}

        return obses, actions, rewards, next_obses, dones, tree_idxes

    def sample_idx_from_tree(self, tree, batch_size, num_steps):
        segment = tree.total() / batch_size       # scalar or (feature_dim,)
        if not self.full:
            # - self.max_priority * num_steps to avoid infinite loop of sampling the newly added sample
            segment -= self.max_priority * num_steps / batch_size

        if isinstance(tree, SumTree):
            s = np.random.uniform(size=batch_size) + np.arange(batch_size)
            s = s * segment                                             # (batch_size,)
        elif isinstance(tree, BatchSumTree):
            s = np.random.uniform(size=(self.feature_dim, batch_size)) + np.arange(batch_size)
            s = s * segment[:, None]                                    # (feature_dim, batch_size)
        else:
            raise NotImplementedError

        tree_idxes, data_idxes = tree.get(s)                            # (batch_size,) or (feature_dim, batch_size)

        if isinstance(tree, BatchSumTree):
            data_idxes = np.array(data_idxes).flatten()                 # (feature_dim * batch_size)

        return tree_idxes, data_idxes

    def sample_idx(self, batch_size, num_steps, type, use_part="all"):
        if type == "policy" or use_part == "eval":
            idxes = super(PrioritizedReplayBuffer, self).sample_idx(type, use_part)
            return None, idxes

        assert use_part != "all"
        if type == "reward":
            tree = self.reward_train_tree
        elif type == "inference":
            tree = self.inference_train_tree
        else:
            raise NotImplementedError

        return self.sample_idx_from_tree(tree, batch_size, num_steps)

    def load(self, save_dir):
        super(PrioritizedReplayBuffer, self).load(save_dir)
        num_data = self.capacity if self.full else self.idx
        dones = self.dones[:num_data, 0]
        is_trains = self.is_trains[:num_data, 0]

        valid_mask = np.array([(dones[i:i + self.num_inference_pred_steps] != -1).all()
                               for i in range(num_data)])
        train_priorities = valid_mask * is_trains * self.max_priority
        self.inference_train_tree.init_trees(train_priorities)

        valid_mask = dones != -1
        train_priorities = valid_mask * is_trains * self.max_priority
        self.reward_train_tree.init_tree(train_priorities)
