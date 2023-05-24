import os
import torch
import numpy as np

from utils.utils import to_numpy
from utils.replay_buffer import take, assign


class ImaginationReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, params):
        self.params = params
        self.device = params.device

        training_params = params.training_params
        policy_params = params.policy_params
        imagination_replay_buffer_params = policy_params.imagination_replay_buffer_params

        self.capacity = capacity = imagination_replay_buffer_params.capacity

        self.saving_dir = params.imagination_replay_buffer_dir
        self.policy_batch_size = params.policy_params.batch_size

        # init data
        feature_dim = params.feature_dim
        goal_dim = params.goal_dim
        action_dim = params.action_dim
        num_td_steps = params.policy_params.num_td_steps

        self.features = np.empty((capacity, num_td_steps, feature_dim + goal_dim), dtype=np.float32)
        self.actions = np.empty((capacity, num_td_steps, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, num_td_steps, 1), dtype=np.float32)
        self.dones = np.empty((capacity, num_td_steps, 1), dtype=np.float32)
        self.next_features = np.empty((capacity, num_td_steps, feature_dim + goal_dim), dtype=np.float32)

        # init writing
        self.idx = 0
        self.last_save = 0
        self.full = False
        self.full_since_last_save = False

        # loading
        self.load(training_params.load_imagination_replay_buffer)

    def add(self, features, actions, rewards, dones, next_features):
        idxes = (np.arange(len(features)) + self.idx) % self.capacity

        self.features[idxes] = to_numpy(features)
        self.actions[idxes] = to_numpy(actions)
        self.rewards[idxes] = to_numpy(rewards)
        self.dones[idxes] = to_numpy(dones)
        self.next_features[idxes] = to_numpy(next_features)

        self.idx = self.idx + len(features)
        self.full = self.full or self.idx >= self.capacity

        if self.idx >= self.capacity:
            self.full_since_last_save = True

        self.idx = self.idx % self.capacity

    def sample(self, size=None):
        if size is None: 
            size = self.policy_batch_size
        idxes = np.random.randint(len(self), size=size)

        features = torch.tensor(self.features[idxes], device=self.device)
        actions = torch.tensor(self.actions[idxes], device=self.device)
        rewards = torch.tensor(self.rewards[idxes], device=self.device)
        dones = torch.tensor(self.dones[idxes], device=self.device)
        next_features = torch.tensor(self.next_features[idxes], device=self.device)

        return features, actions, rewards, dones, next_features

    def save(self):
        if self.full_since_last_save and self.idx >= self.last_save:
            # all samples in the replay buffer have not been saved before
            self.last_save = self.idx

        for chunk in os.listdir(self.saving_dir):
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            if (self.last_save < end or (end < start < self.last_save)) and (self.idx >= end or (self.idx < self.last_save < end)):
                chunk = os.path.join(self.saving_dir, chunk)
                print("remove", chunk)
                os.remove(chunk)

        chunk = "%d_%d.p" % (self.last_save, self.idx)
        path = os.path.join(self.saving_dir, chunk)

        payload = {"features": take(self.features, self.last_save, self.idx),
                   "actions": take(self.actions, self.last_save, self.idx),
                   "rewards": take(self.rewards, self.last_save, self.idx),
                   "dones": take(self.dones, self.last_save, self.idx),
                   "next_features": take(self.next_features, self.last_save, self.idx)}

        self.last_save = self.idx
        self.full_since_last_save = False
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
            assign(self.features, start, end, payload["features"])
            assign(self.actions, start, end, payload["actions"])
            assign(self.rewards, start, end, payload["rewards"])
            assign(self.dones, start, end, payload["dones"])
            assign(self.next_features, start, end, payload["next_features"])

            self.idx = end
            if end < start or end == self.capacity:
                self.full = True

            print("loaded", chunk)

        if len(chunks):
            # episode ends
            self.dones[self.idx - 1] = -1

        print("imagination replay buffer loaded from", save_dir)

    def __len__(self):
        return self.capacity if self.full else self.idx
