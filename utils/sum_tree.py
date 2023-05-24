# Adapted from https://github.com/rlcode/per/blob/master/SumTree.py

import numpy as np
# SumTree
# a binary tree data structure where the parent’s value is the sum of its children


class SumTreeOrig:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, dataIdx

    def init_tree(self, ps):
        assert (self.tree == 0).all() and self.write == 0
        assert len(ps) <= self.capacity
        self.tree[self.capacity - 1:self.capacity - 1 + len(ps)] = ps
        self.write = len(ps)

        last_idx = len(ps) - 1 + self.capacity - 1
        last_parent = (last_idx - 1) // 2
        for i in reversed(range(last_parent + 1)):
            left = 2 * i + 1
            right = left + 1
            self.tree[i] = self.tree[left] + self.tree[right]

        assert self.total() == ps.sum()


def mask_occured(a, first=True):
    """
    check if each element already occurs on axis=-1
    first:
        if True, check from idx 0 to -1
        if False, check from idx -1 to 0
    """
    sidx = a.argsort(axis=-1, kind='stable')
    b = np.take_along_axis(a, sidx, axis=-1)
    if first:
        mask = np.concatenate([np.zeros((*a.shape[:-1], 1), dtype=bool), b[..., :-1] == b[..., 1:]], axis=-1)
    else:
        mask = np.concatenate([b[..., :-1] == b[..., 1:], np.zeros((*a.shape[:-1], 1), dtype=bool)], axis=-1)
    out = np.empty(a.shape, dtype=bool)
    np.put_along_axis(out, sidx, mask, axis=-1)
    return out


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.full = False

        self.num_updates = 0
        self.valid_freq = 10000

    # update to the root node
    def _propagate(self, idxes, changes):
        # idxes, changes: (batch_size,)
        zeros = np.zeros_like(changes)
        while True:
            idxes = (idxes - 1) // 2
            np.add.at(self.tree, idxes, changes)

            finish_props = (idxes <= 0)
            changes = np.where(finish_props, zeros, changes)

            if finish_props.all():
                return
 
    # find sample on leaf node
    def _retrieve(self, idxes, values):
        # idxes, value: (batch_size,)
        tree_len = self.capacity if self.full else self.write
        tree_len += self.capacity - 1
        while True:
            lefts = 2 * idxes + 1
            rights = lefts + 1

            found_idxes = lefts >= tree_len
            if found_idxes.all():
                return idxes

            modified_lefts = np.where(found_idxes, idxes, lefts)
            left_values = self.tree[modified_lefts]
            le_lefts = values <= left_values
            idxes = np.where(le_lefts, modified_lefts, rights)
            values = np.where(le_lefts, values, values - left_values)

    def total(self):
        # return: scalar
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        if isinstance(p, np.ndarray):
            assert p.ndim == 1
            idx = np.arange(self.write, self.write + len(p)) % self.capacity
            idx += self.capacity - 1
        elif isinstance(p, float):
            idx = self.write + self.capacity - 1
            idx = np.array([idx])
            p = np.array([p])
        else:
            raise NotImplementedError

        self.update(idx, p)

        self.write += len(p)
        if self.write >= self.capacity:
            self.full = True
            self.write -= self.capacity

    # update priority
    def update(self, idxes, ps):
        # idxes, ps: (batch_size,)

        changes = ps - self.tree[idxes]

        # for each idx in the batch, whether the idx already occurs in the reverse order
        m = mask_occured(idxes, first=False)
        changes[m] = 0

        # m is computed using the reverse idx order
        # because if there are duplicate idxes in the following assignment,
        # only the value corresponding to the last idx is assigned
        self.tree[idxes] = ps
        self._propagate(idxes, changes)

        self.num_updates += 1
        if self.num_updates % self.valid_freq == 0:
            # recompute all non-leaf nodes to maintain numerical stability
            self.init_tree()

    # get priority and sample
    def get(self, values):
        # values: (batch_size,)
        idxes = self._retrieve(np.zeros_like(values, dtype=np.int32), values)
        dataIdxes = idxes - self.capacity + 1
        return idxes, dataIdxes

    def init_tree(self, ps=None):
        if ps is None:
            # recompute all non-leaf nodes to maintain numerical stability
            last_idx = 2 * self.capacity - 2
        else:
            assert (self.tree == 0).all() and self.write == 0
            assert len(ps) <= self.capacity
            self.tree[self.capacity - 1:self.capacity - 1 + len(ps)] = ps
            self.write = len(ps) % self.capacity
            self.full = len(ps) == self.capacity

            last_idx = len(ps) - 1 + self.capacity - 1

        last_parent = (last_idx - 1) // 2
        for i in reversed(range(last_parent + 1)):
            left = 2 * i + 1
            right = left + 1
            self.tree[i] = self.tree[left] + self.tree[right]

        if ps is None:
            assert (np.abs(self.total() - self.tree[self.capacity - 1:].sum()) < 1e-8).all()
        else:
            assert (self.total() == ps.sum()).all()


class BatchSumTree:
    def __init__(self, num_trees, capacity, batch_size):
        self.num_trees = num_trees
        self.capacity = capacity
        self.batch_size = batch_size

        self.trees = np.zeros((num_trees, 2 * capacity - 1), dtype=np.float64)
        self.write = 0
        self.full = False

        self.tree_idxes = np.tile(np.arange(num_trees)[:, None], (1, batch_size))

        self.num_updates = 0
        self.valid_freq = 10000

    # update to the root node
    def _propagate(self, idxes, changes):
        # idxes, changes: (num_trees, batch_size)
        zeros = np.zeros_like(changes)
        while True:
            idxes = (idxes - 1) // 2

            # similar to self.trees[self.tree_idxes, idxes] += changes but handles repeated inxes
            np.add.at(self.trees, (self.tree_idxes, idxes), changes)

            finish_props = (idxes <= 0)
            changes = np.where(finish_props, zeros, changes)

            if finish_props.all():
                return
 
    # find sample on leaf node
    def _retrieve(self, idxes, values):
        # idxes, value: (num_trees, batch_size)
        tree_len = self.capacity if self.full else self.write
        tree_len += self.capacity - 1

        while True:
            lefts = 2 * idxes + 1
            rights = lefts + 1

            found_idxes = lefts >= tree_len
            if found_idxes.all():
                return idxes

            modified_lefts = np.where(found_idxes, idxes, lefts)
            left_values = self.trees[self.tree_idxes, modified_lefts]
            le_lefts = values <= left_values
            idxes = np.where(le_lefts, modified_lefts, rights)
            values = np.where(le_lefts, values, values - left_values)

    def total(self):
        return self.trees[:, 0]

    # store priority and sample
    def add(self, p):
        if isinstance(p, np.ndarray):
            assert p.shape == (self.batch_size,)
            idx = np.arange(self.write, self.write + self.batch_size) % self.capacity
            idx += self.capacity - 1

            p = np.tile(p, (self.num_trees, 1))
            idx = np.tile(idx, (self.num_trees, 1))
        else:
            raise NotImplementedError

        self.update(idx, p)

        self.write += self.batch_size
        if self.write >= self.capacity:
            self.full = True
            self.write -= self.capacity

    # update priority
    def update(self, idxes, ps):
        # idxes, ps: (num_trees, batch_size)

        changes = ps - self.trees[self.tree_idxes, idxes]

        # for each idx in the batch, whether the idx already occurs in the reverse order
        m = mask_occured(idxes, first=False)
        changes[m] = 0

        # m is computed using the reverse idx order
        # because if there are duplicate idxes in the following assignment,
        # only the value corresponding to the last idx is assigned
        self.trees[self.tree_idxes, idxes] = ps
        self._propagate(idxes, changes)

        self.num_updates += 1
        if self.num_updates % self.valid_freq == 0:
            # recompute all non-leaf nodes to maintain numerical stability
            self.init_trees()

    # get priority and sample
    def get(self, values):
        # values: (num_trees, batch_size)
        idxes = self._retrieve(np.zeros((self.num_trees, self.batch_size), dtype=np.int32), values)
        dataIdxes = idxes - self.capacity + 1
        return idxes, dataIdxes

    def init_trees(self, ps=None):
        if ps is None:
            # recompute all non-leaf nodes to maintain numerical stability
            last_idx = 2 * self.capacity - 2
        else:
            assert (self.trees == 0).all() and self.write == 0
            assert len(ps) <= self.capacity
            self.trees[:, self.capacity - 1:self.capacity - 1 + len(ps)] = ps
            self.write = len(ps) % self.capacity
            self.full = len(ps) == self.capacity

            last_idx = len(ps) - 1 + self.capacity - 1

        last_parent = (last_idx - 1) // 2
        for i in reversed(range(last_parent + 1)):
            left = 2 * i + 1
            right = left + 1
            self.trees[:, i] = self.trees[:, left] + self.trees[:, right]

        if ps is None:
            m = np.abs(self.total() - self.trees[:, self.capacity - 1:].sum(axis=-1)) >= 1e-8
            if m.any():
                print("total", self.total()[m])
                print("truth", self.trees[:, self.capacity - 1:].sum(axis=-1)[m])
                print("diff", np.abs(self.total() - self.trees[:, self.capacity - 1:].sum(axis=-1))[m])
            assert (np.abs(self.total() - self.trees[:, self.capacity - 1:].sum(axis=-1)) < 1e-8).all()
        else:
            assert (self.total() == ps.sum()).all()