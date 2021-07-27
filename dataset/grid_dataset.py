import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class GridDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, path, prefix, normalize_input=False):
        super().__init__()
        self.path = Path(path)
        if prefix not in ["train", "val", "test"]:
            raise ValueError(prefix)
        self.prefix = prefix

        self.images = self.load_from_file("images")
        if normalize_input:
            self.images = self.normalize(self.images)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def load_from_file(self, name):
        filename = self.prefix + "_" + name + ".npy"
        path = os.path.join(self.path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError("File " + str(path) + " does not exist.")
        array = torch.from_numpy(np.load(path)).float()
        return array

    @staticmethod
    def normalize(i):
        assert len(i.shape) == 4
        batch, rows, cols, channels = i.shape
        i = i.reshape((i.shape[0], -1))
        i -= i.min(1, keepdim=True)[0]
        i /= i.max(1, keepdim=True)[0]
        i = i.reshape((batch, rows, cols, channels))
        return i


class WarcraftDataset(GridDataset):
    def __init__(self, path, prefix, normalize_input=True):
        super().__init__(path, prefix, normalize_input)
        self.weights = self.load_from_file("weights")
        self.sources = self.load_from_file("sources").long()
        self.targets = self.load_from_file("targets").long()
        self.paths = self.load_from_file("paths")
        self.exp_nodes = self.load_from_file("exp_nodes")
        self.opt_exp_nodes = self.load_from_file("opt_exp_nodes")
        self.heuristic = self.load_from_file("heuristic")

    def __len__(self):
        return self.sources.shape[0] * self.sources.shape[1] * self.sources.shape[2]

    def __getitem__(self, index):
        t_per_i = self.sources.shape[1]
        s_per_t = self.sources.shape[2]
        b = index // (t_per_i * s_per_t)
        i = index % (t_per_i * s_per_t) // t_per_i
        j = index % (t_per_i * s_per_t) % t_per_i
        source = self.sources[b, i, j]
        path = self.paths[b, i, j]
        exp_nodes = self.exp_nodes[b, i, j]
        opt_exp_nodes = self.opt_exp_nodes[b, i, j]

        image = self.images[b]
        weights = self.weights[b]
        target = self.targets[b, i]
        heuristic = self.heuristic[b, i]

        image_st = torch.zeros((image.shape[0], image.shape[1], 5))
        image_st[:, :, 0:3] = image

        dx = image.shape[0] // path.shape[0]
        dy = image.shape[1] // path.shape[1]
        tx = target[0] * dx
        ty = target[1] * dy
        image_st[tx:tx + dx, ty:ty + dy, 3] = 1.0
        sx = source[0] * dx
        sy = source[1] * dy
        image_st[sx:sx + dx, sy:sy + dy, 4] = 1.0

        return (
            image,
            image_st,
            weights,
            heuristic,
            source,
            target,
            path,
            exp_nodes,
            opt_exp_nodes
        )
