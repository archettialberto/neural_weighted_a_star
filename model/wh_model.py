from abc import ABC, abstractmethod

import torch

from utils.collections import ModelSolution


class WHModel(ABC, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._act = kwargs["activation"]

    def forward(self, x, s, t) -> ModelSolution:
        batch, _, _, in_channels = x.shape
        assert t.shape == (batch, 2), t.shape

        sol = self.extract_wh(x, s, t)
        w = sol.weights
        h = sol.heuristic

        _, x_max, y_max = w.shape
        assert w.shape[0] == batch, w.shape[0]
        assert (batch, x_max, y_max) == h.shape, h.shape

        return sol

    def activation(self, x):
        if self._act == "relu":
            act = torch.relu
        elif self._act == "sigm":
            act = torch.sigmoid
        elif self._act == "norm":
            act = self.normalize
        else:
            raise NotImplementedError("Activation " + str(self._act) + " not implemented.")
        return act(x)

    @abstractmethod
    def extract_wh(self, x, s, t) -> ModelSolution:
        pass

    @staticmethod
    def batchwise_min(a):
        min = torch.min(a.detach(), dim=1, keepdim=True).values
        min = torch.min(min, dim=2, keepdim=True).values
        return min

    @staticmethod
    def batchwise_max(a):
        max = torch.max(a.detach(), dim=1, keepdim=True).values
        max = torch.max(max, dim=2, keepdim=True).values
        return max

    @staticmethod
    def normalize(a):
        min = WHModel.batchwise_min(a)
        max = WHModel.batchwise_max(a)
        return (a - min) / (max - min)
