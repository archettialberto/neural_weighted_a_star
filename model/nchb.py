import itertools
import math

import torch

from model.node import NodeModel
from utils.collections import ModelSolution


class NodeCEModel(NodeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eucl_matrix = None

    def __build_eucl_matrix(self):
        x_max_base, y_max_base = self.nu.x_max * 2 - 1, self.nu.y_max * 2 - 1
        base = torch.zeros((x_max_base, y_max_base)).float()
        tbx, tby = self.nu.x_max - 1, self.nu.y_max - 1
        for x, y in itertools.product(range(x_max_base), range(y_max_base)):
            base[x, y] = math.sqrt((x - tbx) ** 2 + (y - tby) ** 2)
        return base

    def get_eucl_heuristic(self, weights, targets):
        weights = weights.detach()
        batch, x_max, y_max = weights.shape
        assert x_max == self.nu.x_max, x_max
        assert y_max == self.nu.y_max, y_max
        x_max_base, y_max_base = x_max * 2 - 1, y_max * 2 - 1
        h = torch.zeros((batch, x_max, y_max)).float().to(weights.device)
        eucl = self.eucl_matrix.to(weights.device)
        for b in range(batch):
            tx, ty = targets[b]
            h[b] = eucl[x_max - tx - 1: x_max_base - tx, y_max - ty - 1: y_max_base - ty] * .001
        return h

    def extract_wh(self, x, s, t) -> ModelSolution:
        w, h = super().extract_wh(x, s, t)

        if self.eucl_matrix is None:
            self.eucl_matrix = self.__build_eucl_matrix()

        return ModelSolution(
            weights=w,
            heuristic=self.nu.get_euclidean_heuristic(torch.ones_like(w).float().to(w.device), t) +
                      self.get_eucl_heuristic(w.detach(), t)
        )
