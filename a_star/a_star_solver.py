from abc import ABC, abstractmethod

import torch

from utils.collections import AStarSolution


class AStarSolver(ABC, torch.nn.Module):
    def forward(self, weights, heuristic, source, target) -> AStarSolution:
        batch, x_max, y_max = weights.shape
        if heuristic is not None:
            assert heuristic.shape == (batch, x_max, y_max), heuristic.shape
        assert source.shape == (batch, 2), source.shape
        assert target.shape == (batch, 2), target.shape

        sol = self.run(weights, heuristic, source, target)

        assert sol.paths.shape == (batch, x_max, y_max), sol.paths.shape
        assert sol.exp_nodes.shape == (batch, x_max, y_max), sol.exp_nodes.shape
        return sol

    @abstractmethod
    def run(self, weights, heuristic, source, target) -> AStarSolution:
        pass
