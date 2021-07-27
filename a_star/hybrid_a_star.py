import torch.nn as nn

from a_star.a_star_solver import AStarSolver
from a_star.bb_a_star import BBAStar
from a_star.neural_a_star import NeuralAStar
from utils.collections import AStarSolution


class HybridAStar(nn.Module):
    def __init__(self, lambda_val, tau_val, detach):
        super().__init__()
        self.bb_solver = BBAStar(lambda_val)
        self.nn_solver = NeuralAStar(tau_val)
        self.detach = detach

    def forward(self, weights, heuristic, source, target) -> AStarSolution:
        bb_sol = self.bb_solver(weights, source, target)
        w = weights.detach() if self.detach else weights
        nn_sol = self.nn_solver(w, heuristic, source, target)
        return AStarSolution(
            paths=bb_sol.paths,
            exp_nodes=nn_sol.exp_nodes
        )


class HybridAStarSolver(AStarSolver):
    def __init__(self, lambda_val, tau_val, detach=True):
        super().__init__()
        self.solver = HybridAStar(lambda_val, tau_val, detach=detach)

    def run(self, weights, heuristic, source, target) -> AStarSolution:
        return self.solver(weights, heuristic, source, target)
