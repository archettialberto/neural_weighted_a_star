import torch
import torch.nn as nn
import torch.nn.functional as F

from a_star.a_star_solver import AStarSolver
from a_star.ray_a_star import a_star_batch
from utils.collections import AStarSolution


class BBAStar(nn.Module):
    def __init__(self, lambda_val):
        super().__init__()

        class BBAStarFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weights, source, target, lambda_val, paths):
                weights = weights.detach()
                ctx.save_for_backward(weights, source, target, paths)
                ctx.lambda_val = lambda_val
                return paths

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = grad_output.detach()
                weights, source, target, paths = ctx.saved_tensors
                assert weights.shape == grad_output.shape, (weights.shape, grad_output.shape)
                lambda_val = ctx.lambda_val
                weights_prime = F.relu(weights + lambda_val * grad_output)
                paths_prime = a_star_batch(weights_prime, None, source, target).paths
                gradient = -(paths - paths_prime) / lambda_val
                return gradient, None, None, None, None

        self.lambda_val = lambda_val
        self.solver = BBAStarFunction.apply

    def forward(self, weights, source, target) -> AStarSolution:
        ray_sol = a_star_batch(weights, None, source, target)
        bb_sol = self.solver(weights, source, target, self.lambda_val, ray_sol.paths)
        return AStarSolution(
            paths=bb_sol,
            exp_nodes=ray_sol.exp_nodes
        )


class BBAStarSolver(AStarSolver):
    def __init__(self, lambda_val):
        super().__init__()
        self.solver = BBAStar(lambda_val)

    def run(self, weights, heuristic, source, target) -> AStarSolution:
        return self.solver(weights, source, target)
