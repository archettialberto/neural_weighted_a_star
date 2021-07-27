from a_star.bb_a_star import BBAStarSolver
from a_star.hybrid_a_star import HybridAStarSolver
from a_star.neural_a_star import NeuralAStarSolver


def get_a_star_solver(name, params):
    solvers = {
        "bb": BBAStarSolver,
        "nn": NeuralAStarSolver,
        "hb": HybridAStarSolver,
    }
    if name not in solvers.keys():
        raise NotImplementedError("Solver " + name + " not implemented.")
    return solvers[name](**params)
