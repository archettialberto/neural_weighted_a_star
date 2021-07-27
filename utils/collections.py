from collections import namedtuple

AStarSolution = namedtuple(
    "AStarSolution",
    [
        "paths",
        "exp_nodes",
    ]
)

ModelSolution = namedtuple(
    "ModelSolution",
    [
        "weights",
        "heuristic",
    ]
)

LogData = namedtuple(
    "LogData",
    [
        "image",
        "source",
        "target",
        "path",
        "weights",
        "heuristic",
        "exp_nodes",
    ]
)
