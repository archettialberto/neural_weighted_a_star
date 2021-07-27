from heapq import heappush, heappop
from itertools import count

import ray
import torch

from a_star.neighbor_utils import NeighborUtils
from utils.collections import AStarSolution


def a_star(nu: NeighborUtils, weights, heuristic, source, target) -> AStarSolution:
    source = (source[0].item(), source[1].item())
    target = (target[0].item(), target[1].item())
    x_max, y_max = nu.get_shape()
    if heuristic is None:
        def get_h(n, t, w=torch.min(weights)):
            nx, ny = n
            tx, ty = t
            dx = nx - tx if nx > tx else tx - nx
            dy = ny - ty if ny > ty else ty - ny
            d = dx if dx > dy else dy
            return d * w
    else:
        def get_h(n, t):
            nx, ny = n
            return heuristic[nx, ny]

    push = heappush
    pop = heappop
    c = count()
    queue = [(0, next(c), source, 0, None)]
    enqueued = {}
    explored = {}

    path = torch.zeros((x_max, y_max)).float()
    exp_nodes = torch.zeros((x_max, y_max)).float()

    while queue:
        _, __, curnode, dist, parent = pop(queue)
        exp_nodes[curnode[0], curnode[1]] = 1.0

        if curnode == target:
            path[curnode[0], curnode[1]] = 1.0
            node = parent
            while node is not None:
                x, y = node
                path[x, y] = 1.0
                node = explored[node]
            return AStarSolution(paths=path, exp_nodes=exp_nodes)

        if curnode in explored:
            if explored[curnode] is None:
                continue
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor in nu.get_neighbors(curnode):
            if neighbor is not None:
                ncost = dist + weights[neighbor[0], neighbor[1]]
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    if qcost <= ncost:
                        continue
                else:
                    h = get_h(neighbor, target)
                enqueued[neighbor] = ncost, h
                push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise Exception("Target %s not reachable from source %s." % (target, source))


@ray.remote
def a_star_ray(nu: NeighborUtils, weights, heuristic, source, target) -> AStarSolution:
    return a_star(nu, weights, heuristic, source, target)


def a_star_batch(weights, heuristic, source, target, use_ray=True) -> AStarSolution:
    device = weights.device
    batch, x_max, y_max = weights.shape
    nu = NeighborUtils(x_max, y_max)

    weights = weights.detach().cpu()
    if heuristic is not None:
        heuristic = heuristic.detach().cpu()
    source = source.detach().cpu()
    target = target.detach().cpu()

    paths = torch.zeros((batch, x_max, y_max)).float()
    exp_nodes = torch.zeros((batch, x_max, y_max)).float()

    results = []
    for b in range(batch):
        hb = None if heuristic is None else heuristic[b]
        wb = weights[b]
        if use_ray:
            res = a_star_ray.remote(nu, wb, hb, source[b], target[b])
        else:
            res = a_star(nu, wb, hb, source[b], target[b])
        results.append(res)
    if use_ray:
        results = ray.get(results)

    for b in range(batch):
        paths[b] = results[b].paths
        exp_nodes[b] = results[b].exp_nodes

    return AStarSolution(
        paths=paths.to(device),
        exp_nodes=exp_nodes.to(device),
    )
