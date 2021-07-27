import itertools

import torch


class NeighborUtils:
    def __init__(self, x_max: int, y_max: int):
        self.x_max = x_max
        self.y_max = y_max
        self.neighbors_dict = self.__build_neighbors_dict()
        self.steps_matrix = self.__build_steps_matrix()

    def __iter__(self) -> (int, int):
        for x, y in itertools.product(range(self.x_max), range(self.y_max)):
            yield x, y

    def get_shape(self) -> (int, int):
        return self.x_max, self.y_max

    def get_neighbors(self, n: (int, int)) -> (int, int):
        neighbors = self.neighbors_dict[n]
        for m in neighbors:
            yield m

    def get_euclidean_heuristic(self, weights, targets):
        weights = weights.detach()
        batch, x_max, y_max = weights.shape
        assert x_max == self.x_max, x_max
        assert y_max == self.y_max, y_max
        assert targets.shape[0] == batch
        assert targets.shape[1] == 2
        x_max_base, y_max_base = x_max * 2 - 1, y_max * 2 - 1
        h = torch.zeros((batch, x_max, y_max)).float().to(weights.device)
        steps = self.steps_matrix.to(weights.device)
        for b in range(batch):
            w_min = torch.min(weights[b])
            tx, ty = targets[b]
            mask = steps[x_max - tx - 1: x_max_base - tx, y_max - ty - 1: y_max_base - ty]
            h[b] = mask * w_min
        return h

    def __build_neighbors_dict(self):
        neighbors_dict = dict()
        for n in self.__iter__():
            nx, ny = n
            neighbors_dict[n] = []
            for (dx, dy) in itertools.product((-1, 0, 1), (-1, 0, 1)):
                mx, my = nx + dx, ny + dy
                if 0 <= mx < self.x_max and 0 <= my < self.y_max and (dx, dy) != (0, 0):
                    neighbors_dict[n].append((mx, my))
                elif (dx, dy) != (0, 0):
                    neighbors_dict[n].append(None)
        return neighbors_dict

    def __build_steps_matrix(self):
        x_max_base, y_max_base = self.x_max * 2 - 1, self.y_max * 2 - 1
        base = torch.zeros((x_max_base, y_max_base)).float()
        tbx, tby = self.x_max - 1, self.y_max - 1
        for x, y in itertools.product(range(x_max_base), range(y_max_base)):
            dx = x - tbx if x > tbx else tbx - x
            dy = y - tby if y > tby else tby - y
            d = dx if dx > dy else dy
            base[x, y] = d
        return base
