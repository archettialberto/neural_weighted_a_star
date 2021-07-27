import torch

from a_star.neighbor_utils import NeighborUtils
from model.feature_extractor.fe_factory import get_feature_extractor
from model.wh_model import WHModel
from utils.collections import ModelSolution


class DNWA(WHModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_extractor = get_feature_extractor(
            kwargs["weights_extractor"]["name"],
            kwargs["weights_extractor"]["params"]
        )
        self.h_extractor = get_feature_extractor(
            kwargs["heuristic_extractor"]["name"],
            kwargs["heuristic_extractor"]["params"]
        )
        self.nu = None
        self.epsilon = kwargs["epsilon"]
        self._range = kwargs["range"]
        assert self.epsilon >= 0.0, self.epsilon

    def set_range(self, x):
        max, min = self._range
        return x * (max - min) + min

    def extract_wh(self, x, s, t, rand_epsilon=False) -> ModelSolution:

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            w = self.w_extractor(x, s, t)[:, :, :, 0]

        with torch.cuda.stream(s2):
            h = self.h_extractor(x, s, t)[:, :, :, 0]

        torch.cuda.synchronize()

        w = self.activation(w)
        w = self.set_range(w)

        if rand_epsilon:
            epsilon = torch.rand((w.shape[0], 1, 1)).to(w.device) * 9.0
        else:
            epsilon = self.epsilon

        if self.nu is None:
            x_max, y_max = w.shape[1], w.shape[2]
            self.nu = NeighborUtils(x_max, y_max)


        h = self.nu.get_euclidean_heuristic(w.detach(), t) * (1.0 + self.activation(h) * epsilon)

        return ModelSolution(
            weights=w,
            heuristic=h
        )
