import torch

from a_star.neighbor_utils import NeighborUtils
from model.feature_extractor.fe_factory import get_feature_extractor
from model.wh_model import WHModel
from utils.collections import ModelSolution


class NodeModel(WHModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = get_feature_extractor(
            kwargs["feature_extractor"]["name"],
            kwargs["feature_extractor"]["params"]
        )
        self.nu = None

    def extract_wh(self, x, s, t) -> ModelSolution:
        w = self.feature_extractor(x, s, t)[:, :, :, 0]
        w = self.activation(w)

        if self.nu is None:
            x_max, y_max = w.shape[1], w.shape[2]
            self.nu = NeighborUtils(x_max, y_max)

        return ModelSolution(
            weights=w,
            heuristic=self.nu.get_euclidean_heuristic(w.detach(), t)
        )
