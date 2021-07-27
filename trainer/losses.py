from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossWrapper:
    def __init__(self, args):
        self.args = args
        self.losses = self.__build_losses()

    def __build_losses(self):
        losses = []
        loss_class_dict = {
            "swloss": MSEWeightLoss,
            "shloss": MSEHeuristicLoss,
            "hlossp": HammingLossPaths,
            "hlosse": HammingLossExpNodes,
        }
        for loss in self.args.values():
            if loss.name not in loss_class_dict.keys():
                raise NotImplementedError("Loss " + loss.name + " not implemented.")
            losses.append(loss_class_dict[loss.name](loss.name, **loss.params))
        return losses


class LossComponent(nn.Module, ABC):
    def __init__(self, name, coefficient=1.0):
        super().__init__()
        self.name = name
        self.coefficient = coefficient

    def get_name(self):
        return self.name

    def forward(self, **kwargs):
        return self.coefficient * self.eval_loss(**kwargs)

    @abstractmethod
    def eval_loss(self, **kwargs):
        pass


class MSEWeightLoss(LossComponent):
    def eval_loss(self, **kwargs):
        true_weights = kwargs["true_weights"]
        pred_weights = kwargs["pred_weights"]
        return F.mse_loss(true_weights, pred_weights)


class MSEHeuristicLoss(LossComponent):
    def eval_loss(self, **kwargs):
        true_heuristic = kwargs["true_heuristic"]
        pred_heuristic = kwargs["pred_heuristic"]
        return F.mse_loss(true_heuristic, pred_heuristic)


class HammingLossPaths(LossComponent):
    def eval_loss(self, **kwargs):
        pred_tensor = kwargs["pred_paths"]
        true_paths = kwargs["true_paths"]
        assert (true_paths >= 0).all() and (true_paths <= 1).all()
        assert (pred_tensor >= 0).all() and (pred_tensor <= 1).all()
        return torch.mean(
            torch.sum(true_paths * (1. - pred_tensor) + pred_tensor * (1. - true_paths), dim=(-1, -2))
        )


class HammingLossExpNodes(LossComponent):
    def eval_loss(self, **kwargs):
        pred_tensor = kwargs["pred_exp_nodes"]
        true_paths = kwargs["true_paths"]
        assert (true_paths >= 0).all() and (true_paths <= 1).all()
        assert (pred_tensor >= 0).all() and (pred_tensor <= 1).all()
        return torch.mean(
            torch.sum(true_paths * (1. - pred_tensor) + pred_tensor * (1. - true_paths), dim=(-1, -2))
        )
