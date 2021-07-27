from abc import ABC, abstractmethod
from collections import OrderedDict

import cv2
import numpy as np
import torch
from scipy import ndimage

from a_star.ray_a_star import a_star_batch, a_star
from a_star.neighbor_utils import NeighborUtils
from utils.region_detector import RegionDetector

WARCRAFT = False


class MetricsWrapper:
    def __init__(self, use_ray=True):
        self.use_ray = use_ray
        self.metrics = [
            CostRatio(),
            CostRatioAccuracy(0),
            CostRatioAccuracy(2),
            CostRatioAccuracy(4),
            NodeExpansions("pred"),
            NodeExpansions("eucl"),
            NodeExpansions("true"),
            NodeExpansions("true_opt"),
            HammingDistance("pred"),
            HammingDistance("eucl"),
            HeuristicAdmissibility(),
            HeuristicOptimality("cr", use_ray=use_ray),
            HeuristicOptimality("exp", use_ray=use_ray),
            SummaryMetric("weights", "mean"),
            SummaryMetric("weights", "var"),
            SummaryMetric("heuristic", "mean"),
            SummaryMetric("heuristic", "var"),
            PerfectMatchAccuracy(),
        ]

    def compute_metrics(self, metrics, **kwargs):
        pred_weights = kwargs["pred_weights"]
        source = kwargs["source"]
        target = kwargs["target"]
        pred_heuristic = kwargs["pred_heuristic"]
        a_star_sol_pred = a_star_batch(pred_weights, pred_heuristic, source, target, use_ray=self.use_ray)
        a_star_sol_eucl = a_star_batch(pred_weights, None, source, target, use_ray=self.use_ray)
        kwargs["pred_paths"] = a_star_sol_pred.paths
        kwargs["pred_exp_nodes"] = a_star_sol_pred.exp_nodes
        kwargs["eucl_paths"] = a_star_sol_eucl.paths
        kwargs["eucl_exp_nodes"] = a_star_sol_eucl.exp_nodes
        for m in self.metrics:
            metrics[m.name] = m.eval(**kwargs)
        return metrics

    @staticmethod
    def get_stats_with_tqdm(metrics, meters, running=False):
        for key, metric in metrics.items():
            meters[key].update(metric)

        running_tqdm, final_tqdm = OrderedDict(), OrderedDict()
        for key, meter in meters.items():
            running_tqdm[key] = meter.val
            final_tqdm[key] = meter.avg
        if running:
            output = OrderedDict({
                'running_tqdm': running_tqdm,
                'final_tqdm': final_tqdm,
                'stats': metrics
            })
        else:
            output = OrderedDict({
                'final_tqdm': final_tqdm,
                'stats': final_tqdm
            })
        return output


class Metric(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def eval(self, **kwargs):
        pass


class NodeExpansions(Metric):
    def __init__(self, target):
        super().__init__(target + "_node_expansions")
        self.target = target

    def eval(self, **kwargs):
        exp_nodes = kwargs[self.target + "_exp_nodes"]
        return torch.mean(torch.sum(exp_nodes, dim=(-1, -2)))


class HeuristicAdmissibility(Metric):
    def __init__(self):
        super().__init__("heuristic_admissibility")

    def eval(self, **kwargs):
        pred_heuristic = kwargs["pred_heuristic"]
        batch, x_max, y_max = pred_heuristic.shape
        num_checks = 1
        indexes = np.random.choice(x_max * y_max, num_checks, replace=False)
        adm = torch.zeros(num_checks).to(pred_heuristic.device)
        pred_weights = kwargs["pred_weights"]
        nu = NeighborUtils(x_max, y_max)
        target = kwargs["target"]
        for i in range(num_checks):
            nx, ny = indexes[i] // y_max, indexes[i] % y_max
            path = a_star(nu,
                          pred_weights[0].detach(),
                          None,
                          (torch.tensor(nx).long(), torch.tensor(ny).long()),
                          target[0]).paths.to(pred_heuristic.device)
            path[nx, ny] = 0.0
            adm[i] = pred_heuristic[0, nx, ny] <= torch.sum(path * pred_weights[0]) + 1e-5
        return torch.mean(adm.float())


class HeuristicOptimality(Metric):
    def __init__(self, target, use_ray=True):
        assert target in ['cr', 'exp'], target
        super().__init__("heuristic_" + target + "_optimality")
        self.target = target
        self.use_ray = use_ray
        self.kernel = ndimage.generate_binary_structure(2, 2)

    def eval(self, **kwargs):
        true_weights = kwargs["true_weights"]
        pred_weights = kwargs["pred_weights"]
        pred_heuristic = kwargs["pred_heuristic"]
        target = kwargs["target"]
        batch, x_max, y_max = true_weights.shape
        device = true_weights.device

        tries = 1

        sources = torch.zeros((batch * tries, 2)).long().to(device)
        targets = torch.zeros((batch * tries, 2)).long().to(device)
        tw = torch.zeros((batch * tries, x_max, y_max)).float().to(device)
        ph = torch.zeros((batch * tries, x_max, y_max)).float().to(device)
        pw = torch.zeros((batch * tries, x_max, y_max)).float().to(device)

        rd = RegionDetector(x_max, y_max, sources_per_target=tries, f=0.25)

        for b in range(batch):

            if WARCRAFT:
                region = None
                for r in rd.regions.keys():
                    if rd.is_in_region(target[b], [r]):
                        region = r
                assert region is not None
                opp_region = rd.opposite_regions[region]

            for i in range(tries):
                idx = b * tries + i
                tw[idx] = true_weights[b]
                pw[idx] = pred_weights[b]
                ph[idx] = pred_heuristic[b]
                targets[idx] = target[b]

                source_torch = torch.zeros(2).long()

                if WARCRAFT:
                    source_np = rd.sample_from_region(opp_region)
                    source_torch[0] = source_np[0]
                    source_torch[1] = source_np[1]
                else:
                    true_weights_numpy = true_weights[b].detach().cpu().numpy()
                    walls = np.equal(true_weights_numpy, 25.0).astype(np.uint8)
                    l_num, labels = cv2.connectedComponents(1 - walls, connectivity=8)
                    component_label = labels[targets[idx][0].item(), targets[idx][1].item()]
                    steps = np.zeros_like(true_weights_numpy).astype(np.uint8)
                    steps[targets[idx][0].item(), targets[idx][1].item()] = 1
                    for _ in range(12):
                        _steps = ndimage.binary_dilation(steps, structure=self.kernel)
                        steps = (steps + _steps) * (1 - walls)
                    other_regions = (labels != component_label).astype(np.uint8)
                    valid_region = other_regions + walls + steps
                    x_s, y_s = np.where(valid_region == 0)
                    rnd_idx = np.random.randint(0, x_s.shape[0])
                    source_torch[0] = x_s[rnd_idx]
                    source_torch[1] = y_s[rnd_idx]

                sources[idx] = source_torch.to(device)

        if self.target == 'cr':
            tp = a_star_batch(tw, None, sources, targets, use_ray=self.use_ray).paths
            pp = a_star_batch(pw, ph, sources, targets, use_ray=self.use_ray).paths
            pred_paths_costs = torch.sum(pp * tw, dim=(-1, -2))
            true_paths_costs = torch.sum(tp * tw, dim=(-1, -2))
            return torch.mean(pred_paths_costs / true_paths_costs)
        else:
            pexp = a_star_batch(pw, ph, sources, targets, use_ray=self.use_ray).exp_nodes
            return torch.mean(torch.sum(pexp, dim=(-1, -2)))


class HammingDistance(Metric):
    def __init__(self, target):
        super().__init__(target + "_hamming_distance")
        self.target = target

    def eval(self, **kwargs):
        a = kwargs[self.target + "_paths"]
        b = kwargs["true_paths"]
        return torch.mean(torch.sum(a * (1.0 - b) + b * (1.0 - a), dim=(-1, -2)))


class PerfectMatchAccuracy(Metric):
    def __init__(self):
        super().__init__("perfect_match_accuracy")

    def eval(self, **kwargs):
        a = kwargs["pred_paths"]
        b = kwargs["true_paths"]
        diff = torch.sum(torch.abs(a - b), dim=(-1, -2))
        return torch.mean((diff == 0).float())


class CostRatio(Metric):
    def __init__(self):
        super().__init__("cost_ratio")

    def eval(self, **kwargs):
        pred_paths = kwargs["pred_paths"]
        true_paths = kwargs["true_paths"]
        true_weights = kwargs["true_weights"]
        pred_paths_costs = torch.sum(pred_paths * true_weights, dim=(-1, -2))
        true_paths_costs = torch.sum(true_paths * true_weights, dim=(-1, -2))
        return torch.mean(pred_paths_costs / true_paths_costs)


class CostRatioAccuracy(Metric):
    def __init__(self, exp):
        super().__init__("cr_below_" +
                         "{:06.3f}".format(10 ** -(exp - 1)).replace(".", "_") +
                         "_pc_accuracy")
        self.exp = exp

    def eval(self, **kwargs):
        pred_paths = kwargs["pred_paths"]
        true_paths = kwargs["true_paths"]
        true_weights = kwargs["true_weights"]
        pred_paths_costs = torch.sum(pred_paths * true_weights, dim=(-1, -2))
        true_paths_costs = torch.sum(true_paths * true_weights, dim=(-1, -2))
        cost_ratios = pred_paths_costs / true_paths_costs
        return torch.mean((cost_ratios <= 1.0 + 10 ** -(self.exp + 1)).float())


class SummaryMetric(Metric):
    def __init__(self, target, metric):
        super().__init__(target + "_" + metric)
        assert target in ['weights', 'heuristic'], target
        assert metric in ['mean', 'var'], metric
        self.target = target
        self.metric = metric

    def eval(self, **kwargs):
        target = kwargs["pred_" + self.target]
        if self.metric == "mean":
            return torch.mean(target)
        else:
            return torch.mean(torch.var(target, dim=(1, 2)))
