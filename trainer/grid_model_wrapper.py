import numpy as np
import torch
from yapt import BaseModel
from yapt.utils.metrics import AverageMeter

from a_star.neighbor_utils import NeighborUtils
from a_star.solver_factory import get_a_star_solver
from model.model_factory import get_model
from trainer.losses import LossWrapper
from trainer.metrics import MetricsWrapper
from visual_logger import VisualLogger


class GridModelWrapper(BaseModel):
    def _build_model(self):
        args = self.args
        self.model = get_model(args.model.name, args.model.params)
        self.solver = get_a_star_solver(args.solver.name, args.solver.params)
        self.loss_wrapper = LossWrapper(args.loss)
        self.metrics_wrapper = MetricsWrapper()
        self.trained = False
        self.nu = None

    def forward(self, x, s, t):
        return self.model(x, s, t)

    def process_batch(self, batch):
        image, image_st = batch[0:2]
        true_weights = batch[2]
        true_heuristic = batch[3]
        source, target = batch[4:6]
        true_paths = batch[6]
        true_exp_nodes, true_opt_exp_nodes = batch[7:9]

        batch, x_max, y_max = true_weights.shape
        if self.nu is None:
            self.nu = NeighborUtils(x_max, y_max)

        model_sol = self.model(image_st, source, target)
        pred_weights = model_sol.weights
        pred_heuristic = model_sol.heuristic

        assert (pred_weights >= 0).all(), torch.min(pred_weights)
        assert (pred_heuristic >= 0).all(), torch.min(pred_heuristic)

        solver_sol = self.solver(pred_weights, pred_heuristic, source, target)
        pred_paths = solver_sol.paths
        pred_exp_nodes = solver_sol.exp_nodes

        batch_data = dict()
        batch_data["image"] = image
        batch_data["source"] = source
        batch_data["target"] = target
        batch_data["true_weights"] = true_weights
        batch_data["true_paths"] = true_paths
        batch_data["true_exp_nodes"] = true_exp_nodes
        batch_data["true_opt_exp_nodes"] = true_opt_exp_nodes
        batch_data["true_heuristic"] = true_heuristic
        batch_data["pred_weights"] = pred_weights
        batch_data["pred_heuristic"] = pred_heuristic
        batch_data["pred_paths"] = pred_paths
        batch_data["pred_exp_nodes"] = pred_exp_nodes

        metrics = dict()
        metrics["loss"] = torch.tensor(0).float().to(image.device)
        for loss in self.loss_wrapper.losses:
            metrics[loss.get_name()] = loss(**batch_data)
            metrics["loss"] += metrics[loss.get_name()]

        return batch_data, metrics

    def _log_images(self, data):
        if self.args.loggers.img_logs is not None:
            num_logs = self.args.loggers.img_logs
            batch = data["image"].shape[0]
            indexes = np.random.choice(batch, num_logs, replace=False)
            for i in indexes:
                img = VisualLogger.stacked_log(data, self.epoch, i, show=False)
                self.logger.log_image('ep{}'.format(self.epoch), img)

    def _training_step(self, batch):
        batch_data, metrics = self.process_batch(batch)
        self.optimizer.zero_grad()
        metrics["loss"].backward()
        if self.args.optimizer.grad_clipping is not None:
            clip_val = self.args.optimizer.grad_clipping.value
            opt_params = sum([pg['params'] for pg in self.optimizer.param_groups], [])
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=clip_val, norm_type=2)
        self.optimizer.step()
        output = self.metrics_wrapper.get_stats_with_tqdm(metrics, self._train_meters, running=True)

        self.trained = True
        return output

    def _validation_step(self, batch):
        with torch.no_grad():
            batch_data, metrics = self.process_batch(batch)
            metrics = self.metrics_wrapper.compute_metrics(metrics, **batch_data)
            output = self.metrics_wrapper.get_stats_with_tqdm(metrics, self._val_meters, running=False)
            if self.trained:
                self._log_images(batch_data)
            self.trained = False
            return output

    def _reset_train_stats(self):
        self._train_meters['loss'] = AverageMeter('loss')
        for loss in self.loss_wrapper.losses:
            self._train_meters[loss.name] = AverageMeter(loss.name)

    def _reset_val_stats(self):
        self._val_meters['loss'] = AverageMeter('loss')
        for loss in self.loss_wrapper.losses:
            self._val_meters[loss.name] = AverageMeter(loss.name)
        for metric in self.metrics_wrapper.metrics:
            self._val_meters[metric.name] = AverageMeter(metric.name)

    @staticmethod
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index.to(a.device))
