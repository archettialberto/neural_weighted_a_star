import cv2
import numpy as np

from a_star.ray_a_star import a_star_batch
from utils.collections import LogData


class VisualLogger:
    PATH_COLOR     = [  0, 225, 255]
    EXP_NODE_COLOR = [255,   0, 255]
    SOURCE_COLOR   = [  0, 255, 255]
    TARGET_COLOR   = [  0, 150, 255]

    WEIGHTS_COLOR = [255, 128, 255]
    HEURISTIC_COLOR = [200, 0, 0]

    GRID_COLOR = [200, 200, 200]

    @staticmethod
    def stacked_log(data: dict, epoch: int, sample_id: int, show=True):
        log_data = VisualLogger.tensors_to_numpy(data, 'true', sample_id)
        label = 'ep{}_{}_image_{}'.format(epoch, 'true', sample_id)
        image = VisualLogger.generate_image_log(label, log_data, show=False)
        logs = [image]

        for prefix in ["true", "pred"]:
            log_data = VisualLogger.tensors_to_numpy(data, prefix, sample_id)

            label = 'ep{}_{}_journey_{}'.format(epoch, prefix, sample_id)
            journey = VisualLogger.generate_journey_log(label, log_data, show=False)
            logs.append(journey)

            label = 'ep{}_{}_weights_{}'.format(epoch, prefix, sample_id)
            weights = VisualLogger.generate_tensor_log(label, log_data, pick='weights', show=False)
            logs.append(weights)

            label = 'ep{}_{}_heuristic_{}'.format(epoch, prefix, sample_id)
            heuristic = VisualLogger.generate_tensor_log(label, log_data, pick='heuristic', show=False)
            logs.append(heuristic)

        stack = np.hstack(logs)

        if show:
            cv2.imshow('ep{}_{}_log_{}'.format(epoch, prefix, sample_id), stack)
            cv2.waitKey()
        return stack

    @staticmethod
    def tensors_to_numpy(data: dict, prefix: str, sample_id: int) -> LogData:
        if prefix == 'pred':
            a_star_sol_pred = a_star_batch(
                data["pred_weights"],
                data["pred_heuristic"],
                data["source"],
                data["target"]
            )
            data["pred_paths"] = a_star_sol_pred.paths
            data["pred_exp_nodes"] = a_star_sol_pred.exp_nodes
        return LogData(
            image=data["image"][sample_id].detach().cpu().numpy(),
            source=data["source"][sample_id].detach().cpu().numpy(),
            target=data["target"][sample_id].detach().cpu().numpy(),
            path=data[prefix + "_paths"][sample_id].detach().cpu().numpy(),
            weights=data[prefix + "_weights"][sample_id].detach().cpu().numpy(),
            heuristic=data[prefix + "_heuristic"][sample_id].detach().cpu().numpy(),
            exp_nodes=data[prefix + "_exp_nodes"][sample_id].detach().cpu().numpy(),
        )

    @staticmethod
    def generate_image_log(label: str, log_data: LogData, show=True):
        # image array
        image = log_data.image.copy()
        image = VisualLogger.normalize(image)
        image = VisualLogger.resize(image)

        # show
        if show:
            cv2.imshow(label, image)
            cv2.waitKey()
        return image

    @staticmethod
    def image_to_grayscale(array):
        g_img = np.zeros_like(array)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        g_img[:, :, 0] = array
        g_img[:, :, 1] = array
        g_img[:, :, 2] = array
        return g_img

    @staticmethod
    def generate_journey_log(label: str, log_data: LogData, show=True):
        # image array
        image = log_data.image.copy()
        image = VisualLogger.normalize(image)
        image = VisualLogger.resize(image)
        image = VisualLogger.image_to_grayscale(image)

        # path (without source and target), source, and target arrays
        path = log_data.path.copy()
        source = np.zeros_like(path)
        source[log_data.source[0], log_data.source[1]] = 1
        target = np.zeros_like(path)
        target[log_data.target[0], log_data.target[1]] = 1
        path -= source
        path -= target

        path = VisualLogger.array_to_three_channels(path, color=VisualLogger.PATH_COLOR)
        path = VisualLogger.normalize(path)
        path = VisualLogger.resize(path)

        source = VisualLogger.array_to_three_channels(source, color=VisualLogger.SOURCE_COLOR)
        source = VisualLogger.normalize(source)
        source = VisualLogger.resize(source)

        target = VisualLogger.array_to_three_channels(target, color=VisualLogger.TARGET_COLOR)
        target = VisualLogger.normalize(target)
        target = VisualLogger.resize(target)

        # expanded nodes (without path) array
        exp_nodes = log_data.exp_nodes.copy() - log_data.path.copy()

        exp_nodes = VisualLogger.array_to_three_channels(exp_nodes, color=VisualLogger.EXP_NODE_COLOR)
        exp_nodes = VisualLogger.normalize(exp_nodes)
        exp_nodes = VisualLogger.resize(exp_nodes)

        # merging step
        merge = cv2.addWeighted(image, 0.5, exp_nodes, 0.5, 0)
        merge = cv2.addWeighted(merge, 1.0, source, 1.0, 0)
        merge = cv2.addWeighted(merge, 1.0, target, 1.0, 0)
        merge = cv2.addWeighted(merge, 1.0, path, 1.0, 0)

        # rows, cols, _ = merge.shape
        # x_max, y_max = log_data.path.shape
        # grid = VisualLogger.generate_grid((rows, cols), (x_max, y_max))
        # merge = cv2.addWeighted(merge, 1.0, grid, 0.5, 0)

        # show
        if show:
            cv2.imshow(label, merge)
            cv2.waitKey()
        return merge

    @staticmethod
    def generate_tensor_log(label: str, log_data: LogData, pick: str, show=True):
        assert pick in ['weights', 'heuristic']
        # image array
        image = log_data.image
        image = VisualLogger.normalize(image)
        image = VisualLogger.resize(image)
        image = VisualLogger.image_to_grayscale(image)
        rows, cols, _ = image.shape

        # weights/heuristic array
        if pick == 'weights':
            COLOR = VisualLogger.PATH_COLOR
            tensor = log_data.weights
        else:
            COLOR = VisualLogger.PATH_COLOR
            tensor = log_data.heuristic
        x_max, y_max = tensor.shape
        tensor = VisualLogger.normalize(tensor, mul_255=False)
        tensor = VisualLogger.array_to_three_channels(tensor, color=COLOR, color2=VisualLogger.EXP_NODE_COLOR)
        tensor = VisualLogger.resize(tensor)

        # merging step

        merge = cv2.addWeighted(image, 0.25, tensor, 0.75, 0)

        # grid = VisualLogger.generate_grid((rows, cols), (x_max, y_max))
        # merge = cv2.addWeighted(merge, 1.0, grid, 0.5, 0)

        # show
        if show:
            cv2.imshow(label, merge)
        return merge

    @staticmethod
    def generate_grid(img_shape, array_shape):
        rows, cols = img_shape
        x_max, y_max = array_shape
        dx = int(rows * 1.0 / x_max)
        dy = int(cols * 1.0 / y_max)
        grid = np.zeros(img_shape + (3,), dtype=np.uint8)
        for i in range(x_max):
            grid[i * dx, :] = VisualLogger.GRID_COLOR
        for j in range(y_max):
            grid[:, j * dy] = VisualLogger.GRID_COLOR
        grid[:, -1] = VisualLogger.GRID_COLOR
        grid[-1, :] = VisualLogger.GRID_COLOR
        return grid

    @staticmethod
    def array_to_three_channels(array, color, color2=None):
        shape = array.shape
        shape = shape + (3,)
        img = np.zeros(shape, dtype=np.uint8)
        if color2 is None:
            img[:, :, 0] = array * (255 - color[0])
            img[:, :, 1] = array * (255 - color[1])
            img[:, :, 2] = array * (255 - color[2])
        else:
            img[:, :, 0] = array * 255
            img[:, :, 1] = 0
            img[:, :, 2] = (1 - array) * 255
        return img

    @staticmethod
    def resize(array, shape=(200, 200)):
        return cv2.resize(array, shape, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def normalize(array, mul_255=True):
        min = np.min(array)
        max = np.max(array)
        if max == 0:
            max = 1
        if mul_255:
            return ((array - min) / (max - min) * 255).astype(np.uint8)
        else:
            return (1. * (array - min) / (max - min)).astype(np.float32)
