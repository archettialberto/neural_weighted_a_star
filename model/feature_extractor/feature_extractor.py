from abc import abstractmethod, ABC

import torch


class FeatureExtractor(torch.nn.Module, ABC):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, s, t):
        assert len(x.shape) == 4, x.shape
        assert x.shape[-1] >= self.in_channels, x.shape[-1]
        batch = x.shape[0]
        x = self.extract_features(x[:, :, :, 0:self.in_channels], s, t)
        assert len(x.shape) == 4
        assert x.shape[0] == batch, x.shape[0]
        assert x.shape[-1] == self.out_channels, x.shape[-1]
        return x

    @abstractmethod
    def extract_features(self, x, s, t):
        pass
