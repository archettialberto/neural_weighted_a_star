import torch
import torch.nn as nn

from model.feature_extractor.feature_extractor import FeatureExtractor
from model.feature_extractor.resnet_builder import resnet18


class CombResnet18Pad(FeatureExtractor):
    def __init__(self, in_channels, out_channels, x_max=12, y_max=12, mean_features=False):
        super().__init__(in_channels, out_channels)
        self.resnet_model = resnet18(
            pretrained=False,
            num_classes=x_max * y_max
        )
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self.mean_features = mean_features
        if self.mean_features:
            assert out_channels == 1, out_channels

    def extract_features(self, x, s, t):
        x = x.permute(0, 3, 1, 2)
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        x = self.resnet_model.layer2(x)
        x = self.resnet_model.layer3(x)
        if self.mean_features:
            x = torch.mean(x, dim=1, keepdim=True)
        else:
            x = self.out_conv(x)
        return x.permute(0, 2, 3, 1)
