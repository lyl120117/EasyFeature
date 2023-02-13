from torchvision import models
from torch import nn
import torch


class MobileNetV2(nn.Module):

    def __init__(self, in_channels=3, out_channels=128):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False, num_classes=1000)
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.model.features[0][0] = nn.Conv2d(in_channels,
                                              32,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              bias=False)
        self.model.classifier = nn.Linear(1280, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNetV3(nn.Module):

    def __init__(self, in_channels=3, out_channels=128, large=True):
        super(MobileNetV3, self).__init__()
        if large:
            self.model = models.mobilenet_v3_large(pretrained=False,
                                                   num_classes=1000)
        else:
            self.model = models.mobilenet_v3_small(pretrained=False,
                                                   num_classes=1000)
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.model.features[0][0] = nn.Conv2d(in_channels,
                                              16,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              bias=False)
        self.model.classifier = nn.Linear(960, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MobileNetV3(out_channels=256)
    x = torch.randn(8, 3, 28, 28)
    print(model)
    y = model(x)
    print(y.shape)
