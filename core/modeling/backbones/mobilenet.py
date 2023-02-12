from torchvision import models
from torch import nn
import torch


class MobileNetV2(nn.Module):

    def __init__(self, in_channels=192, out_channels=128):
        super(MobileNetV2, self).__init__()
        self.model = models.MobileNetV2(num_classes=1000,
                                        width_mult=1.0).features
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.fc = nn.Linear(1280, out_channels)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2(out_channels=256)
    x = torch.randn(8, 3, 28, 28)
    y = model(x)
    print(y.shape)
