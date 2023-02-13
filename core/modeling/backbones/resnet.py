from torchvision import models
from torch import nn
import torch


class ResNet18(nn.Module):

    def __init__(self, in_channels=3, out_channels=128):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=1000)
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.model.conv1 = nn.Conv2d(in_channels,
                                     64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)
        self.model.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = ResNet18(out_channels=256)
    x = torch.randn(8, 3, 28, 28)
    y = model(x)
    print(y.shape)