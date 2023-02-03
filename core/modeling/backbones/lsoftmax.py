from torch import nn
from torch.nn import init
from collections import OrderedDict


class LSoftmaxBackbone(nn.Module):

    def __init__(self,
                 in_channels=192,
                 input_size=28,
                 input_conv=[3, 1, 64],
                 conv_channels=[[3, 64, 3], [3, 64, 3], [3, 64, 3]],
                 use_bias=True,
                 out_channels=128):
        super(LSoftmaxBackbone, self).__init__()
        self.out_channels = out_channels
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()
        out_channels = input_conv[2]
        self.linear_relu_stack.add_module('bn0_0', nn.BatchNorm2d(in_channels))
        self.linear_relu_stack.add_module(
            'conv0_0',
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=input_conv[0],
                      padding=input_conv[1]))
        self.linear_relu_stack.add_module('prelu0_0', nn.PReLU(out_channels))
        self.linear_relu_stack.add_module('bn0_1',
                                          nn.BatchNorm2d(out_channels))

        in_channels = out_channels
        for i, layers in enumerate(conv_channels):
            i += 1
            for j in range(layers[2]):
                out_channels = layers[1]
                self.linear_relu_stack.add_module(
                    f'conv{i}_{j}',
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1))
                in_channels = out_channels
                self.linear_relu_stack.add_module(f'prelu{i}_{j}',
                                                  nn.PReLU(out_channels))
                self.linear_relu_stack.add_module(f'bn{i}_{j}',
                                                  nn.BatchNorm2d(out_channels))
            self.linear_relu_stack.add_module(f'pool{i}', nn.MaxPool2d(2, 2))

        pool_num = len(conv_channels)
        feature_size = int(input_size /
                           (2**pool_num))**2 * conv_channels[-1][1]

        self.fc = nn.Sequential(
            OrderedDict([('fc0',
                          nn.Linear(in_features=feature_size,
                                    out_features=self.out_channels,
                                    bias=use_bias)),
                         ('fc0_bn', nn.BatchNorm1d(self.out_channels))]))

        self.reset_parameters(conv_channels)

    def reset_parameters(self, conv_channels):

        def init_kaiming(layer: nn.Module):
            init.kaiming_normal_(layer.weight.data)
            if layer.bias is not None:
                init.constant_(layer.bias.data, val=0)

        init_kaiming(self.linear_relu_stack.conv0_0)
        for i, layers in enumerate(conv_channels):
            i += 1
            for j in range(layers[2]):
                init_kaiming(getattr(self.linear_relu_stack, f'conv{i}_{j}'))
        init_kaiming(self.fc.fc0)

    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits