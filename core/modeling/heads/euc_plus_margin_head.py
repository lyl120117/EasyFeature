from torch import nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import numpy as np

# Euclidean distance to the margin of the feature space


class EucPlusMarginHead(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_channels,
                 class_num,
                 s=1,
                 margin_add=0.5,
                 easy_margin=False):
        super(EucPlusMarginHead, self).__init__()
        self.in_features = in_channels
        self.out_features = class_num

        self.margin_add = margin_add
        self.easy_margin = easy_margin
        self.s = s

        assert self.margin_add >= 0 and self.margin_add < 4, "margin_add must be lager than 0 and less than 4, but got {}".format(
            self.margin_add)

        self.weight = Parameter(
            torch.FloatTensor(self.out_features, self.in_features))
        nn.init.kaiming_normal_(self.weight)

    def get_weights(self):
        return self.weight

    def forward(self, input, targets=None):

        # Euclidean distance
        # (input - weight) ^ 2 =  input ^ 2 + weight ^ 2 - 2 * input * weight
        #                      => 2 - 2 * input * weight
        input = F.normalize(input)
        weight = F.normalize(self.weight)

        if self.training:
            input *= self.s
            weight *= self.s
            sp = self.s**2
            cosine = torch.matmul(input, weight.t())
            output = torch.sum(torch.square(input), dim=1).view(
                -1, 1) + torch.sum(torch.square(weight), dim=1) - 2 * cosine

            # Add Euclidean margin
            one_hot = torch.zeros(output.size(), device=output.device)
            one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
            output = output + one_hot * self.margin_add * sp
            output = torch.clamp(output, min=1e-6, max=4 * sp)
        else:
            cosine = torch.matmul(input, weight.t())
            output = torch.sum(torch.square(input), dim=1).view(
                -1, 1) + torch.sum(torch.square(weight), dim=1) - 2 * cosine

        output = torch.sqrt(output)
        output = -output
        return output


if __name__ == '__main__':
    input = torch.randn(2, 3)

    print('input:', input)
    head = EucPlusMarginHead(3, 4)

    output = head(input, None)
    print(output.shape)
    print('output:', output)