from torch import nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import numpy as np

# Euclidean distance to the margin of the feature space


class EucMarginHead(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_channels, class_num, margin_add=0.5):
        super(EucMarginHead, self).__init__()
        self.in_features = in_channels
        self.out_features = class_num

        self.margin_add = margin_add

        assert self.margin_add >= 0, "margin_add must be lager than 0, but got {}".format(
            self.margin_add)

        self.weight = Parameter(
            torch.FloatTensor(self.out_features, self.in_features))
        nn.init.kaiming_normal_(self.weight)

    def get_weights(self):
        return self.weight

    def forward(self, input, targets):

        # Euclidean distance
        # (input - weight) ^ 2 =  input ^ 2 + weight ^ 2 - 2 * input * weight
        #                      => 2 - 2 * input * weight
        norm_input = input.norm(dim=1, keepdim=True, p=2)
        norm_weight = self.weight.norm(dim=1, keepdim=True, p=2)
        logits = input.mm(self.weight.t())
        output = torch.pow(norm_input, 2) + torch.pow(norm_weight,
                                                      2).t() - 2 * logits
        output = torch.sqrt(output)
        if self.training:
            one_hot = torch.zeros(output.size(), device=output.device)
            one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
            output = output + one_hot * self.margin_add
            output = torch.clamp_min(output, min=1e-12)
        output = -output
        return output


if __name__ == '__main__':
    input = torch.randn(2, 3)

    print('input:', input)
    head = EucsMarginHead(3, 4)

    output = head(input, None)
    print(output.shape)
    print('output:', output)