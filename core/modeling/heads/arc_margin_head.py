from torch import nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import math
import numpy as np


class ArcMarginHead(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """

    def __init__(self, in_channels, class_num, s=64., m=0.5):
        super(ArcMarginHead, self).__init__()
        self.feat_dim = in_channels
        self.num_class = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(self.feat_dim, self.num_class))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def get_weights(self):
        return self.weight.t()

    def forward(self, input, targets):
        with torch.no_grad():
            w = F.normalize(self.weight.data, dim=0)

        input = F.normalize(input, dim=1)
        if self.training:
            cos_theta = input.mm(w)
            with torch.no_grad():
                theta_m = torch.acos(cos_theta)
                theta_m.scatter_(1, targets.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, 3.14159)
                d_theta = torch.cos(theta_m) - cos_theta
                # d_theta = torch.where(cos_theta > 0, d_theta,
                #                       torch.zeros_like(d_theta))

            logits = cos_theta + d_theta
            logits *= self.s
        else:
            logits = input.mm(w)
        return logits