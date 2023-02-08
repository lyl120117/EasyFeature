import math
import torch
from torch import nn
from scipy.special import binom

import matplotlib.pyplot as plt


class LSoftmaxPlusHead(nn.Module):

    def __init__(self,
                 in_channels,
                 class_num,
                 margin=0.5,
                 base=1000,
                 gamma=0.00002,
                 power=1,
                 beta_min=1e-6):
        super(LSoftmaxPlusHead, self).__init__()
        self.margin = margin  # m
        self.beta = 0
        self.beta_min = beta_min

        self.base = base  # base
        self.gamma = gamma  # gamma
        self.power = power  # power
        self.iter = 0

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, class_num))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def get_weights(self):
        return self.weight.t()

    def forward(self, input, targets=None):
        if self.training:
            self.iter += 1
            self.beta = self.base * (1 + self.gamma * self.iter)**(-self.power)
            assert targets is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, targets]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, targets].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            with torch.no_grad():
                theta_target = torch.acos(cos_theta_target)
                theta_m_target = theta_target + self.margin
                theta_m_target.clamp_(1e-5, 3.14159)
                d_theta_m_target = torch.cos(theta_m_target) - cos_theta_target
            logit_target_updated = (cos_theta_target + d_theta_m_target) * (
                w_target_norm * x_norm + 1e-10)

            # update logit
            logit_target_updated_beta = (logit_target_updated +
                                         beta * logit_target) / (1 + beta)
            logit[indexes, targets] = logit_target_updated_beta

            return logit
        else:
            assert targets is None
            return input.mm(self.weight)


if __name__ == '__main__':
    batch_size = 128
    in_channels = 256
    class_num = 10
    margin = 0.5
    device = 'cuda'

    input = torch.randn(batch_size, in_channels)
    targets = torch.randint(0, class_num, (batch_size, ))
    input = input.to(device)
    targets = targets.to(device)

    head = LSoftmaxPlusHead(in_channels, class_num, margin=margin)
    head.reset_parameters()
    output = head(input, targets)
    print('output:', output.shape)