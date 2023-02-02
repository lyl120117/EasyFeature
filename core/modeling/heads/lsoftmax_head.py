import math
import torch
from torch import nn
from scipy.special import binom

import matplotlib.pyplot as plt


class LSoftmaxHead(nn.Module):

    def __init__(self,
                 in_channels,
                 class_num,
                 margin=2,
                 base=1000,
                 gamma=0.00002,
                 power=1,
                 beta_min=0):
        super(LSoftmaxHead, self).__init__()
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
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1,
                                                       2)))  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2))  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers)))  # n
        self.signs = torch.ones(margin // 2 + 1)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def get_weights(self):
        return self.weight

    def to(self, device):
        self.device = device
        self.weight = self.weight.to(device)
        self.C_m_2n = self.C_m_2n.to(device)
        self.cos_powers = self.cos_powers.to(device)
        self.sin2_powers = self.sin2_powers.to(device)
        self.signs = self.signs.to(device)
        return self

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1)**self.cos_powers.unsqueeze(
            0)  # cos^{m - 2n}
        sin2_terms = (
            sin2_theta.unsqueeze(1)  # sin2^{n}
            **self.sin2_powers.unsqueeze(0))

        cos_m_theta = (
            self.signs.unsqueeze(0)
            *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
            self.C_m_2n.unsqueeze(0) * cos_terms * sin2_terms).sum(
                1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

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

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm * x_norm *
                                    (((-1)**k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta *
                                         logit[indexes, targets]) / (1 + beta)

            logit[indexes, targets] = logit_target_updated_beta
            # self.beta *= self.scale
            return logit
        else:
            assert targets is None
            return input.mm(self.weight)


if __name__ == '__main__':
    min_beta = 0

    base = 1000
    gamma = 0.000025
    power = 45
    iteration = 0

    lambdas = []
    for i in range(70 * 210):
        iteration += 1
        lambda_ = base * (1 + gamma * iteration)**(-power)
        lambda_ = max(lambda_, min_beta)
        lambdas.append(lambda_)
        if i % 50 == 0:
            print(i, lambda_)

    plt.plot(lambdas)
    plt.show()