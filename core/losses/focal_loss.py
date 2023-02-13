import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, reduction='mean', gamma=2):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, loss):
        p = torch.exp(-loss)
        loss = (1 - p)**self.gamma * loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return {'loss': loss}
