from torch import nn
import torch

from .euc_loss import EucLoss
from .focal_loss import FocalLoss


class FocalEucLoss(nn.Module):

    def __init__(self, reduction='mean', m=5, gamma=2):
        super(FocalEucLoss, self).__init__()
        self.reduction = reduction
        self.euc_loss = EucLoss(reduction='none', m=m)
        self.focal_loss = FocalLoss(reduction=reduction, gamma=gamma)

    def forward(self, predicts, batch):
        loss = self.euc_loss(predicts, batch)['loss']
        loss = self.focal_loss(loss)
        return loss
