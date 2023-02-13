from torch import nn
import torch


class EucLoss(nn.Module):

    def __init__(self, reduction='mean', m=-1, s=-1):
        super(EucLoss, self).__init__()
        self.reduction = reduction
        self.m = m
        self.max_euc = 2 * s**2

    def forward(self, predicts, batch):
        label = batch[1]
        predicts = -predicts
        indexes = range(predicts.size(0))
        p_loss = predicts[indexes, label]
        one_hot = torch.zeros_like(predicts)
        one_hot[indexes, label] = 1
        if self.m > 0:
            margin_loss = (1 - one_hot) * (predicts - self.m)
        else:
            margin_loss = (1 - one_hot) * (predicts - self.max_euc)
        n_loss = torch.sqrt(torch.mean(torch.pow(margin_loss, 2), dim=1))
        loss = n_loss + p_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return {'loss': loss}