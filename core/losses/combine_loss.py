import torch
import torch.nn as nn
from core.losses import *


def _build_loss(module_name, config):
    module_class = eval(module_name)(**config)
    return module_class


class CombineLoss(nn.Module):

    def __init__(self, losses=None):
        super(CombineLoss, self).__init__()
        self.losses = self.build_losses(losses)

    def build_losses(self, configs: dict):
        losses = {}
        for config in configs:
            name = list(config.keys())[0]
            losses[name] = (_build_loss(name, config[name]))
        return losses

    def forward(self, predicts, batch):
        loss = 0
        for name, loss_func in self.losses.items():
            if 'ClsLoss' in name:
                loss += loss_func(predicts, batch)['loss']
            elif 'MSE' in name:
                loss += loss_func(predicts, batch)['loss']
        return {'loss': loss}
