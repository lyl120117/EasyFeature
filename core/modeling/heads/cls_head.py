from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import numpy as np


class ClsHead(nn.Module):

    def __init__(self, in_channels, class_num=10):
        super(ClsHead, self).__init__()
        self.fc = nn.Linear(in_channels, class_num)
        self.out_channels = class_num

    def forward(self, x, targets=None):
        logits = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return logits