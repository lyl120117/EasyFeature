# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch


class ClsLoss(nn.Module):

    def __init__(self, label_smoothing=0., reduction='mean'):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction,
                                             label_smoothing=label_smoothing)
        self.reduction = reduction

    def forward(self, predicts, batch):
        label = batch[1]
        loss = self.loss_func(input=predicts, target=label)
        if self.reduction == 'mean':
            loss = loss.mean()
        return {'loss': loss}


if __name__ == '__main__':
    import os
    import sys
    import torch
    import numpy as np
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(__dir__)
    sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..')))

    from core.utils.contrast_acc import ContrastAccuracy

    ca = ContrastAccuracy('Softmax')
    preds = ca.get_input_tensor(prefix='preds')
    if preds is None:
        preds = np.random.randn(5, 10)
        ca.save_tensor(preds, is_input=True, prefix='preds')
    label = ca.get_input_tensor('label')
    if label is None:
        label = np.array([4, 3, 9, 8, 5])
        ca.save_tensor(label, is_input=True, prefix='label')

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_func(input=torch.tensor(preds),
                     target=torch.tensor(label)).numpy()
    print('loss:', loss)
    ca.save_tensor(loss, is_input=False)
