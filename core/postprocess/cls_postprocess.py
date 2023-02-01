# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import torch
from core.utils.utility import parse_dict
import numpy as np


class ClsPostProcess(object):
    """Convert between text-label and text-index"""

    def __init__(self, dict_path, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.dicts = parse_dict(dict_path)
        self.label_list = self.dicts

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().detach().numpy()
        pred_idxs = preds.argmax(axis=1)

        decode_out = [(self.dicts[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(self.dicts[idx], 1.0) for idx in label]
        return decode_out, label
