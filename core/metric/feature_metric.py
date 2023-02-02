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
import numpy as np
from tqdm import tqdm


def cal_acc(args):
    cosines, _, _, _, _, pred_labels, true_labels = args

    match_cosines = cosines[pred_labels == true_labels]
    best_acc = len(match_cosines) / len(cosines)
    # best_acc = np.mean((pred_labels == true_labels).astype(int))
    best_th = match_cosines[np.argmin(match_cosines)]
    cond = cosines >= best_th
    best_precision = np.mean(
        (pred_labels[cond] == true_labels[cond]).astype(int))
    best_ap = best_precision * best_acc

    return best_acc, best_th, best_precision, best_ap


class FeatureMetric(object):

    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        self.best_acc, self.best_th, self.best_precision, self.best_ap = cal_acc(
            pred_label)

        return {
            "acc": self.best_acc,
            "th": self.best_th,
            "precision": self.best_precision,
            "AP": self.best_ap,
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0
            }
        """
        return {
            "acc": self.best_acc,
            "th": self.best_th,
            "precision": self.best_precision,
            "AP": self.best_ap,
        }

    def reset(self):
        self.best_acc = 0
        self.best_th = 0
        self.best_precision = 0
        self.best_ap = 0
