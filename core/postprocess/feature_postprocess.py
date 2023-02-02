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
from core.utils.utility import parse_dict


class FeaturePostProcess(object):
    """Convert between text-label and text-index"""

    def __init__(self, dict_path, debug=False, **kwargs):
        super(FeaturePostProcess, self).__init__()
        self.label_list = parse_dict(dict_path)
        self.debug = debug

    def normalize(self, x):
        """L2-normalize x."""
        return x / np.linalg.norm(x, ord=2)

    def cal_cosine(self, template, pred, s=4):
        """
        calculate cosine similarity between template and pred
        """
        return np.dot(self.normalize(pred), self.normalize(template))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def cal_cosine_template(self, templates, pred, label_index):
        label_indexs = []
        cosines = []
        for l, values in templates.items():
            for template in values:
                cosine = self.cal_cosine(template, pred)
                cosines.append(cosine)
                label_indexs.append(self.label_list.index(l))

        index = np.argmax(cosines)
        cosine = cosines[index]
        pred_label_index = label_indexs[index]

        label_indexs = np.array(label_indexs)
        cosines = np.array(cosines)
        # Inter class cosine
        inter_cond = label_indexs != label_index
        inter_index = np.argmax(cosines[inter_cond])
        inter_cosine = cosines[inter_cond][inter_index]
        inter_cosine_avg = np.mean(cosines[inter_cond])

        # Inner class cosine
        inner_cond = label_indexs == label_index
        inner_index = np.argmin(cosines[inner_cond])
        inner_cosine = cosines[inner_cond][inner_index]
        inner_cosine_avg = np.mean(cosines[inner_cond])

        return cosine, pred_label_index, cosines, inter_cosine, inner_cosine, inter_cosine_avg, inner_cosine_avg

    def __call__(self, templates, features, *args, **kwargs):
        cosines = []
        pred_labels = []
        true_labels = []
        count = 0
        total = 0
        inter_cosines = []
        inner_cosines = []
        inter_cosines_avg = []
        inner_cosines_avg = []
        for pred, label_index, img in features:
            cosine, pred_label_index, _cosines, inter_cosine, inner_cosine, inter_cosine_avg, inner_cosine_avg = self.cal_cosine_template(
                templates, pred, label_index)
            cosines.append(cosine)
            inter_cosines.append(inter_cosine)
            inner_cosines.append(inner_cosine)
            inter_cosines_avg.append(inter_cosine_avg)
            inner_cosines_avg.append(inner_cosine_avg)
            pred_labels.append(self.label_list[pred_label_index])
            true_labels.append(self.label_list[label_index])
            _cosines = np.array(_cosines)

            if len(_cosines[_cosines < 0]) != 0:
                count += 1
                total += len(_cosines[_cosines < 0])
            if self.debug:
                if label_index != pred_label_index:
                    print(len(pred), len(pred[pred < 0]))
                    print(
                        "true label: {}, pred label: {}, cosine: {}, cosines: {}"
                        .format(self.label_list[label_index],
                                self.label_list[pred_label_index], cosine,
                                sorted(_cosines)[-3:]))
        return (np.array(cosines), np.array(inter_cosines),
                np.array(inner_cosines), np.array(inter_cosines_avg),
                np.array(inner_cosines_avg), np.array(pred_labels),
                np.array(true_labels))
