# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

from .lsoftmax import LSoftmaxBackbone
from .mobilenet import MobileNetV2
from .resnet import ResNet18

__all__ = ["build_backbone"]


def build_backbone(config):
    config = copy.deepcopy(config)
    support_dict = ['LSoftmaxBackbone', 'MobileNetV2', 'ResNet18']

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "backbone only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
