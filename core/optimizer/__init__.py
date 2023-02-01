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

import copy

from core.optimizer.optimizer import *
from core.optimizer.learning_rate import *

__all__ = ['build_optimizer']


def build_scheduler(config, optimizer, max_steps):
    support_dict = ["CosineAnnealing", 'MultiStep']

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "schedule only support {}".format(support_dict))

    module_class = eval(module_name)(**config)
    return module_class(optimizer)


def build_optimizer(config, epochs, step_each_epoch, parameters):
    support_dict = ["Adam", 'SGD']

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "optimizer only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class(parameters)
