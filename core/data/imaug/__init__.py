from torchvision.transforms import Compose
from .transforms import *
from .aug import *

__all__ = ['create_transform', 'create_transform']


def create_transform(op_param_list):
    ops = create_operators(op_param_list)
    return Compose(ops)


def create_operators(op_param_list):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list,
                      list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = eval(op_name)(**param)
        ops.append(op)
    return ops