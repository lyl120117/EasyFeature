import copy

from torch.utils.data import DataLoader
from .simple_dataset import SimpleDataset

__all__ = ["build_dataloader"]


def build_dataloader(config, mode, logger):
    config = copy.deepcopy(config)

    support_dict = ["SimpleDataset"]
    module_name = config[mode]["dataset"]["name"]
    assert module_name in support_dict, Exception(
        "DataSet only support {}".format(support_dict))
    assert mode in [
        "Train",
        "Eval",
        "Test",
        "Template",
        "TestTemplate",
    ], "Mode should be Train, Eval, Test, Template or TestTemplate."

    dataset = eval(module_name)(config, mode, logger)

    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader
