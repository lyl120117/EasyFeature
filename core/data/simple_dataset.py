import torch
from torch.utils.data import Dataset
from core.data.imaug import create_transform
import numpy as np
import os
from typing import Any, Tuple
import cv2


class SimpleDataset(Dataset):

    def __init__(self, config, mode, logger):
        self.logger = logger
        self.data_idx_order_list = ['']

        dataset_config = config[mode]['dataset']

        self.transform = create_transform(dataset_config['transforms'])
        self.target_transform = None

        self.data_root = dataset_config['data_root']
        data_names = dataset_config['data_names']
        self.image_name = data_names[0]
        self.label_name = data_names[1]

        self.images, self.labels = self.load_data()

    def load_data(self):
        images = np.load(os.path.join(self.data_root, self.image_name))
        labels = np.load(os.path.join(self.data_root, self.label_name))
        return images, labels

    def __len__(self):
        return len(self.images)

    # Returns:
    #     tuple: (image, target) where target is index of the target class.
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.images[index], int(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target