import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from core.data import build_dataloader
from core.postprocess import build_post_process
from core.utils.logging import get_logger
from tools.program import ArgsParser, load_config, merge_config

import torch

logger = get_logger()


def get_size(config_transforms):
    size = (28, 28)
    for transform in config_transforms:
        print("get_size:", transform)
        if "ClsResize" in transform.keys():
            size = transform["ClsResize"].get("size", (28, 28))
        elif "CVResize" in transform.keys():
            size = transform["CVResize"].get("size", (28, 28))
    return size


def main(config):
    device = ("cuda" if config["Global"]["use_gpu"]
              and torch.cuda.is_available() else "cpu")

    output = "output/test_reader/"
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    valid_dataloader = build_dataloader(config, "Train", logger)
    size = get_size(config["Train"]["dataset"]["transforms"])
    global_config = config["Global"]
    post_process_class = build_post_process(config["PostProcess"],
                                            global_config)
    means_path = config['Global'].get('means_path', None)
    if means_path:
        means = np.load(means_path)
    else:
        means = 127.5
    print("means:", means)
    label_list = post_process_class.label_list

    index = 0
    for batch in valid_dataloader:
        image = batch[0].numpy()[0]
        label = batch[1].numpy()[0]
        if len(batch) == 3:
            path = batch[2][0]
            name = os.path.basename(path)
        else:
            name = f'{index}.jpg'
        # image = image.reshape(size)
        label = label_list[int(label)]
        label = label.replace("\\", "_")
        path = os.path.join(output, f"{label}_{name}")
        logger.info("save image to {}".format(path))
        image = np.transpose(image, (1, 2, 0))
        image = (image + 1) * means
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
        cv2.imwrite(path, image)
        index += 1
        # break


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    main(config)
