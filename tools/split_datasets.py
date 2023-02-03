import torchvision.datasets as datasets
import numpy as np
import cv2
import shutil
import os
import random
from argparse import ArgumentParser

DATASETS = {
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CelebA': [datasets.CelebA, [{
        'split': 'all'
    }, {
        'split': 'test'
    }]],
    'Omniglot':
    [datasets.Omniglot, [{
        'background': True
    }, {
        'background': False
    }]],
}


def load_data(data_root, dataset_type, download=True):
    #将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
    dataset = DATASETS[dataset_type]
    if type(dataset) is list:
        dataset, args = dataset
        train_data = dataset(root=data_root, download=download, **args[0])
        test_data = dataset(root=data_root, download=download, **args[1])
    else:
        train_data = dataset(root=data_root, train=True, download=download)
        test_data = dataset(root=data_root, train=False, download=download)

    label_dicts = train_data.classes if train_data.hasattr('classes') else None

    #将数据转换成numpy格式
    train_images = np.array(train_data.data)
    train_labels = np.array(train_data.targets)
    test_images = np.array(test_data.data)
    test_labels = np.array(test_data.targets)

    labels = np.concatenate((train_labels, test_labels))
    class_num = len(np.unique(labels))
    train_class_num = len(np.unique(train_labels))
    test_class_num = len(np.unique(test_labels))

    return train_images, train_labels, test_images, test_labels, class_num, train_class_num, test_class_num, label_dicts


def split_datasets(datasets, train_val_ratio=0.9, shuffle=True):
    if shuffle:
        random.shuffle(datasets)
    images = []
    labels = []
    for image, label in datasets:
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    count = len(images)
    train_count = int(count * train_val_ratio)
    train_images = images[:train_count]
    train_labels = labels[:train_count]
    val_images = images[train_count:]
    val_labels = labels[train_count:]
    return train_images, train_labels, val_images, val_labels


def debug_datasets(images, labels, label_dicts, dataset_type, debug_count=10):
    output = 'output/debug'
    output = os.path.join(output, dataset_type)
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    for i, (image,
            label) in enumerate(zip(images[:debug_count],
                                    labels[:debug_count])):
        l = label_dicts[label]
        cv2.imwrite(os.path.join(output, l + '_' + str(i) + '.jpg'), image)


def choose_template(images, labels, template_nums=1):
    template_images = []
    template_labels = []
    images_ = []
    labels_ = []
    for image, label in zip(images, labels):
        if template_labels.count(label) < template_nums:
            template_images.append(image)
            template_labels.append(label)
        else:
            images_.append(image)
            labels_.append(label)
    np.delete(template_images, 0)
    return images_, labels_, template_images, template_labels


def ignore_datas(images, labels, ignore_labels):
    images_ = []
    labels_ = []
    for image, label in zip(images, labels):
        if label not in ignore_labels:
            images_.append(image)
            labels_.append(label)
    return images_, labels_


def save_lists(lists, save_path):
    with open(save_path, 'w') as f:
        for l in lists:
            f.write(l + '\n')


def vec(v):
    return [int(i) for i in v.split(',')]


class ArgsParser(ArgumentParser):

    def __init__(self):
        super(ArgsParser, self).__init__()
        self.add_argument("-i",
                          "--ignore_labels",
                          type=vec,
                          default=[],
                          help="ignore labels")
        self.add_argument("-t",
                          "--template_nums",
                          type=int,
                          default=1,
                          help="template nums")
        self.add_argument("-s",
                          "--seed",
                          default=0,
                          type=int,
                          help="random seed")
        self.add_argument("-d",
                          "--dataset_dir",
                          type=str,
                          default='datasets/',
                          help="Save dataset dir")
        self.add_argument("-dt",
                          "--dataset_type",
                          type=str,
                          default='CIFAR10',
                          help="Only support [{}]".format(', '.join(
                              DATASETS.keys())))


def cal_mean(images):
    images = np.array(images)
    mean = np.mean(images, axis=(0, 1, 2))
    return mean


def save_seed(seed, save_path):
    with open(save_path, 'w') as f:
        f.write(str(seed))


if __name__ == '__main__':
    args = ArgsParser().parse_args()
    ignore_labels = args.ignore_labels
    template_nums = args.template_nums
    dataset_dir = args.dataset_dir
    dataset_type = args.dataset_type

    seed = args.seed
    if seed == 0:
        seed = random.randint(10000, 100000)
    random.seed(seed)
    np.random.seed(seed)

    train_images, train_labels, test_images, test_labels, class_num, train_class_num, test_class_num, label_dicts = load_data(
        dataset_dir, dataset_type)

    # Ignore training labels if needed
    train_images, train_labels = ignore_datas(train_images, train_labels,
                                              ignore_labels)

    batchs = []
    for image, label in zip(train_images, train_labels):
        batchs.append((image, label))
    train_images, train_labels, val_images, val_labels = split_datasets(batchs)

    data_root = os.path.join(dataset_dir, dataset_type)
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if label_dicts is not None:
        save_lists(label_dicts, os.path.join(data_root, 'dicts.txt'))

    save_seed(seed, os.path.join(data_root, 'seed.txt'))
    if dataset_type in ['MNIST']:
        means = 127.5
    else:
        means = cal_mean(train_images)
    np.save(os.path.join(data_root, 'means.npy'), means)

    np.save(os.path.join(data_root, 'train_images.npy'), train_images)
    np.save(os.path.join(data_root, 'train_labels.npy'), train_labels)
    np.save(os.path.join(data_root, 'val_images.npy'), val_images)
    np.save(os.path.join(data_root, 'val_labels.npy'), val_labels)
    np.save(os.path.join(data_root, 'test_images.npy'), test_images)
    np.save(os.path.join(data_root, 'test_labels.npy'), test_labels)

    # Split testing dataset into testing and template dataset
    test_images, test_labels, template_images, template_labels = choose_template(
        test_images, test_labels, template_nums=template_nums)

    np.save(os.path.join(data_root, 'test_images_split.npy'), test_images)
    np.save(os.path.join(data_root, 'test_labels_split.npy'), test_labels)
    np.save(os.path.join(data_root, 'template_images.npy'), template_images)
    np.save(os.path.join(data_root, 'template_labels.npy'), template_labels)
    print('Total: %d, Train: %d, Val: %d, Test: %d, Template: %d' %
          (len(batchs) + len(test_images) + len(template_images),
           len(train_images), len(val_images), len(test_images),
           len(template_images)))
    print(f'Ignore Labels: {ignore_labels}')
    print(
        f'Class Num: {class_num}, Train Class Num: {train_class_num}, Test Class Num: {test_class_num}'
    )
    print('Means: {}'.format(means))
    print('sedd: {}'.format(seed))

    debug_datasets(train_images,
                   train_labels,
                   label_dicts,
                   dataset_type,
                   debug_count=100)
