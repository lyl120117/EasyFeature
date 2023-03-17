# EasyFeature
The EasyFeature repo aims to use softmax-based metric learning methods for feature learning, including L-Softmax, ArcFace, and EucMargin, which is the proposed method. Experiments on the MNIST, CIFAR10, and CIFAR100 datasets show that EucMargin has good performance and generalization ability, particularly outstanding on the CIFAR100 dataset. 


# Install
To install the required packages, run the following commands:
```
pip3 install torch torchvision
pip3 install requirements.txt
```

# Support Algorithms
- [x] [L-Softmax](https://arxiv.org/abs/1612.02295)
- [x] [ArcFace](https://arxiv.org/abs/1801.07698)
- [x] [EucMargin(Ours)](./docs/EucMargin.md)

# Results
The following table shows the classification error rate achieved by each method on three datasets: MNIST, CIFAR-10, and CIFAR-100.

| Datasets | L-Softmax | ArcFace | EucMargin |
| :---: | :-------: | :-----: | :----------: |
| MNIST | 0.31% | 0.31% | **0.28%** |
| CIFAR-10 | 7.58% | 7.90% | **7.38%** |
| CIFAR-100 | 29.53% | 30.23% | **28.42%** |


# Usage
To reproduce the results, follow the steps below:

1. Download the dataset and split it into train/val/test/template:

```
python tools/split_datasets.py -dt CIFAR100 --seed 30673
```

2. Train the model

```
# CIFAR-100 EucMargin
python tools/train.py --c configs/cifar100/cifar100_euc.yaml -o Global.seed=49504 Architecture.Head.margin_add=2.5

# CIFAR-100 ArcFace
python tools/train.py --c configs/cifar100/cifar100_arc.yaml

# CIFAR-100 L-Softmax
python tools/train.py --c configs/cifar100/cifar100_lsoftmax.yaml
```
Note that L-Softmax is hard to train on CIFAR-100, so you may need to try different hyperparameters to obtain better results.

## License
This project is released under the [Apache 2.0 license](./LICENSE).
