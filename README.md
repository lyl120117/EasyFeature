# EasyFeature
The repo aims to using softmax based metric learning methods for feature learning. Such as L-Softmax, SphereFace, ArcFace and AdaFace


# Install
```
pip3 install torch torchvision

```

# Support Algorithms
- [x] [L-Softmax](https://arxiv.org/abs/1612.02295)
- [ ] [SphereFace](https://arxiv.org/abs/1704.08063)
- [ ] [ArcFace](https://arxiv.org/abs/1801.07698)
- [ ] [AdaFace](https://arxiv.org/abs/2204.00964)

# Results
## MNIST
|     Method     |  Acc   | Acc(Paper) |
| :------------: | :----: | :--------: |
|    Softmax     | 99.68% |   99.60%   |
| L-Softmax(m=2) | 99.69% |   99.68%   |
| L-Softmax(m=3) | 99.69% |   99.69%   |
| L-Softmax(m=4) | 99.69% |   99.69%   |



# Usage
1. Download the MNIST dataset and split it into train/val/test/template
```
python tools/split_datasets.py -dt MNIST --seed 30673
```

2. Train the model, *seed* is used to reproduce the results, you can change it to get different results.
2.1 Softmax
```
python tools/train.py --c configs/mnist/mnist_digits.yaml -o Global.seed=38667
```

2.2 m=2 acc=0.9968
python tools/train.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Architecture.Head.margin=2 Global.save_model_dir=output/mnist/mnist_digits_lsoftmax_m2 Global.seed=57554

m=3  acc=0.9969
python tools/train.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Architecture.Head.margin=3 Global.save_model_dir=output/mnist/mnist_digits_lsoftmax_m3 Global.seed=83024

m=4  acc=9969
python tools/train.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Architecture.Head.margin=4 Global.save_model_dir=output/mnist/mnist_digits_lsoftmax_m4 Global.seed=41403
```

1. Test the model
```
m=1
python tools/test.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Global.pretrained_model=output/mnist/mnist_digits_lsoftmax Global.save_model_dir=./output/mnist/mnist_digits_lsoftmax Global.template_mode=0 

m=2
python tools/test.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Global.pretrained_model=output/mnist/mnist_digits_lsoftmax_m2/best_accuracy.pth Global.save_model_dir=./output/mnist/mnist_digits_lsoftmax_m2 Global.template_mode=0 

m=3
python tools/test.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Global.pretrained_model=output/mnist/mnist_digits_lsoftmax_m3/best_accuracy.pth Global.save_model_dir=./output/mnist/mnist_digits_lsoftmax_m3 Global.template_mode=0 

m=4
python tools/test.py --c configs/mnist/mnist_digits_lsoftmax.yaml -o Global.pretrained_model=output/mnist/mnist_digits_lsoftmax_m4/best_accuracy.pth Global.save_model_dir=./output/mnist/mnist_digits_lsoftmax_m4 Global.template_mode=0 
```

