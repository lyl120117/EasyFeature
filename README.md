# EasyFeature
The repo aims to using softmax based metric learning methods for feature learning. Such as L-Softmax ArcFace


# Install
```
pip3 install torch torchvision

```

# Usage
1. Download the MNIST dataset
```
python tools/split_datasets.py -dt MNIST
```

2. Train the model
```
python3 train.py
```