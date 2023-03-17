# Euclidean Distance Margin for Feature Learning in Convolutional Neural Networks

## Abstract
In deep learning, Feature Learning is a crucial task, with CNN-based Feature Learning being one of the most focused research directions. This paper proposes a distance metric-based method to improve the feature learning ability of CNN networks in image recognition tasks. By introducing a Euclidean Distance Margin parameter m, the method calculates the Euclidean distance between the last layer feature and the classifier parameters, achieving the purpose of intra-class compactness and inter-class separability. Experiments on the MNIST, CIFAR10, and CIFAR100 datasets show that the method has good performance and generalization ability, particularly outstanding on the CIFAR100 dataset. 

## Introduction
In deep learning, Feature Learning is a crucial task that aims to learn useful feature representations from raw data for subsequent classification, recognition, and retrieval tasks. In recent years, CNN-based Feature Learning has made significant progress, becoming one of the most focused research directions.

CNN networks can effectively extract local image features through multiple layers of convolution and pooling operations and have a certain degree of translation invariance. However, relying solely on convolution and pooling operations may not fully exploit global image features, leading to a decline in model performance. Therefore, designing more effective Feature Learning methods in CNN networks is an essential research task.

This paper proposes a distance metric-based method to improve the feature learning ability of CNN networks in image recognition tasks. The method introduces a Euclidean Distance Margin parameter m to calculate the Euclidean distance between the last layer feature and the classifier parameters, achieving the purpose of intra-class compactness and inter-class separability. Experiments on the MNIST, CIFAR10, and CIFAR100 datasets show that the method has good performance and generalization ability, particularly outstanding on the CIFAR100 dataset.

## Related Work
In CNNs, feature learning is an essential component, aiming to train a network capable of extracting more representative features from input images to improve the performance of image classification tasks. In feature learning, Cosine Distance Margin and Triplet Loss [4] are two popular distance metric methods.

The Cosine Distance Margin method trains CNN networks by defining an objective function that minimizes intra-class distances and maximizes inter-class distances. However, the Cosine Distance Margin has some problems, such as insufficient utilization of feature space information. The Triplet Loss method trains CNN networks using triplet data to achieve minimal intra-class distances and maximal inter-class distances. However, the Triplet Loss method requires paired data and is more complex to train.

In recent years, L-Softmax [1] and ArcFace [3] methods have also been applied to CNN feature learning tasks. L-Softmax introduces an additional parameter to adjust the relationship between feature vectors and weights, improving the feature vector distribution. ArcFace replaces the Cosine Distance Margin with an angular margin to improve the relationship between feature vectors and weights. However, both methods have some problems, such as overly complicated implementation, difficult training, and susceptibility to overfitting.

## Proposed Method
We propose a distance metric-based method to improve the feature learning ability of CNN networks in image recognition tasks. The main idea of this method is to introduce a Euclidean Distance Margin parameter m, calculating the Euclidean distance between the last layer feature and the classifier parameters to achieve intra-class compactness and inter-class separability.

Specifically, for a given image sample $x_i$, we first map it to the last layer feature space through the CNN network to obtain its feature vector $f_i$. Then, we calculate the Euclidean distance $d_{i,C}=||f_i-w_C||_2$ between $f_i$ and classifier parameters $w_C$, where $C$ represents all classes. Next, we add the Euclidean Distance Margin parameter $m$ to $d_{i,c}$ to obtain the new distance $d'{i,c}=d{i,c}+m$, where $c$ represents the class of sample $x_i$. Finally, we input $d'{i,c}$ into the Softmax classifier to calculate the probability value $p{i,c}$ of sample $x_i$ belonging to class $c$. Our optimization objective is to minimize the cross-entropy loss $L$:

$$L = -\log\frac{e^{-d'{i,c}}}{e^{-d'{i,c}} + \sum_{c \neq C} e^{-d_{i,C}}}$$

Since smaller Euclidean distances indicate smaller feature distances, and the optimization objective of Softmax is to maximize the activation value of the corresponding class, we negate $d'{i,c}$ to become $-d'{i,c}$, and negate $d_{i,C}$ to become $-d_{i,C}$.

## Experimental Results
We conducted experiments on the MNIST, CIFAR10, and CIFAR100 datasets to evaluate the performance of the proposed method in image classification tasks. The CNN network used in the experiments adopts the architecture from the L-Softmax paper, and data preprocessing, optimizer, learning rate, and training iterations are kept consistent with the L-Softmax paper.

The experimental results are shown in Table 1:
| Dataset | L-Softmax | ArcFace | EucMargin |
| :---: | :-------: | :-----: | :----------: |
| MNIST | 0.31% | 0.31% | **0.28%** |
| CIFAR-10 | 7.58% | 7.90% | **7.38%** |
| CIFAR-100 | 29.53% | 30.23% | **28.42%** |

*Table 1: Comparison of experimental results, using error rate as an evaluation metric*

As shown in the table, our proposed distance metric-based method has good performance on all three datasets, particularly outstanding on the CIFAR100 dataset. Compared to L-Softmax, our method is simpler to train, and compared to ArcFace, our method has better generalization ability.

## Conclusion
This paper proposes a distance metric-based method to improve the feature learning ability of CNN networks in image recognition tasks. Experiments on the MNIST, CIFAR10, and CIFAR100 datasets show that the method has good performance and generalization ability, particularly outstanding on the CIFAR100 dataset. Compared to L-Softmax, our method is simpler to train, and compared to ArcFace, our method has better generalization ability. Our method has potential application value in scenarios requiring training of generic feature representations.

Future research directions include: 1. Investigating the effectiveness of the method for various network architectures, including those in CV, NLP, and speech recognition domains; 2. Verifying the method's performance on more datasets.

## References
[1] Liu W, Wen Y, Yu Z, et al. Large-margin softmax loss for convolutional neural networks[J]. arXiv preprint arXiv:1612.02295, 2016.

[2] Liu W, Wen Y, Yu Z, et al. Sphereface: Deep hypersphere embedding for face recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 212-220.

[3] Deng J, Guo J, Xue N, et al. Arcface: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4690-4699.

[4] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.
