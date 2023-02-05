import torch
import numpy as np
from PIL import Image
import cv2


class ToTensor(object):

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image))
        else:
            raise TypeError('image should be numpy.ndarray or PIL.Image')
        return image


class Normalize(object):

    def __init__(self, mean, std, scale=1.0 / 255.0):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError('image should be np.ndarray')
        image = image.copy()
        image = image.astype(np.float32)

        image *= self.scale
        c, _, _ = image.shape
        for i in range(c):
            image[i] -= self.mean[i]
            image[i] /= self.std[i]
        return image


class ClsResize(object):

    def __init__(self, size, interpolation='CUBIC'):
        interpolations = {
            'CUBIC': cv2.INTER_CUBIC,
            'LINEAR': cv2.INTER_LINEAR,
            'NEAREST': cv2.INTER_NEAREST
        }
        assert (interpolation in interpolations
                ), f"interpolation mode must be one of {interpolations}"
        self.size = size
        self.interpolation = interpolations[interpolation]

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('img should be numpy.ndarray')
        if self.size[0] == img.shape[0]:
            return img
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        return img


class ToCHWImage(object):

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError('image should be np.ndarray')
        image = image.copy()
        if image.ndim == 2:
            image = image[:, :, None]
        image = image.transpose(2, 0, 1)
        return image