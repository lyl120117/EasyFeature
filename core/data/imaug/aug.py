import cv2
import numpy as np


class RandomCrop:

    def __init__(self, padding=2, prob=0.5):
        self.padding = padding
        self.prob = prob

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('img should be numpy.ndarray')
        if np.random.rand() < self.prob:
            return img
        img = img.copy()
        h, w = img.shape[:2]
        img = cv2.copyMakeBorder(img,
                                 self.padding,
                                 self.padding,
                                 self.padding,
                                 self.padding,
                                 cv2.BORDER_CONSTANT,
                                 value=0)
        start_w = np.random.randint(0, self.padding * 2)
        start_h = np.random.randint(0, self.padding * 2)
        return img[start_h:start_h + h, start_w:start_w + w]


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('img should be numpy.ndarray')
        if np.random.rand() < self.prob:
            return img
        img = img.copy()
        return cv2.flip(img, 1)

