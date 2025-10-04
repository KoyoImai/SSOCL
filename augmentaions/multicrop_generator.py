
import numpy as np

import torch
import torch.nn as nn


import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from PIL import Image, ImageFilter, ImageOps




class Solarization:

    def __call__(self, img: Image) -> Image:

        return ImageOps.solarize(img)

class GBlur(object):

    def __init__(self, p):

        self.p = p

    def __call__(self, img):

        if np.random.rand() < self.p:

            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class MultiCropViewGenerator(object):

    def __init__(self, num_crops=4):

        self.num_crops = num_crops

    
    def __call__(self, img):

        # mean=(0.430, 0.411, 0.296)
        # std=(0.213, 0.156, 0.143)
        mean=(0.5, 0.5, 0.5)
        std=(0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=mean, std=std)


        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.data_insize, scale=(0.25, 0.25), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize,
        ])

        transformed_x = [transform(img) for i in range(self.num_crops)]


        return transformed_x








