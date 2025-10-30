
import math
from typing import Dict, Iterable, List, Optional, Tuple


import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode





class DetectionCompose:
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class DetectionToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class DetectionRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = image.flip(-1)
            w = image.shape[-1]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
                flipped_boxes = torch.stack([w - xmax, ymin, w - xmin, ymax], dim=1)
                target["boxes"] = flipped_boxes
        return image, target


class DetectionNormalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class DetectionResize:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        orig_height, orig_width = image.shape[-2:]
        scale = self.min_size / min(orig_height, orig_width)
        if orig_height * scale > self.max_size:
            scale = self.max_size / orig_height
        if orig_width * scale > self.max_size:
            scale = min(scale, self.max_size / orig_width)

        if not math.isclose(scale, 1.0, rel_tol=1e-3):
            new_height = int(round(orig_height * scale))
            new_width = int(round(orig_width * scale))
            image = F.resize(image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes * torch.tensor([scale, scale, scale, scale], dtype=boxes.dtype, device=boxes.device)
                target["boxes"] = boxes
        return image, target



def build_detection_transforms(cfg, train: bool):
    mean = cfg.detection.image_mean
    std = cfg.detection.image_std
    transforms: List = [DetectionToTensor()]
    if train and cfg.detection.random_horizontal_flip > 0:
        transforms.append(DetectionRandomHorizontalFlip(cfg.detection.random_horizontal_flip))
    if cfg.detection.resize_images:
        transforms.append(DetectionResize(cfg.detection.min_size, cfg.detection.max_size))
    transforms.append(DetectionNormalize(mean, std))
    return DetectionCompose(transforms)




