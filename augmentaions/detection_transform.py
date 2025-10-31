from __future__ import annotations

from dataclasses import dataclass
from typing import List, MutableMapping, Sequence, Tuple

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

BoxTensor = torch.Tensor
TargetDict = MutableMapping[str, torch.Tensor]


@dataclass
class _Transform:
    """Callable transform base class.

    Declared purely for typing/IDE help and to highlight the expected
    ``(image, target)`` signature shared across all augmentations in the
    detection pipeline.
    """

    def __call__(self, image: torch.Tensor, target: TargetDict) -> Tuple[torch.Tensor, TargetDict]:
        raise NotImplementedError


class Compose:
    """Compose multiple detection transforms together."""

    def __init__(self, transforms: Sequence[_Transform]):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(_Transform):
    """Convert a PIL image (or numpy array) to a float tensor."""

    def __call__(self, image, target):
        # ``torch.nn.functional`` does not implement ``to_tensor``.  Using
        # torchvision's functional implementation keeps behaviour identical to
        # the rest of the pipeline and fixes AttributeError crashes.
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlip(_Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = image.flip(-1)
            if "boxes" in target:
                boxes = target["boxes"]
                if boxes.numel() > 0:
                    width = image.shape[-1]
                    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
                    flipped = torch.stack([width - xmax, ymin, width - xmin, ymax], dim=1)
                    target["boxes"] = flipped
        return image, target


class Resize(_Transform):
    """Resize images while preserving aspect ratio.

    The bounding boxes are scaled by the same factor so that they keep
    alignment with the resized image.
    """

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: torch.Tensor, target: TargetDict):
        height, width = image.shape[-2:]
        scale = float(self.min_size) / min(height, width)

        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

        if max(new_height, new_width) > self.max_size:
            ratio = float(self.max_size) / max(new_height, new_width)
            new_height = int(round(new_height * ratio))
            new_width = int(round(new_width * ratio))

        if new_height != height or new_width != width:
            image = TF.resize(
                image,
                [new_height, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            if "boxes" in target:
                boxes = target["boxes"]
                if boxes.numel() > 0:
                    scale_x = new_width / width
                    scale_y = new_height / height
                    scale_tensor = torch.tensor([
                        scale_x,
                        scale_y,
                        scale_x,
                        scale_y,
                    ], dtype=boxes.dtype, device=boxes.device)
                    target["boxes"] = boxes * scale_tensor
        return image, target


class Normalize(_Transform):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: torch.Tensor, target: TargetDict):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target





def build_detection_transforms(cfg, train: bool = True) -> Compose:
    """Factory to mirror the previous detection augmentation pipeline."""

    mean = cfg.detection.train.image_mean
    std = cfg.detection.train.image_std

    transforms: List[_Transform] = [ToTensor()]

    if train and getattr(cfg.detection, "random_horizontal_flip", 0) > 0:
        transforms.append(RandomHorizontalFlip(cfg.detection.train.random_horizontal_flip))

    if getattr(cfg.detection, "resize_images", False):
        transforms.append(Resize(cfg.detection.train.min_size, cfg.detection.train.max_size))

    # transforms.append(Normalize(mean, std))
    return Compose(transforms)




# def build_detection_transforms(cfg, train: bool):

#     mean = cfg.detection.train.image_mean
#     std = cfg.detection.train.image_std

#     transforms: List = [DetectionToTensor()]

#     if train and cfg.detection.train.random_horizontal_flip > 0:
#         transforms.append(DetectionRandomHorizontalFlip(cfg.detection.train.random_horizontal_flip))
#     if cfg.detection.train.resize_images:
#         transforms.append(DetectionResize(cfg.detection.train.min_size, cfg.detection.train.max_size))

#     transforms.append(DetectionNormalize(mean, std))
    
#     return DetectionCompose(transforms)




