

from typing import Dict, Iterable, List, Optional, Tuple


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection


class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir: str, ann_file: str, transforms=None):
        self.dataset = CocoDetection(image_dir, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image, annotations = self.dataset[idx]
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for anno in annotations:
            # Ignore annotations with no bounding box (area == 0)
            if anno.get("bbox") is None:
                continue
            x, y, w, h = anno["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(anno.get("category_id", 0))
            areas.append(anno.get("area", w * h))
            iscrowd.append(anno.get("iscrowd", 0))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        areas_tensor = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(self.dataset.ids[idx], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.dataset)




