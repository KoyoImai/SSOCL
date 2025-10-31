


import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

from torchvision.ops import misc as misc_nn_ops



def build_detection_model(cfg, backbone: nn.Module) -> nn.Module:

    backbone_fpn = resnet_fpn_backbone(
        "resnet50",
        weights=None,
        trainable_layers=cfg.detection.train.trainable_layers,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        returned_layers=None,
        extra_blocks=None,
    )

    # print("backbone: ", backbone)
    # print("backbone_fpn: ", backbone_fpn)
    # print("backbone.state_dict().keys(): ", backbone.state_dict().keys())
    # print("backbone_fpn.body.state_dict().keys(): ", backbone_fpn.body.state_dict().keys())

    missing, unexpected = backbone_fpn.body.load_state_dict(backbone.state_dict(), strict=False)
    # print("missing keys: ", missing)
    # print("unexpectec keys: ", unexpected)

    expected_missing = {"fc.weight", "fc.bias"}
    expected_unexpected = {"fc.weight", "fc.bias"}

    if set(missing) - expected_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading backbone: {missing}"
        )
    if set(unexpected) - expected_unexpected:
        raise RuntimeError(
            f"Unexpected keys found when loading backbone: {unexpected}"
        )

    model = FasterRCNN(
        backbone=backbone_fpn,
        num_classes=cfg.detection.dataset.num_classes,
        image_mean=cfg.detection.train.image_mean,
        image_std=cfg.detection.train.image_std,
    )

    return model


