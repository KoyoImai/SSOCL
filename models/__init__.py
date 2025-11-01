import os
from typing import Dict, Tuple

import torch
from torch import nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP


from models.linear_classifier import LinearClassifier






def make_model(cfg, use_ddp=True):

    model = None


    if cfg.method.name in ["ours"]:

        from models.resnet_ours import ResNetProjectorHead
        
        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
    
    elif cfg.method.name in ["minred"]:

        from models.resnet_simsiam import ResNetProjectorHead
        
        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, dim=cfg.model.simsiam_dim,
                                     pred_dim=cfg.model.pred_dim, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, dim=cfg.model.simsiam_dim,
                                     pred_dim=cfg.model.pred_dim, cfg=cfg)

    elif cfg.method.name in ["empssl"]:

        from models.resnet_empssl import ResNetProjectorHead

        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)

    else:
        assert False
    


    # DDP で ラッピング
    if use_ddp:
        model = DDP(model.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])
        model2 = DDP(model2.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])
    else:
        # model.to(cfg.device)
        # model2.to(cfg.device)
        model.cuda()
        model2.cuda()


    return model, model2


def make_classifier(cfg):

    if cfg.model.type == "resnet50":
        classifier = LinearClassifier(num_classes=cfg.dataset.num_classes,
                                      feat_dim=2048,
                                      seed=cfg.seed)
    else:
        assert False

    if torch.cuda.is_available():
        classifier.cuda()

    return classifier



def make_backbone(cfg):

    from models.resnet_detection import Backbone

    backbone = Backbone(name=cfg.model.type, seed=cfg.seed, cfg=cfg)


    return backbone




def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint parameter names for torchvision compatibility."""

    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Remove the leading "module." prefix used by DDP checkpoints.
        if key.startswith("module."):
            key = key[7:]

        # ``ResNetProjectorHead`` stores intermediate downsample blocks under
        # ``i_downsample`` while torchvision's reference implementation uses
        # ``downsample``.  Remap these keys so we can load the encoder weights
        # into ``torchvision`` models without triggering missing/unexpected key
        # errors.
        if ".i_downsample." in key:
            key = key.replace(".i_downsample.", ".downsample.")

        if "head" in key:
            continue

        cleaned_state_dict[key] = value

    return cleaned_state_dict

# 物体検出やセグメンテーションで使用する backbone の学習ずみパラメータを読み込む
def load_pretrained_resnet_backbone(cfg, backbone) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:

    # 事前学習済みパラメータのパス
    ckpt = cfg.detection.ckpt
    model_path = cfg.log.model_path
    checkpoint_path = os.path.join(model_path, ckpt)
    print("checkpoint_path: ", checkpoint_path)

    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _clean_state_dict(state_dict)

    backbone.load_state_dict(state_dict, strict=True)
    encoder_state_dict = backbone.encoder.state_dict()
    
    tv_resnet = torchvision.models.resnet50(weights=None)

    missing, unexpected = tv_resnet.load_state_dict(encoder_state_dict, strict=False)
    # print("missing keys:", missing)
    # print("unexpected keys:", unexpected)
    # print("len(missing keys):", len(missing))
    # print("len(unexpected keys):", len(unexpected))

    expected_missing = {"fc.weight", "fc.bias"}
    if set(missing) - expected_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading encoder weights: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys found when loading encoder weights: {unexpected}"
        )


    return tv_resnet, encoder_state_dict










