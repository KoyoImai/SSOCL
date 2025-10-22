

import torch
from torch.nn.parallel import DistributedDataParallel as DDP







def make_model(cfg):

    model = None


    if cfg.method.name in ["ours"]:

        from models.resnet_ours import ResNetProjectorHead
        
        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)

        model = DDP(model.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])
        model2 = DDP(model2.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])
    
    elif cfg.method.name in ["minred"]:

        from models.resnet_simsiam import ResNetProjectorHead
        
        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, dim=cfg.model.simsiam_dim,
                                     pred_dim=cfg.model.pred_dim, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, dim=cfg.model.simsiam_dim,
                                     pred_dim=cfg.model.pred_dim, cfg=cfg)

        model = DDP(model.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])
        model2 = DDP(model2.to(cfg.ddp.local_rank), device_ids=[cfg.ddp.local_rank])

    elif cfg.method.name in ["empssl"]:

        assert False

    else:
        assert False






    return model, model2