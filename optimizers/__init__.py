

import torch
import torch.nn as nn
import torch.optim as optim


from optimizers.lars import LARSWrapper



def make_optimizer(cfg, model):

    optimizer = None

    if cfg.method.name in ["ours", "empssl"]:

        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.train.learning_rate,
            momentum=cfg.optimizer.train.momentum,
            weight_decay=cfg.optimizer.train.weight_decay,
        )

        optimizer = LARSWrapper(
            optimizer=optimizer,
            eta=cfg.optimizer.train.eta,
            clip=True,
            exclude_bias_n_norm=True
        )

    elif cfg.method.name in ["minred"]:
        
        if cfg.model.fix_pred_lr:
            optim_params = [{
                'params': model.module.encoder.parameters() if int(cfg.ddp.world_size) > 0 else model.encoder.parameters(),
                'fix_lr': False
            },
            {
                "params": model.module.projector.parameters() if int(cfg.ddp.world_size) > 0 else model.projecotr.parameters(),
                "fix_lr": False
            }, 
            {
                'params': model.module.predictor.parameters() if int(cfg.ddp.world_size) > 0 else model.predictor.parameters(),
                'fix_lr': True
            }]
        else:
            optim_params = model.parameters()

        optimizer = optim.SGD(
            optim_params,
            lr=cfg.optimizer.train.learning_rate,
            momentum=cfg.optimizer.train.momentum,
            weight_decay=cfg.optimizer.train.weight_decay,
        )

        optimizer = LARSWrapper(
            optimizer=optimizer,
            eta=cfg.optimizer.train.eta,
            clip=True,
            exclude_bias_n_norm=True
        )

    else:

        assert False
    




    return optimizer