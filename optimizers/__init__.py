

import torch
import torch.nn as nn
import torch.optim as optim


from optimizers.lars import LARSWrapper



def make_optimizer(cfg, model):

    optimizer = None

    if cfg.method.name in ["ours", "minred"]:

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


    else:

        assert False
        


    return optimizer