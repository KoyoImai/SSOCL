

from models.resnet_ours import ResNetProjectorHead
from models.resnet_cifar_empssl import ResNet_Projection



def make_model(cfg):

    model = None


    if cfg.method.name in ["ours"]:
        
        model  = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
        model2 = ResNetProjectorHead(name=cfg.model.type, seed=cfg.seed, cfg=cfg)
        
    else:
        assert False






    return model, model2