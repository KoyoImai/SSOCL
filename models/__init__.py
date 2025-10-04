


from models.resnet_cifar_empssl import ResNet_Projection



def make_model(cfg):

    if (cfg.model.type == "resnet18_empssl") and (cfg.dataset.type in ["cifar10", "cifar100"]):

        model = ResNet_Projection(name='resnet50', seed=777, cfg=cfg)
    
    else:
        model = None
        assert False

    print("model: ", model)






    return model