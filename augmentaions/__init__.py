

from augmentaions.multicrop_generator import MultiCropViewGenerator


def make_transform(cfg):


    
    if cfg.method.name in ['ours']:

        transform = MultiCropViewGenerator(cfg=cfg, num_crops=cfg.method.num_crops)
    
    else:
        
        assert False

    
    return transform
    