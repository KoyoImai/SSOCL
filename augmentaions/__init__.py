

from augmentaions.multicrop_generator import MultiCropViewGenerator
from augmentaions.simsiam_generator import make_simsiam_view_generator


def make_transform(cfg):


    
    if cfg.method.name in ['ours', 'empssl']:

        transform = MultiCropViewGenerator(cfg=cfg, num_crops=cfg.method.num_crops)
    
    elif cfg.method.name in ["minred"]:

        transform = make_simsiam_view_generator(cfg)
    
    else:
        
        assert False

    
    return transform
    