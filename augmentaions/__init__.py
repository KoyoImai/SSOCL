


import torchvision.transforms as transforms



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
    

def make_transform_eval(cfg):

    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(224 * 256 / float(224))),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
