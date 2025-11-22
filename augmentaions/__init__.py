


import torchvision.transforms as transforms



from augmentaions.multicrop_generator import MultiCropViewGenerator
from augmentaions.simsiam_generator import make_simsiam_view_generator
from augmentaions.detection_transform import build_detection_transforms


def make_transform(cfg):

    if cfg.method.name in ['ours', 'empssl']:

        transform = MultiCropViewGenerator(cfg=cfg, num_crops=cfg.method.num_crops)
    
    elif cfg.method.name in ["minred", "scale"]:

        transform = make_simsiam_view_generator(cfg)
    
    else:
        
        assert False

    
    return transform
    

def make_transform_eval(cfg):

    if cfg.method.name in ["ours", "empssl"]:
        mean=(0.5, 0.5, 0.5)
        std=(0.5, 0.5, 0.5)
        # mean=(0.430, 0.411, 0.296)
        # std=(0.213, 0.156, 0.143)
    elif cfg.method.name in ["minred"]:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)

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

    return train_transform, val_transform



def make_detection_augmentation(cfg):

    train_augmentation = build_detection_transforms(cfg, train=True)
    test_augmentation = build_detection_transforms(cfg, train=False)

    return train_augmentation, test_augmentation









