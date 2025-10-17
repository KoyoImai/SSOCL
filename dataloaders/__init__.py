


from dataloaders.imagenet21k import ImageNet21K
from dataloaders.stream_sampler import StreamSampler



def make_dataset(cfg, transform):

    if cfg.dataset.type == "imagenet21k":
        
        # def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True):

        train_dataset = ImageNet21K(cfg=cfg,
                                    transform=transform,
                                    filelist=cfg.dataset.filelist,
                                    num_task=cfg.continual.n_task,
                                    train=True)
        
    else:
        
        assert False

    

    return train_dataset



def make_sampler(cfg, dataset):

    sampler = StreamSampler(cfg=cfg, dataset=dataset, rank=cfg.ddp.local_rank, world_size=cfg.ddp.world_size,
                            drop_last=cfg.dataset.drop_last, base_seed=cfg.seed, sharding=cfg.dataset.sharding, start_index=cfg.dataset.start_index)




    assert False

    return sampler





